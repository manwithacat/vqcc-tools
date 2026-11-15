"""Run the unchanged official verifiers locally with a locally built kickmix.

Mirrors the hosted invocation of the official verifiers in a Kaggle notebook, so the
results can be compared digit for digit with a hosted run of the reference circuits.
Writes one JSON report.

Usage (with a Python where `import kickmix` resolves to the local build):

    $QHDL_KICKMIX_PYTHON scripts/local_verify_reference.py \
        --root $QHDL_OFFICIAL_ROOT --out $QHDL_OUTPUTS/<new dir>/report.json \
        --tasks dfthcsos invsqrt dqi --workers 2

The verifiers are the official files; this script only wires paths and records
provenance (kickmix module path, its file hash, Python version, wall time).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import tempfile
import time
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


# ---------------------------------------------------------------------------
# FCIDUMP cache
#
# The official chemistry verifier spends about two thirds of its time parsing the 128 MB FCIDUMP in pure Python.
# The cache keeps the arrays that parse produces before expansion (the packed float32 lower triangle of the
# two-electron integrals, plus oei, ecore, NORB, NELEC), keyed by the FCIDUMP's SHA-256, and rebuilds the rest with
# the verifier's own helper. No official file is modified: the runner swaps `load_reference` in the imported
# `dfthcsos.verify` module at run time, and only when a cache directory is given. Any mismatch or load error falls
# back to the official parse.
# ---------------------------------------------------------------------------

FCIDUMP_CACHE_ENV = "VQCC_FCIDUMP_CACHE"
DEFAULT_FCIDUMP_CACHE = Path(__file__).resolve().parents[1] / "outputs" / "fcidump-cache"
FCIDUMP_CACHE_FORMAT = 1


def fcidump_cache_file(cache_dir: Path, fcidump_sha256: str) -> Path:
    """Cache file for an FCIDUMP with this SHA-256 (full 64-hex-digit key; anything else is refused)."""
    key = fcidump_sha256.lower()
    if len(key) != 64 or any(c not in "0123456789abcdef" for c in key):
        raise ValueError(f"not a SHA-256 hex digest: {fcidump_sha256!r}")
    return Path(cache_dir) / f"{key}.npz"


def resolve_fcidump_cache(option, disabled: bool, environ=None):
    """Cache directory to use, or None. `--no-fcidump-cache` wins, then `--fcidump-cache [DIR]`, then the env var."""
    environ = os.environ if environ is None else environ
    if disabled:
        return None
    if option is not None:
        return Path(option)
    value = environ.get(FCIDUMP_CACHE_ENV, "").strip()
    if not value or value.lower() in ("0", "off", "no", "false"):
        return None
    return DEFAULT_FCIDUMP_CACHE if value.lower() in ("1", "on", "yes", "true", "default") else Path(value)


def _payload_sha256(arrays) -> str:
    digest = hashlib.sha256()
    for array in arrays:
        digest.update(str(array.dtype).encode() + str(array.shape).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def make_cached_load_reference(official_load, h2_full_from_lower, cache_dir: Path, events: list):
    """Wrap the verifier's `load_reference(fcidump)` with the SHA-keyed cache.

    `official_load` and `h2_full_from_lower` are the verifier's own functions. Every call hashes the FCIDUMP it is
    given; one event per call is appended to `events` (status hit, miss-written, miss-not-written or fallback, with
    the reason). The rebuild below repeats the last two lines of the official `load_reference` exactly.
    """
    import numpy as np

    cache_dir = Path(cache_dir)

    def rebuild(h2_lower, oei, ecore, norb, nelec):
        tei = h2_full_from_lower(h2_lower, norb)
        h1_thc = np.array(oei - 0.5 * np.einsum("pqqs->ps", tei))
        return {"orbitals": norb, "electrons": nelec, "ecore": ecore, "oei": oei, "tei": tei, "h1_thc": h1_thc}

    def read_cache(path: Path, key: str):
        with np.load(path, allow_pickle=False) as data:
            if int(data["format"]) != FCIDUMP_CACHE_FORMAT:
                raise ValueError(f"cache format {int(data['format'])} != {FCIDUMP_CACHE_FORMAT}")
            if str(data["fcidump_sha256"]) != key:
                raise ValueError("stored FCIDUMP SHA-256 differs from the key")
            h2_lower, oei = np.array(data["h2_lower"]), np.array(data["oei"])
            norb, nelec, ecore = int(data["norb"]), int(data["nelec"]), float(data["ecore"])
            stored = str(data["payload_sha256"])
        pairs = norb * (norb + 1) // 2
        if h2_lower.dtype != np.float32 or h2_lower.shape != (pairs, pairs):
            raise ValueError(f"h2_lower has dtype {h2_lower.dtype} shape {h2_lower.shape}")
        if oei.dtype != np.float64 or oei.shape != (norb, norb):
            raise ValueError(f"oei has dtype {oei.dtype} shape {oei.shape}")
        if _payload_sha256((h2_lower, oei)) != stored:
            raise ValueError("payload SHA-256 mismatch")
        return rebuild(h2_lower, oei, ecore, norb, nelec)

    def write_cache(path: Path, key: str, ref) -> None:
        n = ref["orbitals"]
        rows, cols = np.tril_indices(n)
        lower64 = ref["tei"][rows[:, None], cols[:, None], rows[None, :], cols[None, :]]
        h2_lower = lower64.astype(np.float32)
        if not np.array_equal(h2_lower.astype(np.float64), lower64):
            raise ValueError("tei is not float32-exact; layout differs from the official parser")
        if not np.array_equal(h2_full_from_lower(h2_lower, n), ref["tei"]):
            raise ValueError("packed lower triangle does not re-expand to the official tei")
        oei = np.ascontiguousarray(ref["oei"], dtype=np.float64)
        cache_dir.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=cache_dir, prefix=f".{key[:16]}.", suffix=".tmp")
        try:
            with os.fdopen(fd, "wb") as stream:
                np.savez(stream, format=np.int64(FCIDUMP_CACHE_FORMAT), fcidump_sha256=np.array(key),
                         h2_lower=h2_lower, oei=oei, ecore=np.float64(ref["ecore"]), norb=np.int64(n),
                         nelec=np.int64(ref["electrons"]), payload_sha256=np.array(_payload_sha256((h2_lower, oei))))
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(tmp, path)
        except BaseException:
            Path(tmp).unlink(missing_ok=True)
            raise

    def cached_load_reference(fcidump):
        start = time.monotonic()
        event = {"fcidump": str(fcidump), "cache_dir": str(cache_dir)}
        events.append(event)
        try:
            key = sha256(Path(fcidump))
            path = fcidump_cache_file(cache_dir, key)
            event.update(fcidump_sha256=key, cache_file=str(path))
        except Exception as error:  # noqa: BLE001  (any failure: official parse)
            event.update(status="fallback", reason=f"hash: {type(error).__name__}: {error}")
            ref = official_load(fcidump)
            event["seconds"] = time.monotonic() - start
            return ref
        if path.is_file():
            try:
                ref = read_cache(path, key)
                event.update(status="hit", seconds=time.monotonic() - start)
                return ref
            except Exception as error:  # noqa: BLE001
                event.update(status="fallback", reason=f"read: {type(error).__name__}: {error}")
                ref = official_load(fcidump)
                event["seconds"] = time.monotonic() - start
                return ref
        ref = official_load(fcidump)
        try:
            write_cache(path, key, ref)
            event["status"] = "miss-written"
        except Exception as error:  # noqa: BLE001  (the official result is still returned)
            event.update(status="miss-not-written", reason=f"write: {type(error).__name__}: {error}")
        event["seconds"] = time.monotonic() - start
        return ref

    cached_load_reference.official = official_load
    return cached_load_reference


def install_fcidump_cache(cache_dir: Path, events: list):
    """Swap `load_reference` in the imported official `dfthcsos.verify` module for the cached wrapper."""
    import dfthcsos.verify as verify_module
    import dfthcsos.scoreinfra.reference as reference_module
    official = verify_module.load_reference
    verify_module.load_reference = make_cached_load_reference(
        official, reference_module._h2_full_from_lower, cache_dir, events)
    return official


def load_task(options):
    """Returns (verifier module, circuit, spec) for "invsqrt" or "dqi"; importable in a spawned worker."""
    root, submission, name, _, invsqrt_shots_per_x, dqi_shots = options
    if str(Path(root) / "verifiers") not in sys.path:
        sys.path.insert(0, str(Path(root) / "verifiers"))
    import kickmix as km
    if name == "invsqrt":
        import invsqrt.verify_submission_invsqrt as module
        spec = module.ProblemSpecInvSqrt.for_n_p(10) if invsqrt_shots_per_x is None \
            else module.ProblemSpecInvSqrt.for_n_p(10, shots_per_x=invsqrt_shots_per_x)
    else:
        import dqi.verify_submission_dqi as module
        spec = module.ProblemSpecDQI.benchmark_instance() if dqi_shots is None \
            else module.ProblemSpecDQI.benchmark_instance(num_shots=dqi_shots)
    circuit = km.Circuit.from_file(str(Path(submission) / f"{name}.kmb"), format="kmb")
    return module, circuit, spec


class _Collected(Exception):
    """Carries one spawned worker's task results out of the official verify()."""

    def __init__(self, results):
        super().__init__()
        self.results = results


def _spawned_share(job):
    """Runs the official verify() up to its _run_parallel call and evaluates tasks[k::n] only."""
    options, k, n = job
    module, circuit, spec = load_task(options)

    def run_share(fn, tasks, max_workers):
        raise _Collected([fn(task) for task in tasks[k::n]])

    module._run_parallel = run_share
    try:
        module.verify(circuit, spec, salt=options[3], max_workers=n)
    except _Collected as collected:
        return collected.results
    raise RuntimeError("verify() returned without calling _run_parallel")


def run_spawned(options, workers):
    """Official verify() with its batch tasks spread over spawned processes (no fork needed).

    Each worker reloads the circuit and recomputes the verifier's own task list (seeds derive from
    the circuit hash and salt), runs every n-th task, and the parent then re-enters verify() with
    _run_parallel returning the gathered results in task order, so all aggregation and the final
    checks are the verifier's own code.
    """
    import concurrent.futures
    import multiprocessing as mp
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn")) as pool:
        shares = list(pool.map(_spawned_share, [(options, k, workers) for k in range(workers)]))
    module, circuit, spec = load_task(options)

    def gathered(fn, tasks, max_workers):
        results = [None] * len(tasks)
        for k, share in enumerate(shares):
            results[k::workers] = share
        return results

    module._run_parallel = gathered
    return module.verify(circuit, spec, salt=options[3], max_workers=workers)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, required=True,
                        help="directory holding official verifiers/ and sample_submission/")
    parser.add_argument("--submission", type=Path, default=None,
                        help="submission directory to verify (default: <root>/sample_submission)")
    parser.add_argument("--out", type=Path, required=True, help="JSON report path (must not exist)")
    parser.add_argument("--tasks", nargs="+", default=["dfthcsos", "invsqrt", "dqi"],
                        choices=["dfthcsos", "invsqrt", "dqi"])
    parser.add_argument("--workers", type=int, default=6,
                        help="workers for DQI (≈ 630 MB each, copy-on-write after load; parent peaks ≈ 5 GB; 6 measured 5.0x faster on an M2 Pro, 2026-10-27)")
    parser.add_argument("--invsqrt-workers", type=int, default=8,
                        help="workers for invsqrt (small circuit; 8 measured 2.9x faster on an M2 Pro, 2026-10-26)")
    parser.add_argument("--scheduler", choices=["official", "threads", "spawn"], default="official",
                        help="how the invsqrt/DQI batch tasks are scheduled: official = the verifiers' own fork "
                             "pool (serial where fork is unavailable, e.g. Windows); threads = a thread pool "
                             "(parallel only with a kickmix whose Simulator.do releases the GIL); spawn = "
                             "spawned processes that each reload the circuit and run every k-th task. Results "
                             "cannot depend on this: every task carries its own seeds and order is kept")
    parser.add_argument("--salt", default="")
    parser.add_argument("--dqi-shots", type=int, default=None, help="override DQI shots (default official)")
    parser.add_argument("--invsqrt-shots-per-x", type=int, default=None, help="override (default official 64)")
    parser.add_argument("--fcidump-cache", nargs="?", type=Path, const=DEFAULT_FCIDUMP_CACHE, default=None,
                        metavar="DIR",
                        help="cache the parsed chemistry FCIDUMP keyed by its SHA-256 (bit-identical arrays; "
                             f"default dir {DEFAULT_FCIDUMP_CACHE}). Also enabled by env {FCIDUMP_CACHE_ENV}=DIR or =1")
    parser.add_argument("--no-fcidump-cache", action="store_true", help="official FCIDUMP parse even if the env var is set")
    args = parser.parse_args()

    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite {args.out}")
    args.out.parent.mkdir(parents=True, exist_ok=True)

    root = args.root.resolve()
    submission = (args.submission or root / "sample_submission").resolve()
    sys.path.insert(0, str(root / "verifiers"))
    import kickmix as km  # noqa: E402  (local build, from the venv)
    import numpy as np  # noqa: E402

    kickmix_file = Path(km.__file__)
    report = {
        "purpose": "local kickmix parity check against the hosted official reference audit",
        "python": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "kickmix_module": str(kickmix_file),
        "kickmix_sha256": sha256(kickmix_file),
        "numpy": np.__version__,
        "root": str(root),
        "submission": str(submission),
        "settings": {"salt": args.salt, "seed": "circuit hash derived", "workers": args.workers,
                     "invsqrt_workers": args.invsqrt_workers, "scheduler": args.scheduler},
        "verifier_sha256": {
            name: sha256(root / "verifiers" / name)
            for name in ("invsqrt/verify_submission_invsqrt.py", "dqi/verify_submission_dqi.py", "dfthcsos/verify.py")
        },
        "results": {},
    }

    def save(name: str, summary: dict, elapsed: float) -> None:
        report["results"][name] = {"summary": summary, "wall_seconds": elapsed}
        args.out.write_text(json.dumps(report, indent=2, default=str))
        print(name, "status", summary.get("status"), "score", summary.get("score"),
              "Q", summary.get("num_qubits"), "T", summary.get("toffoli_count"),
              "seconds", round(elapsed, 1), flush=True)

    args.out.write_text(json.dumps(report, indent=2, default=str))
    print("kickmix from", kickmix_file, flush=True)

    if args.scheduler == "threads":
        import concurrent.futures
        import invsqrt.verify_submission_invsqrt as invsqrt_module
        import dqi.verify_submission_dqi as dqi_module

        def run_threaded(fn, tasks, max_workers):
            workers = min(len(tasks), max_workers or 1)
            if workers <= 1:
                return [fn(t) for t in tasks]
            with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
                return list(pool.map(fn, tasks))

        invsqrt_module._run_parallel = run_threaded
        dqi_module._run_parallel = run_threaded

    if "dfthcsos" in args.tasks:
        from dfthcsos.verify import verify as verify_chemistry
        cache_dir = resolve_fcidump_cache(args.fcidump_cache, args.no_fcidump_cache)
        cache_events: list = []
        report["fcidump_cache"] = {"enabled": cache_dir is not None,
                                   "dir": str(cache_dir.resolve()) if cache_dir else None, "events": cache_events}
        if cache_dir is not None:
            install_fcidump_cache(cache_dir, cache_events)
        start = time.monotonic()
        result = verify_chemistry(submission / "dfthcsos", salt=args.salt)
        report["fcidump_cache"]["used"] = any(e.get("status") == "hit" for e in cache_events)
        save("dfthcsos", result, time.monotonic() - start)

    for name, workers in (("invsqrt", args.invsqrt_workers), ("dqi", args.workers)):
        if name not in args.tasks:
            continue
        options = (str(root), str(submission), name, args.salt, args.invsqrt_shots_per_x, args.dqi_shots)
        start = time.monotonic()
        if args.scheduler == "spawn":
            result = run_spawned(options, workers)
        else:
            module, circuit, spec = load_task(options)
            result = module.verify(circuit, spec, salt=args.salt, max_workers=workers)
            del circuit
        save(name, result, time.monotonic() - start)

    print("Saved", args.out)


if __name__ == "__main__":
    main()
