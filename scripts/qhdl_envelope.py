"""Evidence envelope, tier discipline, stdout limits and stable addresses for the qhdl CLI.
Standard library only, so it can be imported by any verb module
(including a later `surgery` verb) and tested in the stdlib-only project environment.

Envelope (every verb prints exactly one, as JSON):
    {claim, tier, domain, evidence, rerun, tool_sha, circuit_sha, [claim_id, provenance, scope, addresses,] basis,
     parts, facts, rows, rows_shown, rows_total, next}

QHDL v1 Phase 0 stamps (qhdl.main sets them; see scripts/qhdl_investigation.py): `claim_id` = sha12(verb, rerun
without env prefix, tool_sha, circuit_sha); `provenance` computed | recorded | derived; `scope` {domain, extent,
branches} on certify verbs; `addresses` the tag:/op@ inputs as resolved. They are dropped from stdout (never from the
trail) before any row, fact or next command would be.

`parts` (optional): a claim that mixes tiers lists each part with its own tier and basis, e.g.
    [{part: 'value and information test', tier: 'exact-all-x', basis: ...},
     {part: 'recompute search', tier: 'sampled(4000)', basis: ...}].
Each part's tier must be one the verb earned (exactly, not just below the best), so a heuristic part can never ride
on an exact one. `tier` stays the tier of the headline claim.

Tiers, lowest to highest:  static < sampled(n) < exact-all-x < engine-parity < official-local < official-hosted.
A verb records what it actually computed in a `Ledger` (`ledger.earn(tier, basis)`); `Envelope` refuses (TierError)
to print a tier above the best one earned, and refuses rows whose own `tier` field is above it.

Stdout discipline: by default at most MAX_ROWS rows and MAX_BYTES bytes; the full table goes to a file whose path is
in `evidence` / `table`. `--all` prints every row. Output is deterministic: fixed key order, no timestamps.

Addresses (stable for a given circuit sha; a rewritten circuit gets a new sha and new addresses):
    op@<sha12>:<index>   wire q23   interval q23#2   stage <name>   tag <path>   run R17   cut @17552
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

MAX_ROWS = 20
MAX_BYTES = 2048
STAMP_KEYS = ('claim_id', 'provenance', 'scope', 'addresses')  # drop order: last first
PROVENANCE = ('computed', 'recorded', 'derived')

TIER_ORDER = ['static', 'sampled', 'exact-all-x', 'engine-parity', 'official-local', 'official-hosted']
_SAMPLED = re.compile(r'^sampled\((\d+)\)$')


class TierError(ValueError):
    """A verb tried to print a tier it did not compute."""


def parse_tier(tier: str):
    """'sampled(4000)' -> ('sampled', 4000); 'exact-all-x' -> ('exact-all-x', None). Unknown tiers raise."""
    m = _SAMPLED.match(tier)
    if m:
        return 'sampled', int(m.group(1))
    if tier == 'sampled':
        raise TierError("tier 'sampled' needs its sample size: sampled(n)")
    if tier not in TIER_ORDER:
        raise TierError(f'unknown tier {tier!r}; use one of static, sampled(n), exact-all-x, engine-parity, '
                        'official-local, official-hosted')
    return tier, None


def tier_rank(tier: str) -> int:
    return TIER_ORDER.index(parse_tier(tier)[0])


class Ledger:
    """What a verb actually computed (or checked on disk), in order. Each entry: tier, one-line basis."""

    def __init__(self):
        self.entries = []

    def earn(self, tier: str, basis: str):
        parse_tier(tier)
        self.entries.append((tier, basis))
        return tier

    def best(self) -> str | None:
        if not self.entries:
            return None
        return max(self.entries, key=lambda e: tier_rank(e[0]))[0]

    def basis_for(self, tier: str) -> str:
        for t, b in self.entries:
            if t == tier:
                return b
        return ''

    def allows(self, tier: str) -> bool:
        b = self.best()
        return b is not None and tier_rank(tier) <= tier_rank(b)


def file_sha(path, n=None) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 22), b''):
            h.update(chunk)
    d = h.hexdigest()
    return d[:n] if n else d


def files_sha(paths, n=12) -> str:
    """Combined SHA-256 of several files (name + content), order-independent of the caller's ordering."""
    h = hashlib.sha256()
    for p in sorted(str(p) for p in paths):
        h.update(Path(p).name.encode() + b'\0')
        h.update(Path(p).read_bytes())
    return h.hexdigest()[:n]


def dumps(obj) -> str:
    """Deterministic compact JSON (keys in insertion order, which callers fix)."""
    return json.dumps(obj, ensure_ascii=False, separators=(', ', ': '), default=str)


class Envelope:
    FIELDS = ('claim', 'tier', 'domain', 'evidence', 'rerun', 'tool_sha', 'circuit_sha')

    def __init__(self, ledger: Ledger, claim: str, domain: str, evidence: str | None, rerun: str, tool_sha: str,
                 circuit_sha: str | None, tier: str | None = None, facts: dict | None = None, rows=None,
                 table: str | None = None, next_cmds=None, parts=None):
        best = ledger.best()
        if best is None:
            raise TierError('nothing was computed; no tier can be printed')
        if tier is None:
            tier = best
        if tier_rank(tier) > tier_rank(best):
            raise TierError(f'claim tier {tier} is above what this verb computed ({best})')
        for r in list(rows or []) + [facts or {}]:
            rt = r.get('tier') if isinstance(r, dict) else None
            if rt is not None and not ledger.allows(rt):
                raise TierError(f'row tier {rt} is above what this verb computed ({best})')
        earned = {t for t, _ in ledger.entries}
        self.parts = []
        for prt in parts or []:
            pt = prt.get('tier')
            if pt is None:
                raise TierError(f"part {prt.get('part')!r} has no tier")
            parse_tier(pt)
            if pt not in earned:
                raise TierError(f"part {prt.get('part')!r} claims tier {pt}, which this verb did not earn "
                                f"(earned: {', '.join(sorted(earned))})")
            self.parts.append(dict(part=prt.get('part', ''), tier=pt, basis=prt.get('basis') or ledger.basis_for(pt)))
        self.d = dict(claim=claim, tier=tier, domain=domain, evidence=evidence, rerun=rerun, tool_sha=tool_sha,
                      circuit_sha=circuit_sha)
        self.basis = ledger.basis_for(tier)
        self.facts = facts or {}
        self.rows = list(rows or [])
        self.table = table
        self.next = list(next_cmds or [])
        self.stamps = {}  # QHDL v1 Phase 0: claim_id, provenance, scope, addresses (set by qhdl.main or a verb)
        self.max_bytes = None  # a verb may widen its own budget (inv resume); None = MAX_BYTES

    def stamp(self, **kw):
        """Set identity fields printed after circuit_sha: claim_id, provenance (computed | recorded | derived),
        scope ({domain, extent, branches}, certify verbs) and addresses (resolved tag/op@ inputs). None = keep."""
        for k in STAMP_KEYS:
            if kw.get(k) is not None:
                self.stamps[k] = kw[k]
        if 'provenance' in self.stamps and self.stamps['provenance'] not in PROVENANCE:
            raise ValueError(f"provenance {self.stamps['provenance']!r}: use one of {', '.join(PROVENANCE)}")
        return self

    def as_dict(self, rows_limit=None, stamps=None):
        out = dict(self.d)
        keep = STAMP_KEYS if stamps is None else stamps
        for k in STAMP_KEYS:
            if k in keep and k in self.stamps:
                out[k] = self.stamps[k]
        out['basis'] = self.basis
        if self.parts:
            out['parts'] = self.parts
        if self.facts:
            out['facts'] = self.facts
        rows = self.rows if rows_limit is None else self.rows[:rows_limit]
        if self.rows:
            out['rows'] = rows
            out['rows_shown'] = len(rows)
            out['rows_total'] = len(self.rows)
            if len(rows) < len(self.rows):
                out['table'] = self.table or self.d['evidence']
        if self.next:
            out['next'] = self.next
        return out

    def render(self, all_rows=False, max_rows=MAX_ROWS, max_bytes=None) -> str:
        """One JSON object: top-level keys one per line, one row per line. Fits max_bytes unless all_rows.

        Identity stamps never cost content: if printing them would drop rows, facts, next commands or part bases
        that fit without them, the optional stamps go first (addresses, scope, provenance, then claim_id); the
        trail keeps them all."""
        if max_bytes is None:
            max_bytes = self.max_bytes or MAX_BYTES
        if all_rows:
            return _render(self.as_dict())
        if not self.stamps:
            return self._render_limited(max_rows, max_bytes, ())
        saved = (list(self.parts), dict(self.facts), list(self.next))

        def attempt(keep):
            self.parts, self.facts, self.next = list(saved[0]), dict(saved[1]), list(saved[2])
            try:
                text = self._render_limited(max_rows, max_bytes, keep)
            except ValueError:
                return None, None
            d = json.loads(text)
            sig = (d.get('rows_shown', 0), 'moved' in (d.get('facts') or {}), len(d.get('next') or []),
                   sum(1 for p in d.get('parts') or [] if p.get('basis')))
            return text, sig
        base_text, base_sig = attempt(())
        keep = [k for k in STAMP_KEYS if k in self.stamps]
        while keep:
            text, sig = attempt(tuple(keep))
            if text is not None and (base_sig is None or sig == base_sig):
                return text
            keep.pop()
        if base_text is None:
            self.parts, self.facts, self.next = list(saved[0]), dict(saved[1]), list(saved[2])
            return self._render_limited(max_rows, max_bytes, ())  # raises the original error
        return base_text

    def _render_limited(self, max_rows, max_bytes, stamps) -> str:
        limit = min(max_rows, len(self.rows))
        if len(_render(self.as_dict(limit, stamps)).encode()) > max_bytes and any(p.get('basis') for p in self.parts):
            self.parts = [dict(part=p['part'], tier=p['tier']) for p in self.parts]  # first to go: part bases
        # Priority when over budget: claim/header, then up to three rows, then next, then facts, then more rows.
        keep = min(3, limit)

        def size(n):
            return len(_render(self.as_dict(n, stamps)).encode())
        while limit > keep and size(limit) > max_bytes:
            limit -= 1
        if size(limit) > max_bytes and self.facts and 'moved' not in self.facts:
            self.facts = {'moved': 'facts exceed the 2 KB stdout budget; see evidence'}
            while limit < min(max_rows, len(self.rows)) and size(limit + 1) <= max_bytes:
                limit += 1
        while self.next and size(limit) > max_bytes:
            self.next = self.next[:-1]
        while limit > 0 and size(limit) > max_bytes:
            limit -= 1
        text = _render(self.as_dict(limit, stamps))
        if len(text.encode()) > max_bytes:
            raise ValueError(f'envelope header alone exceeds {max_bytes} bytes; shorten the claim')
        return text


def _render(d: dict) -> str:
    lines = []
    for k, v in d.items():
        if k == 'rows':
            inner = ',\n  '.join(dumps(r) for r in v)
            lines.append(f'"rows": [\n  {inner}\n ]' if v else '"rows": []')
        else:
            lines.append(f'{json.dumps(k)}: {dumps(v)}')
    return '{' + ',\n '.join(lines) + '}'


def error_text(msg: str, hint: str | None = None) -> str:
    d = {'error': msg}
    if hint:
        d['hint'] = hint
    return dumps(d)


# ---------------------------------------------------------------- addresses
_ADDR = [
    ('op', re.compile(r'^op@([0-9a-f]{12}):(\d+)$')),
    ('cut', re.compile(r'^@(\d+)$')),
    ('interval', re.compile(r'^q(\d+)#(\d+)$')),
    ('wire', re.compile(r'^q(\d+)$')),
    ('run', re.compile(r'^R(\d+)$')),
]


def parse_address(text: str, noun: str | None = None) -> dict:
    """Parse an address. `noun` (wire, interval, run, stage, tag, op, cut) disambiguates bare forms such as '23'."""
    s = text.strip()
    if noun in ('stage', 'tag'):
        if not s or any(c.isspace() for c in s):
            raise ValueError(f'bad {noun} name {text!r}')
        return {'kind': noun, 'name': s}
    if noun == 'wire' and s.isdigit():
        s = 'q' + s
    if noun == 'run' and s.isdigit():
        s = 'R' + s
    if noun in ('cut', 'op') and s.isdigit():
        s = '@' + s
    for kind, rx in _ADDR:
        m = rx.match(s)
        if not m:
            continue
        if kind == 'op':
            return {'kind': 'op', 'sha12': m.group(1), 'index': int(m.group(2))}
        if kind == 'cut':
            return {'kind': 'cut', 'index': int(m.group(1))}
        if kind == 'interval':
            return {'kind': 'interval', 'wire': int(m.group(1)), 'k': int(m.group(2))}
        if kind == 'wire':
            return {'kind': 'wire', 'wire': int(m.group(1))}
        if kind == 'run':
            return {'kind': 'run', 'run': int(m.group(1))}
    raise ValueError(f'unrecognised address {text!r} (forms: op@<sha12>:<i>, q23, q23#2, R17, @17552, '
                     'stage <name>, tag <path>)')


def format_address(a: dict) -> str:
    k = a['kind']
    if k == 'op':
        return f"op@{a['sha12']}:{a['index']}"
    if k == 'cut':
        return f"@{a['index']}"
    if k == 'interval':
        return f"q{a['wire']}#{a['k']}"
    if k == 'wire':
        return f"q{a['wire']}"
    if k == 'run':
        return f"R{a['run']}"
    if k in ('stage', 'tag'):
        return f"{k} {a['name']}"
    raise ValueError(k)


def number_runs(profile, level):
    """Peak runs at width >= level, numbered R0, R1, ... in op order: [(start, end)] inclusive. Pure Python."""
    runs = []
    i, n = 0, len(profile)
    while i < n:
        if profile[i] >= level:
            j = i
            while j < n and profile[j] >= level:
                j += 1
            runs.append((i, j - 1))
            i = j
        else:
            i += 1
    return runs


def number_intervals(intervals):
    """[(wire, start, end, ...)] -> {(wire, start): k} with k counted per wire in start order (q23#k)."""
    per = {}
    for iv in sorted(intervals, key=lambda t: (t[0], t[1])):
        per.setdefault(iv[0], []).append(iv[1])
    return {(w, s): k for w, ss in per.items() for k, s in enumerate(ss)}
