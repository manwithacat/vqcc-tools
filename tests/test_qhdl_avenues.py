"""qhdl avenues: symbolic window floors, repetition up to wire relabelling, pricing arithmetic, negatives.
Synthetic circuits only (written to a temp .kmx and parsed with qhdl_bitsim). NumPy is needed (qhdl_window); the tests
are skipped in the stdlib-only project environment."""
import importlib.util
import math
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
HAS_NUMPY = importlib.util.find_spec("numpy") is not None

HEADER = "".join(f"APPEND_TO_REGISTER q{i} r0\n" for i in range(4)) + 'REGISTER r0 "x"\n'


def load(body):
    import qhdl_bitsim as B
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "c.kmx"
        p.write_text(HEADER + body)
        ops, _hdr, xq = B.parse_kmx(str(p))
    return ops, xq


@unittest.skipUnless(HAS_NUMPY, "NumPy required (system python3)")
class Avenues(unittest.TestCase):
    def setUp(self):
        import qhdl_avenues as A
        import qhdl_window as W
        self.A, self.W = A, W

    def lv(self, ops):
        return self.W.Liveness(ops)

    def test_floor_at_floor_has_gap_zero(self):
        # q10 = a*b, q11 = q10*c: two chained products need two ANDs (degree 3, rank 2)
        ops, xq = load("CCX q0 q1 q10\nCCX q10 q2 q11\nCX q11 q12\n")
        r = self.A.window_floor(ops, self.lv(ops), xq, 0, 2)
        self.assertEqual(r["status"], "exact_anf")
        self.assertEqual((r["toffoli"], r["lb"]), (2, 2))
        self.assertEqual(r["toffoli"] - r["lb"], 0)

    def test_redundant_pair_has_positive_gap(self):
        # the same product computed twice into one wire cancels: no nonlinear output, floor 0, gap 2
        ops, xq = load("CCX q0 q1 q10\nCX q2 q1\nCX q2 q1\nCCX q0 q1 q10\n")
        r = self.A.window_floor(ops, self.lv(ops), xq, 0, 4)
        self.assertEqual((r["toffoli"], r["lb"]), (2, 0))

    def test_fresh_scratch_is_constant_not_input(self):
        ops, xq = load("CCX q0 q1 q10\nCX q10 q11\n")
        st = self.A.SymState(self.lv(ops), 0, xq)
        for o in ops:
            st.apply(o)
        self.assertEqual(st.names, ["q0", "q1"])  # q10, q11 untouched before the window: |0>

    def test_repeated_block_counted_up_to_relabelling(self):
        blk = "CCX q{a} q{b} q{c}\nCX q{c} q{d}\nCCX q{b} q{d} q{c}\n"
        body = blk.format(a=0, b=1, c=10, d=11) + "X q3\n" + blk.format(a=2, b=3, c=12, d=13)
        ops, xq = load(body)
        found = [dict(ops=[0, 3], toffoli=2, lb=1, run=0), dict(ops=[4, 7], toffoli=2, lb=1, run=0)]
        rows = self.A.attach_multiplicity(ops, found)
        self.assertEqual(rows[0]["sig"], rows[1]["sig"])
        self.assertEqual([r["multiplicity"] for r in rows], [2, 2])
        other = [dict(ops=[0, 3], toffoli=2, lb=1, run=0), dict(ops=[3, 4], toffoli=0, lb=0, run=0)]
        self.assertEqual(self.A.attach_multiplicity(ops, other)[0]["multiplicity"], 1)

    def test_scan_windows_sliding_and_term_cap(self):
        ops, xq = load("CCX q0 q1 q10\nCCX q10 q2 q11\nCCX q11 q3 q12\n")
        found, capped, _ = self.A.scan_windows(ops, self.lv(ops), xq, self.A.value_runs(ops), sizes_small=(2, 3),
                                               sizes_large=())
        self.assertEqual(sorted((tuple(f["ops"]), f["lb"]) for f in found), [((0, 2), 2), ((0, 3), 3), ((1, 3), 2)])
        self.assertEqual(capped, [])

    def test_pricing_arithmetic(self):
        charge = 93863.3572624325 / 62
        self.assertAlmostEqual(self.A.gm_effect_pct(1, charge), 0.0220, places=3)  # price of one CCX
        self.assertAlmostEqual(self.A.gm_effect_pct(7, charge), 0.1546, places=3)  # all seven CCXs of the LUT_b gap
        self.assertEqual(self.A.gm_effect_pct(0, charge), 0.0)
        self.assertAlmostEqual(self.A.gm_effect_pct(10, 100.0), 100 * math.log(100 / 90) / 3)
        self.assertAlmostEqual(self.A.cost_after(62, charge, 1), 93863.3572624325 - 62, places=6)

    def test_negatives_match_by_op_range_and_window_size(self):
        led = [dict(id="NR-X", task="invsqrt", outcome="kill", mechanism="wider exact window search (6 and 8 Toffolis per "
                    "window)", family_target="", key_numbers="", why_it_failed="", keywords="window"),
               dict(id="NR-Y", task="invsqrt", outcome="null", mechanism="edit at op 13800 failed", family_target="",
                    key_numbers="", why_it_failed="", keywords="")]
        hit = self.A.match_negatives(led, 100, 200, 6)
        self.assertEqual([h["id"] for h in hit], ["NR-X"])
        self.assertFalse(hit[0]["covers"])  # related, not closed: it names no op range
        self.assertEqual(self.A.match_negatives(led, 100, 200, 12), [])  # larger than the searched windows
        self.assertEqual([h["id"] for h in self.A.match_negatives(led, 13700, 13900, 12)], ["NR-Y"])

    def test_rank_regions_cluster_and_closed(self):
        A = self.A

        def row(a, b, tof, lb, closed=False, mult=1):
            r = dict(ops=[a, b], toffoli=tof, lb=lb, gap=tof - lb, run=0, multiplicity=mult, closed=closed, negatives=[],
                     sig=f"s{a}", sites=[[a, b]])
            r['gm_effect_pct'] = A.gm_effect_pct(r['gap'] * mult, 1500.0)
            return r
        rows = [row(0, 100, 10, 4), row(50, 150, 8, 4), row(120, 200, 5, 4), row(300, 400, 5, 4),
                row(500, 600, 6, 1, closed=True), row(700, 800, 3, 3)]
        o, c, broad, gap0, pos, reps = A.rank(rows)
        # the three overlapping windows merge into one region whose gap is the best window's (6), not a sum
        self.assertEqual([(r["region"], r["gap"]) for r in o], [([0, 200], 6), ([300, 400], 1)])
        self.assertEqual(o[0]["cluster_windows"], 3)
        self.assertEqual([r["ops"] for r in c], [[500, 600]])
        self.assertEqual(gap0, 1)

    def test_repeated_components_listed_at_gap_zero(self):
        A = self.A
        rows = [dict(ops=[0, 10], toffoli=4, lb=4, gap=0, multiplicity=3, sites=[[0, 10], [20, 30], [40, 50]], sig="a"),
                dict(ops=[2, 10], toffoli=3, lb=3, gap=0, multiplicity=3, sites=[[2, 10], [22, 30], [42, 50]], sig="b"),
                dict(ops=[60, 70], toffoli=2, lb=2, gap=0, multiplicity=1, sites=[[60, 70]], sig="c")]
        reps = A.repeated(rows)
        self.assertEqual([r["sig"] for r in reps], ["a"])  # the sub-window b is folded into a; c does not repeat

    def test_negative_closed_only_when_range_named(self):
        led = [dict(id="NR-K", task="invsqrt", outcome="kill", mechanism="tried ops 13714..13969 deletion", family_target="",
                    key_numbers="", why_it_failed="", keywords=""),
               dict(id="NR-R", task="invsqrt", outcome="null", mechanism="wider exact window search (up to 8 Toffolis)",
                    family_target="", key_numbers="", why_it_failed="", keywords="window")]
        hit = {h["id"]: h for h in self.A.match_negatives(led, 13714, 13969, 6)}
        self.assertTrue(hit["NR-K"]["covers"])
        self.assertFalse(hit["NR-R"]["covers"])
        self.assertTrue(hit["NR-R"]["why"].startswith("related"))

    def test_r_bounds_cubic_and_polar(self):
        W = self.W
        # three disjoint cubic monomials: joint derivative rank 9 -> floor 5 (synthetic control)
        monos = [[0b111], [0b111000], [0b111000000]]
        self.assertEqual(W.cubic_derivative_floor(monos)[1:], (9, 5))
        self.assertEqual(W.cubic_derivative_floor([[0b1111]]), (0, 0, 0))  # degree 4: not applicable
        # x0x1 + x2x3 has alternating rank 4 -> floor 2
        self.assertEqual(W.span_polar_floor([[0b11, 0b1100]], 4), 2)


CHEM_KMX = (
    "".join(f"APPEND_TO_REGISTER q{i} r0\n" for i in range(3)) + 'REGISTER r0 "selection"\n'
    + "".join(f"APPEND_TO_REGISTER q{i} r1\n" for i in range(3, 5)) + 'REGISTER r1 "word_0"\n'
    # caches: R + CCX into fresh wires 5, 6; product run; HMR + conditional CZ tail
    + "R q5\nCCX q0 q1 q5\nR q6\nCCX q1 q2 q6\nCCX q5 q2 q3\nCX q6 q4\nCX q5 q4\n"
    + "HMR q5 b0\nCZ q0 q1 if b0\nHMR q6 b1\nCZ q1 q2 if b1\n")


@unittest.skipUnless(HAS_NUMPY, "NumPy required (system python3)")
class AvenuesChem(unittest.TestCase):
    def setUp(self):
        import qhdl_avenues_chem as C
        import qhdl_bitsim as Bs
        import qhdl_window as W
        self.C, self.B, self.W = C, Bs, W
        self.tmp = tempfile.TemporaryDirectory()
        p = Path(self.tmp.name) / "l.kmx"
        p.write_text(CHEM_KMX)
        self.slot = dict(k=0, part=dict(part_id="k00-test", toffoli_per_shot="6.0"), kmb=p)
        self.L = C.Loader(self.slot, p)
        self.L.lv = W.Liveness(self.L.ops)

    def tearDown(self):
        self.tmp.cleanup()

    def test_regions_and_executed_attribution(self):
        L = self.L
        self.assertEqual(L.regions["cache"], (0, 4))      # R,CCX,R,CCX: last R at 2 -> cache ends after its CCX
        self.assertEqual(L.regions["tail"][0], 7 + 0)      # first HMR
        self.assertEqual((L.ccx["cache"][0], L.ccx["product"][0], L.ccx["tail"][0]), (2, 1, 0))
        self.assertEqual(L.static, 3)
        self.assertEqual(L.per_shot, 6.0)                  # two passes
        fam = dict(T_official="106.0")
        att = self.C.attribution(fam, [L], heur=100)
        self.assertEqual(att["sum"], 106.0)
        self.assertEqual(att["verdict"], "pass")
        self.assertEqual(self.C.attribution(fam, [L], heur=None)["verdict"], "circular")  # derived non-angle is circular
        self.assertEqual(self.C.attribution(dict(T_official="200.0"), [L], heur=100)["verdict"], "KILL")

    def test_r_is_a_zero_write_so_caches_are_not_free_inputs(self):
        found, capped = self.C.scan_chem_windows(self.L, sizes=(2, 3))
        self.assertEqual(capped, [])
        by = {tuple(f["ops"]): f for f in found}
        w = by[(1, 6)] if (1, 6) in by else None
        # the 3-CCX window starting at the first cache's R: products q5=q0q1, q6=q1q2, q3^=q5q2: 3 distinct nonlinear
        # outputs, so the floor is 3 and the gap 0
        three = [f for f in found if f["toffoli"] == 3][0]
        self.assertEqual(three["ops"][0], 0)               # starts at the R that zeroes q5
        self.assertEqual((three["toffoli"], three["lb"]), (3, 3))

    def test_whole_loader_floor_register_outputs(self):
        r = self.C.region_floor(self.L, "whole", (0, 7), register_only=True)
        self.assertEqual(r["status"], "exact_anf")
        self.assertEqual(r["outputs"], 2)                  # word wires q3 (cubic) and q4 (quadratic); caches not required
        self.assertEqual(r["floor"], 2)                    # degree 3 -> 2; nonlinear rank 2; fewer than the 3 shipped CCX

    def test_cache_chain_ignores_later_resets(self):
        # leading CX, a 2-pair cache chain, a product CCX, HMR, then another R+CCX pair (a second compute): the later
        # pair is NOT a cache; it sits in the tail
        kmx = (CHEM_KMX.split("R q5")[0] + "CX q0 q3\nR q5\nCCX q0 q1 q5\nR q6\nCCX q1 q2 q6\nCCX q5 q2 q3\n"
               "HMR q5 b0\nCZ q0 q1 if b0\nR q5\nCCX q0 q2 q5\n")
        p = Path(self.tmp.name) / "l2.kmx"
        p.write_text(kmx)
        L = self.C.Loader(dict(k=1, part=dict(part_id="x", toffoli_per_shot="8.0"), kmb=p), p)
        c0, c1 = L.regions["cache"]
        self.assertEqual(L.ccx["cache"][0], 2)
        self.assertEqual(L.ccx["product"][0], 1)
        self.assertEqual(L.ccx["tail"][0], 1)
        self.assertEqual(L.static, 4)
        self.assertEqual(c0, 0)

    def test_chem_pricing(self):
        fam = dict(lambda_eff=46.9075374, q=427, t=14856.0)
        p = self.C.price(52, fam)
        self.assertAlmostEqual(p["delta_cost"], 46.9075374 * 427 * 52, places=2)
        self.assertAlmostEqual(p["gm_effect_pct"], 100 * math.log(14856 / (14856 - 52)) / 3, places=4)
        self.assertEqual(p["saving_toffoli_per_pass"], 26.0)
        self.assertEqual(self.C.gm_pct(0, 100), 0.0)

    def test_negatives_slot_named_is_related_op_range_closes(self):
        led = [dict(id="NR-S", task="chemistry", outcome="kill", mechanism="AND-minimising loader synthesis",
                    family_target="angle_k_4 of 9023", key_numbers="shipped 270", why_it_failed="", keywords=""),
               dict(id="NR-O", task="chemistry", outcome="kill", mechanism="rewrite ops 150..300 of slot loader",
                    family_target="", key_numbers="", why_it_failed="", keywords="")]
        hit = {h["id"]: h for h in self.C.chem_negatives(led, 4, 100, 400)}
        self.assertFalse(hit["NR-S"]["covers"])            # slot-named is related, never closes a window
        self.assertTrue(hit["NR-S"]["why"].startswith("related"))
        self.assertTrue(hit["NR-O"]["covers"])
        self.assertNotIn("NR-S", {h["id"] for h in self.C.chem_negatives(led, 5, 0, 0)} - {"NR-S"})

    def test_g7_dominated_when_loader_exceeds_family(self):
        import json
        packet = dict(schema="x", all_paths=2, positive_loader_saving_paths=1, actionable_paths=0, next_step="wait",
                      shown=[dict(path_index=0, prefix_table="a", suffix_table="b", shared_block_slot=9,
                                  loader_toffoli_per_shot=8274.0, loader_saving_vs_cheaper_parent=20.0, actionable=False),
                             dict(path_index=1, prefix_table="c", suffix_table="d", shared_block_slot=3,
                                  loader_toffoli_per_shot=8190.0, loader_saving_vs_cheaper_parent=4.0, actionable=False)])
        p = Path(self.tmp.name) / "packet.json"
        p.write_text(json.dumps(packet))
        fam = dict(lambda_eff=46.9, q=427, t=14856.0)
        rows, meta = self.C.g7_rows(p, fam, 8194.0)
        self.assertEqual([r["dominated"] for r in rows], [True, False])
        self.assertEqual(rows[0]["price"]["gm_effect_pct"], 0.0)
        self.assertGreater(rows[1]["price"]["gm_effect_pct"], 0.0)
        self.assertEqual(self.C.g7_rows(Path(self.tmp.name) / "none.json", fam, 8194.0)[1]["status"], "unavailable")

    def test_family_best_known_gap_from_csv(self):
        import csv
        stage = Path(self.tmp.name) / "stage"
        (stage / "parts").mkdir(parents=True)
        (stage / "circuits/chemistry/fam1").mkdir(parents=True)
        kmb = stage / "circuits/chemistry/fam1/angle_k_0.kmb"
        kmb.write_bytes(b"abc")
        sha = self.C.sha256_file(kmb)

        def w(name, rows):
            with open(stage / "parts" / name, "w", newline="") as f:
                wr = csv.DictWriter(f, fieldnames=list(rows[0]))
                wr.writeheader()
                wr.writerows(rows)
        base = dict(slot_k="0", prev_words_sha256="ZERO", words_sha256="w", num_qubits="100", first_author="x")
        w("angle_circuits.csv", [dict(base, part_id="p-own", file_sha256=sha, dataset_path="circuits/chemistry/fam1/angle_k_0.kmb",
                                       toffoli_per_shot="600.0"),
                                  dict(base, part_id="p-best", file_sha256="z", dataset_path="n", toffoli_per_shot="594.0"),
                                  dict(base, part_id="p-over", file_sha256="z", dataset_path="n", toffoli_per_shot="500.0",
                                       num_qubits="129")])
        w("families.csv", [dict(family="fam1abc", parts="p-own", npz_sha256="n")])
        w("factorizations.csv", [dict(npz_sha256="n", R="8", B="143", C="44", brot="8")])
        fam, fr, slots, group = self.C.load_family(stage, "fam1")
        g = group[("0", "ZERO", "w")]
        self.assertEqual(sorted(r["part_id"] for r in g), ["p-best", "p-own"])  # the 129-qubit part breaks the cap: excluded
        self.assertEqual(min(float(r["toffoli_per_shot"]) for r in g), 594.0)
        with self.assertRaises(ValueError):
            self.C.load_family(stage, "nomatch")

    def test_generic_reference_is_labelled_not_a_bound(self):
        r = self.C.generic_reference(self.L)
        self.assertIn("not a bound", r["label"])
        self.assertEqual(r["products_estimate"], 161)  # as chem_c12_bounds.py: generic(76,15,35) = 322 -> 161


if __name__ == "__main__":
    unittest.main()
