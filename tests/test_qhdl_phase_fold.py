"""Independent exact phase/state checks, including dirty inputs and low angle bits."""
import itertools
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import unittest

try:
    import numpy as np
except ModuleNotFoundError:
    raise unittest.SkipTest('NumPy required; use accepted kickmix runtime')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _optional import requires_kickmix  # noqa: E402
import qhdl_kmb as K
import qhdl_edit as ED
import qhdl_rewrite as R
import qhdl_rewrite_phase_fold as P


def build(directory, lines, name='base'):
    empty = directory / (name+'-empty.kmb')
    K.write_kmb(str(empty), np.array([], np.uint8), {}, registers=[('x', [(True,i) for i in range(5)])],
                num_qubits=5, num_bits=0)
    out = directory/(name+'.kmb')
    ED.edit_file(empty, 'insert 0 => '+'; '.join(lines), out)
    return out


def run(k, inp):
    """Independent scalar semantics, all five wires arbitrary, exact integer phases."""
    src=ED.Source(k); q=list(inp); phase=0
    for i in range(len(k.op_types)):
        r=src.op(i); c=r['code']; w=r['q']
        if c == 0x11: q[w[0]] ^= 1
        elif c == 0x21: q[w[1]] ^= q[w[0]]
        elif c == 0x31: q[w[2]] ^= q[w[0]] & q[w[1]]
        elif c == 0x22: q[w[0]],q[w[1]]=q[w[1]],q[w[0]]
        elif c == 0x32:
            if q[w[0]]: q[w[1]],q[w[2]]=q[w[2]],q[w[1]]
        elif c in (0x10,0x20,0x30,0x13):
            enabled=all(q[j] for j in w)
            if enabled: phase += int.from_bytes(r['angle'],'little') if c==0x13 else 1<<127
        elif c == 0x06: pass
        else: raise AssertionError(c)
    return q,phase % (1<<128)


def fold(path, out, **kw):
    k=K.read(str(path)); gs,stats=P.scan(k,R.Ops(k),**kw)
    edits=[]
    for g in gs:
        edits += P.CLASS.apply({'fold':g},None,None)
    ED.edit_file(path,'\n'.join(edits),out)
    return gs,stats


class PhaseFold(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.d=Path(self.tmp.name)
    def tearDown(self): self.tmp.cleanup()
    def equivalent(self,a,b):
        ka,kb=K.read(str(a)),K.read(str(b))
        for x in itertools.product((0,1),repeat=5):
            self.assertEqual(run(ka,x),run(kb,x),x)

    def test_equal_predicate_different_wires_and_dirty_inputs(self):
        a=build(self.d,['CCZ q0 q1 q2','SWAP q2 q3','X q4','CCZ q0 q1 q3'])
        b=self.d/'fold.kmb'; gs,_=fold(a,b)
        self.assertEqual(len(gs),1);self.assertEqual(gs[0]['toffolis_before'],2)
        self.equivalent(a,b)

    def test_nonlinear_uncompute_and_rotation_low_bits(self):
        v=1234567890123456789012345
        a=build(self.d,[f'Z_POW q2 angle128:{v:032x}','CCX q0 q1 q2',
                       'CX q3 q4','CCX q0 q1 q2',f'Z_POW q2 angle128:{(-v)%P.MOD:032x}'])
        b=self.d/'fold.kmb';gs,_=fold(a,b)
        self.assertEqual(gs[0]['rotations_after'],0);self.equivalent(a,b)

    def test_half_turn_specialization_and_global_phase(self):
        a=build(self.d,[f'Z_POW q0 angle128:{P.HALF//2:032x}',
                       f'Z_POW q0 angle128:{P.HALF//2:032x}','X q0','Z q0'])
        b=self.d/'fold.kmb';gs,_=fold(a,b)
        self.assertEqual(gs[0]['replacement'],'Z q0');self.equivalent(a,b)
        self.assertEqual(gs[0]['general_rotations_delta'],0)
        self.assertEqual(gs[0]['t_gates_delta'],0)
        # Z(x)+Z(not x) is a nonzero global phase: do not cancel it.
        self.assertEqual(run(K.read(str(b)),[0]*5)[1],P.HALF)

    def test_barriers_and_nonmatches(self):
        for middle in ['HMR q4 b0','R q4','X q4 if b0','PUSH_CONDITION b0; POP_CONDITION']:
            a=build(self.d,['CCZ q0 q1 q2',middle,'CCZ q0 q1 q2'],name=str(len(middle)))
            gs,stats=P.scan(K.read(str(a)),R.Ops(K.read(str(a))))
            self.assertEqual(gs,[]);self.assertGreater(stats['barriers'],0)
        a=build(self.d,['CCZ q0 q1 q2','X q2','CCZ q0 q1 q2'],name='not')
        self.assertEqual(P.scan(K.read(str(a)),R.Ops(K.read(str(a))))[0],[])

    def test_budget_cuts_remain_sound(self):
        a=build(self.d,['Z q0','CX q1 q0','CX q2 q0','Z q0','Z q0'])
        b=self.d/'fold.kmb';gs,stats=fold(a,b,max_terms=1,max_span=3)
        self.assertGreater(stats['capacity_cuts'],0);self.equivalent(a,b)

    def test_random_reversible_windows(self):
        rng=random.Random(20261102)
        for case in range(40):
            lines=[]
            for _ in range(25):
                gate,n=rng.choice([('X',1),('CX',2),('CCX',3),('SWAP',2),('CSWAP',3),('Z',1),('CZ',2),('CCZ',3),('Z_POW',1)])
                line=gate+' '+ ' '.join(f'q{x}' for x in rng.sample(range(5),n))
                if gate=='Z_POW': line+=f' angle128:{rng.getrandbits(128):032x}'
                lines.append(line)
                if gate in ('Z','CZ','CCZ','Z_POW'): lines.append(line)
            a=build(self.d,lines,name=f'r{case}');b=self.d/f'f{case}.kmb';fold(a,b,max_terms=32,max_span=20)
            self.equivalent(a,b)

    @requires_kickmix
    def test_cli_census_rewrite_and_rotation_units(self):
        a=build(self.d,[f'Z_POW q0 angle128:{1:032x}',f'Z_POW q0 angle128:{2:032x}'])
        env=dict(os.environ,QHDL_WORK=str(self.d/'work'))
        def cli(*args):
            r=subprocess.run([sys.executable,str(ROOT/'scripts/qhdl.py'),*args],env=env,capture_output=True,text=True)
            self.assertEqual(r.returncode,0,r.stderr);return json.loads(r.stdout)
        hit=cli('census','-c',str(a),'--class','phase-fold')
        self.assertEqual(hit['rows'][0]['rotations_delta'],-1)
        b=self.d/'cli.kmb'
        r=cli('rewrite','-c',str(a),'--class','phase-fold','--out',str(b),'--certify','none')
        cert=json.loads(Path(str(b)+'.certificate.json').read_text())
        self.assertEqual(cert['predicted']['toffolis_per_shot_delta'],0)
        self.assertEqual(cert['predicted']['rotations_delta'],-1)
        self.equivalent(a,b)

    @requires_kickmix
    def test_catalog_and_profile(self):
        import qhdl_optimizations as O
        self.assertEqual({r['name'] for r in O.catalog()},set(R.CLASSES))
        for r in O.catalog():
            self.assertTrue(r['obligations']);self.assertTrue(r['footprint'])
        a=build(self.d,['CCX q0 q1 q2','Z q2','Z q2'])
        env=dict(os.environ,QHDL_WORK=str(self.d/'work'))
        r=subprocess.run([sys.executable,str(ROOT/'scripts/qhdl.py'),'phase-profile','-c',str(a)],
                         env=env,capture_output=True,text=True)
        self.assertEqual(r.returncode,0,r.stderr)
        self.assertLessEqual(len(r.stdout.encode()),2048)
        e=json.loads(r.stdout);d=json.loads(Path(e['evidence']).read_text())
        self.assertEqual(d['predicates'][0]['predicate'],['0x3','0x4'])
        self.assertEqual(d['predicates'][0]['degree'],2)
        self.assertEqual(len(d['folds']),1)

    def test_fdiff_certificate_persists_its_evidence(self):
        from unittest.mock import patch
        from types import SimpleNamespace
        import qhdl_window as W
        result=dict(branches={'zero':dict(phase_differs=0,x_not_restored=[0,0],
                                         scratch_dirty=[0,0],mean_toffoli=[1,1])})
        api=SimpleNamespace(ensure_open=lambda p:dict(sha256='candidate'),Loaded=lambda m:m,
                            write_json=lambda p,d:Path(p).write_text(json.dumps(d)),TOOL_FILES=[])
        with patch.object(W,'fdiff_whole',return_value=result):
            r=R.fdiff_whole(api,dict(sha256='parent'),self.d/'candidate.kmb')
        ev=Path(r['evidence']);self.assertTrue(ev.is_file())
        data=json.loads(ev.read_text());self.assertEqual(data['parent'],'parent')
        self.assertEqual(data['candidate'],'candidate');self.assertEqual(data['branches'],result['branches'])

    def test_angle_literal_rejection(self):
        for x in ['-1','0'*33,'gg'*16,'1']:
            with self.assertRaises(ED.EditError): ED.parse_op('Z_POW q0 angle128:'+x)


if __name__=='__main__': unittest.main()
