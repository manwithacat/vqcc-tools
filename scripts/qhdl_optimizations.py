"""Discover implemented rewrites and export exact phase predicates for refactoring."""
from pathlib import Path
import time
import qhdl_envelope as E
import qhdl_rewrite as R
import qhdl_rewrite_phase_fold as P
import qhdl_predicates as QP

CONTRACTS = {
    'and3-dirty-mbu': ('target zero after; dirty borrows restored; exact measurement-phase correction', '1 outcome bit; no extra quantum wire'),
    'flag-ladder-mbu': ('target zero after; clean ladder helpers; measurement-phase correction', 'helper lifetimes must fit'),
    'u3-mbu': ('recognised partial measurement uncompute; correct entry function and fixups', '1 outcome bit'),
    'cc-chain': ('clean target and conditionally clean helper obligations', 'S6/S12 templates only; helper lifetimes must fit'),
    'single-toffoli-mbu': ('target zero after; phase correction matches its two controls', '1 outcome bit'),
    'and-merge': ('agent supplies replacement; reachable-function ranking does not prove placement or phase', 'must validate width and cleanup of supplied replacement'),
    'phase-fold': ('equal bounded ANF predicates on arbitrary entry wires; exact modular phase identity', 'no new wires/bits; original quantum header retained'),
}


def catalog():
    rows=[]
    for name, cls in R.load_classes().items():
        obligations, footprint=CONTRACTS[name]
        rows.append(dict(name=name, summary=cls.summary, tasks=list(cls.tasks),
                         automatic_replacement=name!='and-merge', needs=list(cls.needs),
                         obligations=obligations, footprint=footprint, certificate=cls.certify,
                         cost_units=['Toffolis', 'rotation gates'] if name=='phase-fold' else ['Toffolis'],
                         applicability='matched sites only; no claim of applicability to every circuit',
                         source=f'scripts/qhdl_rewrite_{name.replace("-", "_")}.py'))
    return rows


def v_catalog(api,a):
    rows=catalog()
    if a.cls:
        rows=[r for r in rows if r['name']==a.cls]
        if not rows: raise api.QhdlError('unknown optimization class', ', '.join(R.CLASSES))
    ev=Path(api.WORK)/'optimization-library'/f'catalog-{R._tool_sha(api)}.json'
    api.write_json(ev,dict(schema=1,classes=rows,rubric='docs/qhdl-optimization-library.md'))
    led=E.Ledger();led.earn('static','implementation contracts, not a circuit search or cost result')
    return E.Envelope(led,f'{len(rows)} implemented rewrite class(es); applicability and obligations are explicit',
                      domain='registered QHDL rewrites',evidence=api.rel(ev),
                      rerun=api.cmd('optimizations',*(['--class',a.cls] if a.cls else [])),
                      tool_sha=R._tool_sha(api),circuit_sha=None,
                      rows=[dict(name=r['name'],automatic=r['automatic_replacement'],tasks=','.join(r['tasks']),
                                 units=','.join(r['cost_units'])) for r in rows],
                      facts=dict(contracts=rows if a.cls else 'full contracts in evidence file',
                                 rubric='docs/qhdl-optimization-library.md'))


def v_profile(api,a):
    meta=api.ensure_open(a.circuit);path=api.circuit_path(meta)
    A,B,span=R._span(api,a,path)
    if B-A>2_000_000:
        raise api.QhdlError('phase-profile is bounded to two million ops per query','select --ops or --region')
    if not 1<=a.max_terms<=4096 or not 1<=a.max_span<=65536:
        raise api.QhdlError('budgets: max-terms 1..4096; max-span 1..65536')
    t0=time.monotonic();k=R.K.read(str(path))
    groups,stats=P.scan(k,R.Ops(k),A,B,max_terms=a.max_terms,max_span=a.max_span,include_predicates=True)
    predicates=stats.pop('predicates')
    for row in predicates:
        row['contract'] = predicate_contract(meta['sha256'], row, a.max_terms)
    ev=api.ws_dir(meta['sha256'])/'phase'/f'profile_{A}_{B}_t{a.max_terms}_s{a.max_span}_{R._tool_sha(api)}.json'
    result=dict(schema=1,circuit_sha=meta['sha256'],A=A,B=B,coverage=stats,predicates=predicates,folds=groups,
                seconds=time.monotonic()-t0,
                semantics='ANF masks use bit w for region-entry wire qw; mask 0 is constant one; XOR of monomials',
                angle_units='unsigned integer / 2^128 full turns, modulo one turn',
                contract_schema='typed Boolean phase predicates; computational basis, arbitrary cut inputs',
                scope='each region independently: arbitrary entry wires; unsupported/control/measurement/reset cuts',
                limitations=['no approximate synthesis','no cross-cut predicate identification',
                             'no rotation charge without task margin','coverage limits are not impossibility proofs'])
    api.write_json(ev,result)
    led=E.Ledger();led.earn('static','exact symbolic functions inside bounded unconditional reversible regions')
    return E.Envelope(led,f"{len(predicates)} phase predicates in {stats['regions']} regions; {len(groups)} exact folds; "
                      f"{stats['barriers']} barriers, {stats['capacity_cuts']} capacity cuts",
                      domain=span+': arbitrary region-entry wire states',evidence=api.rel(ev),
                      rerun=api.cmd('phase-profile','-c',meta['sha12'],'--ops',f'{A}:{B}',
                                    '--max-terms',a.max_terms,'--max-span',a.max_span),
                      tool_sha=R._tool_sha(api),circuit_sha=meta['sha256'],
                      facts=dict(coverage=stats,limitations=result['limitations']),
                      rows=[{x:r[x] for x in ('region','ops','degree','terms','phase_ops','rotations')} for r in predicates],
                      next_cmds=[api.cmd('census','-c',meta['sha12'],'--class','phase-fold','--ops',f'{A}:{B}')])


def predicate_contract(source_sha256, row, max_terms=P.MAX_TERMS):
    """Bind a legacy scanner predicate to its exact source and entry cut.

    The scanner has already cut at every direct/stack condition, measurement,
    reset and unsupported operation. True is therefore the explicit guard here;
    this adapter does not infer classical relationships or reachable-state facts.
    """
    monomials = []
    for encoded in row['predicate']:
        mask = int(encoded, 16)
        variables = []
        while mask:
            bit = mask & -mask
            variables.append(QP.Var('quantum', bit.bit_length() - 1, 0))
            mask ^= bit
        monomials.append(frozenset(variables))
    anf = QP.ANF(frozenset(monomials), term_cap=max_terms)
    context = QP.Context(source_sha256, row['entry_cut'])
    predicate = QP.Predicate(context, anf)
    phase = QP.ExactPhaseTerm(predicate, QP.ANF.constant(True), int(row['angle128'], 16))
    return phase.to_dict()


def v_predicate_families(api, a):
    rows = QP.family_catalog()
    ev = Path(api.WORK) / 'optimization-library' / f'predicate-families-{R._tool_sha(api)}.json'
    api.write_json(ev, dict(schema=1, families=rows,
                           literature='docs/qhdl-phase-predicates.md',
                           guide='docs/qhdl-phase-predicates.md',
                           scope='Boolean phase predicates; catalogue is not a circuit cost result'))
    led = E.Ledger()
    led.earn('static', 'documented Boolean constructors and exact phase contracts')
    return E.Envelope(led, f'{len(rows)} reusable Boolean predicate families',
                      domain='computational-basis Boolean functions, not arbitrary quantum effects',
                      evidence=api.rel(ev), rerun=api.cmd('predicate-families'),
                      tool_sha=R._tool_sha(api), circuit_sha=None, rows=rows,
                      facts=dict(guide='docs/qhdl-phase-predicates.md',
                                 circuit_rewrite=False, cost_improvement_claim=False))


def qhdl_register(subparsers,api):
    f=subparsers.add_parser('predicate-families',help='inspect reusable Boolean phase-predicate families and contracts')
    f.add_argument('--all',action='store_true');f.set_defaults(fn=lambda a:v_predicate_families(api,a))
    p=subparsers.add_parser('optimizations',help='browse implemented optimization contracts and refactoring rubric')
    p.add_argument('--class',dest='cls');p.add_argument('--all',action='store_true')
    p.set_defaults(fn=lambda a:v_catalog(api,a))
    q=subparsers.add_parser('phase-profile',help='export bounded exact Boolean phase predicates for refactoring')
    q.add_argument('-c','--circuit',required=True);q.add_argument('--ops');q.add_argument('--region')
    q.add_argument('--max-terms',type=int,default=P.MAX_TERMS);q.add_argument('--max-span',type=int,default=P.MAX_SPAN)
    q.add_argument('--all',action='store_true');q.set_defaults(fn=lambda a:v_profile(api,a))
