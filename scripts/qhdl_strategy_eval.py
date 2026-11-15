"""Local corpus audit and strategy diagnostics; never certify model claims.

Build emits disjoint agent/reviewer folders. The diagnostic score measures
resource arithmetic and evidence scope, NOT discovery or semantic understanding.
Proposal assessment is a static price/receipt gate; verifier admission stays manual.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import re

REFERENCE = {'invsqrt': 2076774.23, 'dqi': 24439463583672.93,
             'chemistry': 3355599588.32}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def write(path, value):
    with Path(path).open('x') as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write('\n')


def rows(path):
    with Path(path).open(newline='') as f:
        return list(csv.DictReader(f))


def finite_positive(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x) and x > 0


def gap(public, leader):
    if not finite_positive(public) or not finite_positive(leader) or leader < public:
        raise ValueError('positive scores and leader >= public required')
    ratio = leader / public
    return dict(public_gm_projection=public, observed_leader_score=leader,
                leader_over_public=ratio, public_fraction_below_leader=1-1/ratio,
                one_task_cost_ratio=ratio**-3, equal_each_task_cost_ratio=ratio**-1,
                tier='static model / projection; compared with observed public score')


def leakage_groups(catalog, episodes):
    """Conservative exclusion graph, not inferred biological/algorithmic lineage.

    Connect same source owner (aliases via URLs), notebook, shared factorization,
    and even unreviewed cited parents. Missing links remain unknown, never safe.
    """
    parent = {r['sha256']: r['sha256'] for r in catalog}
    if len(parent) != len(catalog):
        raise ValueError('duplicate circuit SHA')

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        if a in parent and b in parent:
            a, b = find(a), find(b)
            parent[max(a, b)] = min(a, b)

    anchors = {}
    reasons = Counter()
    for r in catalog:
        text = r.get('source_notebook_url', '') + ' ' + r.get('also_found_in', '')
        keys = set()
        for owner, notebook in re.findall(r'kaggle\.com/code/([^/\s]+)/([^/\s,;"\]]+)', text):
            keys.update([('source_owner', owner.lower()), ('notebook', owner.lower()+'/'+notebook.lower())])
        if r.get('factorization_sha256'):
            keys.add(('factorization', r['factorization_sha256']))
        for key in keys:
            if key in anchors:
                union(r['sha256'], anchors[key]); reasons[key[0]] += 1
            else:
                anchors[key] = r['sha256']
    for e in episodes:
        for p in e['parent_candidates']:
            union(e['child']['sha256'], p['sha256']); reasons['candidate_parent'] += 1
    groups = defaultdict(list)
    for sha in parent:
        groups[find(sha)].append(sha)
    return dict(groups=[dict(id=k, members=sorted(v)) for k, v in sorted(groups.items())],
                edges_by_reason=dict(reasons), split_status='not assigned',
                warning='Exclusion groups only. Missing lineage is unknown; audit before any held-out claim.')


def chemistry_regimes(catalog):
    groups = defaultdict(list)
    fields = ['R', 'B', 'C', 'brot', 'KEEP_BITS']
    for row in catalog:
        if row['task'] == 'chemistry' and all(row.get(k) for k in fields) and row.get('local_task_score'):
            groups[tuple(row[k] for k in fields)].append(row)
    result = []
    for shape, members in groups.items():
        best = min(members, key=lambda r: float(r['local_task_score']))
        result.append(dict(parameters=dict(zip(fields,shape)), count=len(members),
                           representative=best, cost=float(best['local_task_score']),
                           status='descriptive regime; not a causal label or parent relation'))
    return sorted(result, key=lambda r:r['cost'])


def build(a):
    catalog, episodes, accepted, negatives = rows(a.catalog), jsonl(a.episodes), jsonl(a.reviewed), rows(a.negatives)
    by = {r['sha256']: r for r in catalog}
    selected = {}
    for task in REFERENCE:
        key = 'dqi_expected_cost' if task == 'dqi' else 'local_task_score'
        eligible = [r for r in catalog if r['task'] == task and r.get(key) and
                    finite_positive(float(r[key]))]
        if not eligible:
            raise ValueError('missing comparable costs for '+task)
        r = min(eligible, key=lambda r: float(r[key]))
        selected[task] = dict(sha256=r['sha256'], cost=float(r[key]), cost_field=key,
                              author=r['author'], url=r['source_notebook_url'],
                              version=r['source_notebook_version'], qubits=float(r['qubits']),
                              evidence_tier=('recorded expected cost; draw receipts not re-audited' if task == 'dqi'
                                             else r['local_score_tier']))
    public = math.prod(REFERENCE[t]/r['cost'] for t, r in selected.items())**(1/3)
    budget = gap(public, a.leader)
    transitions = []
    for e in episodes:
        c = by[e['child']['sha256']]
        key = 'dqi_expected_cost' if e['task'] == 'dqi' else 'local_task_score'
        for pp in e['parent_candidates']:
            p = by[pp['sha256']]
            if not p.get(key) or not c.get(key):
                continue
            ratio = float(c[key])/float(p[key])
            transitions.append(dict(task=e['task'], parent=p['sha256'], child=c['sha256'],
                                    cost_ratio=ratio, task_reduction=1-ratio,
                                    portfolio_gain=ratio**(-1/3)-1, cost_field=key,
                                    relationship='reviewed' if e['reviewed_parent'] == p['sha256'] else 'candidate only',
                                    status=e['status'], compound=e['status']=='reviewed_compound',
                                    claim=e['author_claim_or_curated_summary'],
                                    child_source=e['child']))
    transitions.sort(key=lambda r: r['cost_ratio'])
    active = {r['id']: r for r in negatives}
    superseded = {s for r in negatives for s in re.findall(r'NR-\d+', r.get('supersedes', ''))}
    groups = leakage_groups(catalog, episodes)
    regimes = chemistry_regimes(catalog)
    audit = dict(catalog_count=len(catalog), tasks=dict(Counter(r['task'] for r in catalog)),
                 change_summaries=sum(bool(r.get('change_summary')) for r in catalog),
                 core_designs=sum(bool(r.get('core_design')) for r in catalog),
                 episode_status=dict(Counter(e['status'] for e in episodes)),
                 accepted_explanations=len(accepted),
                 accepted_hash_matches=sum(e['case_id'] in by and e['comparison_sha256'] in by for e in accepted),
                 negatives=len(active), superseded_negatives=sorted(superseded),
                 comparable_candidate_links=len(transitions),
                 links_over_five_percent=sum(t['task_reduction'] >= .05 for t in transitions),
                 exclusion_group_sizes=sorted([len(g['members']) for g in groups['groups']], reverse=True),
                 catalog_only_sha=[r['sha256'] for r in catalog if not r.get('change_summary')],
                 unavailable_circuit_notices=[r['sha256'] for r in catalog if 'FILE NOT SHIPPED' in r.get('notes', '')])
    audit['chemistry_parameter_regimes'] = len(regimes)
    cases, answers = [], []

    def case(cid, task, prompt, facts, expected):
        cases.append(dict(case_id=cid, task=task, prompt=prompt, facts=facts,
                          response='Return {case_id, answer: number|boolean|null, confidence: 0..1, rationale: string}. Null means abstain.'))
        answers.append(dict(case_id=cid, expected=expected, absolute_tolerance=1e-8,
                            grading_scope='diagnostic only; rationale requires separate review'))

    scores = dict(public_gm=public, leader_gm=a.leader,
                  score_equation='GM=(reference_1/cost_1 * reference_2/cost_2 * reference_3/cost_3)^(1/3)',
                  evidence='public is local expected-cost projection; leader is observed leaderboard')
    case('D01', 'all', 'What fraction of incumbent cost may ONE task retain to close this score gap, other tasks unchanged?', scores, budget['one_task_cost_ratio'])
    case('D02', 'all', 'What fraction of incumbent cost may EACH task retain if all three improve equally?', scores, budget['equal_each_task_cost_ratio'])
    q = selected['dqi']['qubits']
    dqi = dict(source=selected['dqi'], new_qubits=q+24, cost_equation='R * Q^2 * mean_Toffolis', R_ratio=1)
    case('D03', 'dqi', 'What mean-Toffoli ratio breaks even after adding 24 wires?', dqi, (q/(q+24))**2)
    case('D04', 'dqi', 'What mean-Toffoli ratio closes the WHOLE three-task score gap after adding 24 wires?', dict(**dqi, **scores), budget['one_task_cost_ratio']*(q/(q+24))**2)
    case('D05', 'invsqrt', 'If the stated rotation charge disappeared for free, would that alone close the whole three-task score gap?',
         dict(**scores, rotation_fraction_of_charge=.1828, Q_ratio=1,
              assumption='Optimistic hypothetical; 18.28% is a rounded recorded-margin model, not an attainable rewrite.'), (1-.1828)**(-1/3) >= a.leader/public)
    chem = by[selected['chemistry']['sha256']]
    case('D06', 'chemistry', 'What is the three-task score multiplier if lambda_eff becomes 46.8 and Q and Toffolis stay fixed?',
         dict(source=selected['chemistry'], lambda_before=float(chem['lambda_eff']), lambda_after=46.8,
              cost_equation='lambda_eff * Q_total * Toffolis_total', assumption='Hypothetical feasible lambda, not a proven floor.'),
         (float(chem['lambda_eff'])/46.8)**(1/3))
    # Authored scope answers for these two audited records only; never infer a
    # universal-equivalence answer from an arbitrary future explanation.
    scope_cases = {
        '6dbb98b34c3b16f1c4e019350c10d42259a6faa97e464c9b7c9390f877f392f4',
        '9a8627c812833a6feca6eb82239f38d1a9a5e1eee299a9a7af9abd85e36339be',
    }
    scoped = sorted([e for e in accepted if e['case_id'] in scope_cases], key=lambda e:e['case_id'])
    if len(scoped) != 2:
        raise ValueError('the two audited explanation scope controls are required')
    for i, e in enumerate(scoped, 7):
        case(f'D{i:02}', e['source']['task'],
             'Does this supplied record establish exact equivalence for every measurement outcome? Answer true only if that universal claim is established.',
             dict(source_record=e, mode='supplied-explanation evidence-scope control, NOT blind forensics'), False)

    a.out.mkdir(parents=True, exist_ok=False)
    (a.out/'agent').mkdir(); (a.out/'review').mkdir()
    for name, value in [('audit.json', audit), ('gap.json', budget), ('frontier.json', selected),
                        ('leakage-groups.json', groups), ('candidate-transitions.json', transitions),
                        ('chemistry-regimes.json', regimes)]:
        write(a.out/name, value)
    for path, records in [(a.out/'agent/cases.jsonl', cases), (a.out/'review/answers.jsonl', answers)]:
        with path.open('x') as f:
            f.write(''.join(json.dumps(r)+'\n' for r in records))
    write(a.out/'review/oracle-responses.json', [dict(case_id=r['case_id'], answer=r['expected'], confidence=1., rationale='software oracle') for r in answers])
    write(a.out/'review/abstention-responses.json', [dict(case_id=r['case_id'], answer=None, confidence=0., rationale='abstention control') for r in answers])
    write(a.out/'agent/proposal-template.json', dict(task='dqi', base_sha256=selected['dqi']['sha256'],
          mechanism='', changed_interface='', preserved_outputs_and_phase='', schedule_and_cleanup='',
          falsification_test='', negative_result_ids=[], reopening_difference='',
          ratios={'R': None, 'Q': None, 'T': None}, evidence=[]))
    research = []
    questions = {
        'chemistry': 'Can a jointly chosen factorization shape, rounded word table and loader program lower full cost? Reproduce the source, then test one representation change with both residual constraints and the actual loader cost.',
        'dqi': 'Can a byte-derived multi-output update block be compiled jointly more cheaply than independent field products? Preserve the caller interface, exceptional cases, measurement phases and reversible scratch schedule.',
        'invsqrt': 'Can a different low-width Boolean phase program improve the full Q times charge cost? Account for measurement corrections, all-input approximation error, rotation synthesis margin and cleanup.',
    }
    for task, question in questions.items():
        research.append(dict(case_id='F-'+task, task=task, source=selected[task], question=question,
                             single_task_gap_cost_target=selected[task]['cost']*budget['one_task_cost_ratio'],
                             target_tier='static model / projection, not attainable-cost claim',
                             required_result=['byte-derived input/output/phase contract', 'reproduced baseline',
                                              'concrete replacement and reversible schedule',
                                              'full resource accounting and scoped falsification',
                                              'independent verifier evidence before accepting any gain'],
                             status='open research brief; no hidden solution, no automatic semantic score',
                             evaluation='Matched model/budget/tool arms require a separate frozen, preregistered run. This session has seen the briefs; not a blind model trial.'))
    with (a.out/'agent/frontier-tasks.jsonl').open('x') as f:
        f.write(''.join(json.dumps(r)+'\n' for r in research))
    write(a.out/'input-hashes.json', {str(p):digest(p) for p in [a.catalog,a.episodes,a.reviewed,a.negatives]})
    print(json.dumps(dict(audit=audit, gap=budget), indent=2))


def score(answer_key, responses):
    key = {r['case_id']:r for r in answer_key}
    supplied = {}
    for r in responses:
        if r['case_id'] in supplied or r['case_id'] not in key:
            raise ValueError('duplicate or unknown case id')
        supplied[r['case_id']] = r
    result = []
    for cid, gold in key.items():
        r = supplied.get(cid)
        value = r.get('answer') if r else None
        conf = r.get('confidence') if r else None
        valid_conf = isinstance(conf, (int, float)) and not isinstance(conf, bool) and math.isfinite(conf) and 0 <= conf <= 1
        correct = False
        if value is not None and valid_conf:
            target = gold['expected']
            if isinstance(target, bool):
                correct = isinstance(value, bool) and value == target
            elif isinstance(value, (float, int)) and not isinstance(value, bool) and math.isfinite(value):
                correct = abs(value-target) <= gold['absolute_tolerance']
        result.append(dict(case_id=cid, present=r is not None, attempted=value is not None,
                           valid_confidence=valid_conf, correct=correct,
                           brier=(conf-float(correct))**2 if valid_conf and value is not None else None))
    return dict(cases=len(key), present=sum(r['present'] for r in result),
                attempted=sum(r['attempted'] for r in result), correct=sum(r['correct'] for r in result),
                diagnostic_accuracy=sum(r['correct'] for r in result)/len(key) if key else None,
                rows=result, semantic_strategy_score=None, verifier_passing_gain=None,
                warning='Arithmetic/evidence-scope diagnostics only. Rationale and strategy not machine-validated.')


def assess(proposal, frontier, budget):
    task = proposal.get('task')
    if task not in frontier:
        raise ValueError('unknown task')
    errors = []
    if proposal.get('base_sha256') != frontier[task]['sha256']:
        errors.append('base SHA is not this frozen frontier')
    required = ['mechanism','changed_interface','preserved_outputs_and_phase',
                'schedule_and_cleanup','falsification_test']
    errors += ['missing '+k for k in required if not isinstance(proposal.get(k), str) or not proposal[k].strip()]
    if proposal.get('negative_result_ids') and not proposal.get('reopening_difference'):
        errors.append('negative-result reopening needs a distinct mechanism/assumption')
    keys = {'dqi':('R','Q','T'), 'invsqrt':('Q','charge'), 'chemistry':('lambda','Q','T')}[task]
    ratios = proposal.get('ratios', {})
    ratio = None
    if not all(finite_positive(ratios.get(k)) for k in keys):
        errors.append('missing/nonpositive/nonfinite full-cost ratios: '+','.join(keys))
    else:
        ratio = math.prod(ratios[k] for k in keys) * (ratios['Q'] if task == 'dqi' else 1)
    receipts = []
    for e in proposal.get('evidence', []):
        p = Path(e.get('path', ''))
        ok = p.is_file() and bool(e.get('sha256')) and digest(p) == e['sha256']
        receipts.append(dict(path=str(p), hash_matches=ok))
        if not ok: errors.append('missing or mismatched evidence: '+str(p))
    if not receipts: errors.append('no pinned evidence')
    return dict(errors=errors, formal_packet_complete=not errors, projected_task_cost_ratio=ratio,
                projected_portfolio_multiplier=ratio**(-1/3) if ratio else None,
                reaches_gap_on_claimed_ratios=ratio <= budget['one_task_cost_ratio'] if ratio else None,
                evidence_receipts=receipts, tier='static model / projection',
                validated_improvement=False, semantic_review='pending',
                warning='Ratios are proposal assumptions. Hash match proves file identity, not its claim. Independent verifier and task-specific gates required.')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest='command', required=True)
    b = sub.add_parser('build')
    for name in ['catalog','episodes','reviewed','negatives','out']:
        b.add_argument('--'+name, type=Path, required=True)
    b.add_argument('--leader', type=float, required=True)
    s = sub.add_parser('score')
    s.add_argument('--answers', type=Path, required=True); s.add_argument('--responses', type=Path, required=True)
    s.add_argument('--out', type=Path, required=True)
    p = sub.add_parser('assess-proposal')
    for name in ['proposal','frontier','gap','out']:
        p.add_argument('--'+name, type=Path, required=True)
    a = ap.parse_args()
    if a.command == 'build': build(a)
    elif a.command == 'score': write(a.out, score(jsonl(a.answers), json.loads(a.responses.read_text())))
    else: write(a.out, assess(json.loads(a.proposal.read_text()), json.loads(a.frontier.read_text()), json.loads(a.gap.read_text())))


if __name__ == '__main__':
    main()
