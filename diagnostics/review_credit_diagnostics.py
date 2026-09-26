"""Supplement the historical audit without replaying or overwriting it.

Every forced branch retains one stochastic continuation: reported differences
are sample-path contrasts, not estimates of expected-optimal action regret.
"""
from pathlib import Path
import argparse
import hashlib
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
from diagnostics.async_task_credit_audit import centered_scores, optimal_set, TIE_ATOL


def pending_components(frame, gamma):
    """Include events after the decision even if their origin decision is earlier."""
    p = frame[frame.task_role.eq('preexisting_pending')].copy()
    p['pending_event'] = gamma ** (p.completion_time-p.root_decision_time) * p.task_reward
    return p


def run(source, output):
    source, output = Path(source), Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError('Use a new empty output directory; historical results are immutable')
    manifest = json.loads((source/'reward_provenance_manifest.json').read_text())
    gamma = manifest['gamma']
    hashes, pending = {}, []
    for path in sorted((source/'branches').glob('*_provenance.csv.gz')):
        hashes[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
        frame = pd.read_csv(path, usecols=['state_id','forced_action_index','task_id',
                             'task_role','root_decision_time','completion_time','task_reward','task_delay'])
        pending.append(pending_components(frame, gamma))
    if not pending:
        raise ValueError('No original full-precision branch provenance found')
    p = pd.concat(pending, ignore_index=True)
    totals = p.groupby(['state_id','forced_action_index']).pending_event.sum()
    branch = pd.read_csv(source/'decision_vs_event_return.csv')
    branch['pending_event'] = [totals.get((sid,a),0.) for sid,a in zip(branch.state_id,branch.action_index)]
    branch['all_outcomes_event'] = branch.total_event + branch.pending_event
    safe = pd.read_csv(source/'safe_min_latency_component_alignment.csv')
    safe = safe[(safe.semantics=='event') & (safe.component=='total')].set_index('state_id').action_index
    rows, details = [], []
    for sid, f in branch.groupby('state_id', sort=True):
        f=f.sort_values('action_index').copy()
        probabilities=f.policy_probability.to_numpy()
        pending_adv, _=centered_scores(f.pending_event,probabilities)
        total_adv, _=centered_scores(f.all_outcomes_event,probabilities)
        f['pending_advantage']=pending_adv; f['all_outcomes_advantage']=total_adv
        details.append(f[['state_id','action_index','policy_probability','pending_event',
                          'pending_advantage','total_event','all_outcomes_event','all_outcomes_advantage']])
        actions=f.action_index.to_numpy(); values=f.all_outcomes_event.to_numpy()
        oldset=set(actions[optimal_set(f.total_event)]); newset=set(actions[optimal_set(values)])
        rows.append({'state_id':sid,'pending_spread':float(np.ptp(f.pending_event)),
                     'optimal_set_changed':oldset!=newset,
                     'sampled_ppo_regret':float(values.max()-values[np.flatnonzero(actions==f.selected_original_action.iloc[0])[0]]),
                     'sampled_safemin_regret':float(values.max()-values[np.flatnonzero(actions==safe.loc[sid])[0]])})
    stats=pd.DataFrame(rows)
    raw=p.groupby(['state_id','task_id']).task_reward.agg(['min','max'])
    changed=raw[raw['max']-raw['min']>1e-8]
    s=pd.read_csv(source/'safe_min_latency_component_alignment.csv')
    q=pd.read_csv(source/'ppo_component_alignment.csv')
    s=s[(s.semantics=='event') & (s.component=='total')]
    q=q[(q.semantics=='event') & (q.component=='total')]
    gaps=s[['state_id','regret','conflict_state']].merge(q[['state_id','regret']],on='state_id',suffixes=('_safe','_ppo'),validate='one_to_one')
    gaps['gap']=gaps.regret_ppo-gaps.regret_safe
    gap_summary=gaps.groupby('conflict_state').agg(states=('state_id','size'),safe_regret=('regret_safe','mean'),
                         ppo_regret=('regret_ppo','mean'),weighted_gap=('gap',lambda x:x.sum()/len(gaps)))
    summary={'states':len(stats),'branches':len(branch),'pending_affected_states':int(changed.index.get_level_values(0).nunique()),
             'pending_affected_tasks':len(changed),'corrected_optimal_set_changed_states':int(stats.optimal_set_changed.sum()),
             'continuations_per_state_action':1,'causal_root_cause':'unresolved',
             'evidence_scope':'fixed shared random stream; not expected Q or a confidence interval',
             'source_provenance_sha256':hashes}
    output.mkdir(parents=True,exist_ok=True)
    stats.to_csv(output/'pending_effect_summary.csv',index=False)
    pd.concat(details).to_csv(output/'all_outcomes_action_values.csv.gz',index=False,compression='gzip')
    gap_summary.to_csv(output/'performance_gap_decomposition.csv')
    ranking = pd.read_csv(source/'action_ranking_component_comparison.csv')
    ranking['best_set_intersects_reference'] = ranking['top1_hit']
    ranking['top1_hit'] = ranking.reference_regret <= TIE_ATOL
    ranking['selection_rule'] = 'lowest action index among best method scores'
    ranking.to_csv(output/'corrected_action_ranking.csv.gz', index=False, compression='gzip')
    ranking.groupby(['reference','method']).agg(
        top1_hit=('top1_hit','mean'), best_set_intersection=('best_set_intersects_reference','mean'),
        mean_regret=('reference_regret','mean'),
    ).to_csv(output/'corrected_ranking_summary.csv')
    (output/'manifest.json').write_text(json.dumps(summary,indent=2)+'\n')
    (output/'REVIEW_CORRECTIONS.md').write_text(
        '# Review corrections\n\n'
        f'Existing {len(stats)} states and {len(branch)} branches reused; no retraining/replay.\n\n'
        f'Prior task rewards depend on the forced action in {summary["pending_affected_states"]} states '
        f'({len(changed)} prior tasks). Including pending outcomes changes the sampled optimal set in '
        f'{summary["corrected_optimal_set_changed_states"]} states.\n\n'
        'The new all-outcomes event return includes own, future-decision and preexisting pending outcomes. '
        'Policy centering removes action-independent constants; prior-task rewards are not presumed unaffected.\n\n'
        'The historical non-conflict group accounts for the positive PPO-minus-SafeMin regret gap; '
        'the conflict group offsets part of it. See performance_gap_decomposition.csv. '
        'These are fixed-continuation contrasts, not causal attribution of training failure.\n\n'
        'Top-1 now refers to the explicit selected lowest-index tied-best action in ranking tables; '
        'best-set intersection is a separate metric. Actual stochastic choices remain in the deployment-action tables.\n\n'
        'One continuation per action is insufficient for an expected-Q oracle. Repeat continuations with '
        'paired exogenous seeds and estimate state/action means and uncertainty before selecting a causal root cause. '
        'The historical decision-vs-event comparison does not test omitted pending-outcome credit.\n')
    return {k:v for k,v in summary.items() if k!='source_provenance_sha256'}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,default=ROOT/'diagnostics/results/async_task_credit_audit')
    p.add_argument('--output',type=Path,default=ROOT/'diagnostics/results/review_credit_corrections')
    args=p.parse_args(); print(json.dumps(run(args.source,args.output),indent=2))
