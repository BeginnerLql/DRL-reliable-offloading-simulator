"""Independent post-run checks for every formal paired seed."""
from __future__ import annotations
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
from diagnostics.aggregate_masked_pair_ppo_10seed import OUT
from diagnostics.run_masked_pair_ppo_10seed import PRIOR, formal_spec


def validate():
    meta,plan=formal_spec()
    rows=[]
    for trial in range(10):
        base=OUT/'runs'/f'trial_{trial:03d}'
        completed=json.loads((base/'completed.json').read_text())
        if len(completed['runs'])!=4:raise RuntimeError(f'Trial {trial}: missing policy')
        run={r['policy']:r for r in completed['runs']}
        if run['masked_stochastic']['actor_sha256']!=run['masked_greedy']['actor_sha256'] or run['masked_stochastic']['critic_sha256']!=run['masked_greedy']['critic_sha256']:
            raise RuntimeError(f'Trial {trial}: masked checkpoint mismatch')
        previous=None
        for policy in ('flat','pair','masked_stochastic','masked_greedy'):
            sub=base/policy
            a=pd.read_csv(sub/'evaluation_task_assignments.csv').sort_values(['episode','task_id']).reset_index(drop=True)
            d=pd.read_csv(sub/'evaluation_decisions.csv').sort_values(['episode','task_id']).reset_index(drop=True)
            if len(a)!=4000 or len(d)!=4000 or not np.array_equal(a.action_index,d.action_index):
                raise RuntimeError(f'Trial {trial} {policy}: incomplete evaluation')
            if not np.isfinite(a[['Task_Reward','Task_Delay','Execution_Reliability']].to_numpy(dtype=float)).all():
                raise RuntimeError(f'Trial {trial} {policy}: nonfinite evaluation')
            if policy=='masked_stochastic':previous=d.safe_mask.tolist()
            if policy=='masked_greedy' and previous!=d.safe_mask.tolist():
                raise RuntimeError(f'Trial {trial}: masked deployment safe masks differ')
            if policy.startswith('masked'):
                support=np.asarray([json.loads(x) for x in d.effective_mask],dtype=bool)
                actions=d.action_index.to_numpy(dtype=int)
                if not np.all(support[np.arange(len(d)),actions]):
                    raise RuntimeError(f'Trial {trial} {policy}: unsupported selected action')
                empty=d.safe_set_empty.astype(bool).to_numpy()
                if np.any(~empty & ~d.selected_action_safe.astype(bool).to_numpy()):
                    raise RuntimeError(f'Trial {trial} {policy}: avoidable violation')
                if np.any(np.abs(d.loc[empty,'selected_pair_reliability'].to_numpy(dtype=float)-d.loc[empty,'best_achievable_reliability'].to_numpy(dtype=float))>1e-12):
                    raise RuntimeError(f'Trial {trial} {policy}: fallback not max reliability')
                if not np.isfinite(d[['old_log_probability','masked_entropy','reliability_deficit']].to_numpy(dtype=float)).all():
                    raise RuntimeError(f'Trial {trial} {policy}: nonfinite masked diagnostics')
            rows.append({'trial_id':trial,'policy':policy,'evaluated_tasks':len(a),
                         'avoidable_violations':int(((~d.safe_set_empty.astype(bool)) & (~a.Reliability_Satisfied.astype(bool))).sum())})
        for mode in ('flat','pair','masked'):
            sub=base/mode
            if not all((sub/name).exists() for name in ('actor.pt','critic.pt','config.json','seed_metadata.json','training_summary.csv','evaluation_summary.csv')):
                raise RuntimeError(f'Trial {trial} {mode}: missing checkpoint or summary')
            curve=pd.read_csv(sub/'training_curve.csv')
            if len(curve)!=300:raise RuntimeError(f'Trial {trial} {mode}: incomplete training')
            if mode=='masked':
                updates=pd.read_csv(sub/'ppo_updates.csv')
                if len(updates)!=300 or not np.isfinite(updates.select_dtypes(include=[np.number]).to_numpy()).all() or updates.first_minibatch_ratio_max_abs_error.max()>1e-5:
                    raise RuntimeError(f'Trial {trial}: invalid PPO updates')
            else:
                reference=PRIOR/'runs'/f'trial_{trial:03d}'/('flat' if mode=='flat' else 'pair_scoring')/'training_curve.csv'
                if reference.exists():
                    old=pd.read_csv(reference)
                    if not np.allclose(curve.episode_reward,old.Episode_Reward,rtol=0,atol=1e-10):
                        raise RuntimeError(f'Trial {trial} {mode}: historical training mismatch')
    result=pd.DataFrame(rows)
    result.to_csv(OUT/'sanity_checks.csv',index=False)
    summary={'trials':10,'policies_per_trial':4,'evaluated_tasks_per_policy':4000,
             'training_episodes_per_agent':300,'masked_updates_per_seed':300,
             'masked_deployments_share_checkpoint':True,'masked_deployments_share_safe_masks':True,
             'masked_selected_actions_in_effective_support':True,'fallback_max_reliability':True,
             'nonfinite_values_found':False,'masked_avoidable_violations':int(result[result.policy.str.startswith('masked')].avoidable_violations.sum())}
    (OUT/'sanity_checks.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2))
    return summary

if __name__=='__main__':validate()
