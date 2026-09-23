"""Validate saved frozen-selector masks, paired traces, and server counters."""
from __future__ import annotations
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
from PIL import Image
from diagnostics.evaluate_reliability_masked_policy import POLICIES
from tools.paired_ppo_experiment import sha256_file

RESULTS=ROOT/'diagnostics/results/reliability_masked_policy'
PRIOR=ROOT/'diagnostics/results/policy_oracle_alignment'

def validate(root=RESULTS, prior=PRIOR):
    paired=pd.read_csv(root/'paired_action_selection_comparison.csv')
    assert len(paired)==4000
    for mode in POLICIES[1:]:
        assert np.array_equal(paired.greedy_safe_set_size,paired[f'{mode}_safe_set_size'])
    loads=pd.read_csv(root/'server_load_by_policy.csv')
    checks={}
    for mode in POLICIES:
        frame=pd.read_csv(root/f'action_diagnostics_{mode}.csv')
        before=np.stack(frame.probability_before.map(json.loads))
        after=np.stack(frame.probability_after.map(json.loads))
        safe=np.stack(frame.safe_mask.map(json.loads)).astype(bool)
        empty=frame.safe_set_empty.to_numpy(dtype=bool)
        assert before.shape==after.shape==safe.shape==(4000,28)
        assert np.isfinite(before).all() and np.isfinite(after).all()
        assert np.max(np.abs(before.sum(axis=1)-1))<1e-6
        if mode.startswith('masked'):
            assert np.max(np.abs(after[~safe]))==0
            assert np.max(np.abs(after[empty]))==0
            assert np.max(np.abs(after[~empty].sum(axis=1)-1))<1e-6
            assert frame.selected_action_safe.to_numpy(dtype=bool)[~empty].all()
        else:
            assert np.array_equal(before,after)
        assignments=pd.read_csv(root/f'task_assignments_{mode}.csv')
        assert np.array_equal(assignments.action_index,frame.selected_action)
        assert np.allclose(assignments.Execution_Reliability,frame.selected_pair_reliability,atol=1e-12,rtol=0)
        assert np.array_equal(assignments.Reliability_Satisfied.astype(bool),frame.selected_action_safe.astype(bool))
        servers=loads[loads.policy==mode]
        assert servers.selection_count.sum()==servers.replica_wait_count.sum()==8000
        assert (servers.busy_time<=servers.observation_time+1e-8).all()
        checks[mode]={'task_count':len(frame),'empty_safe_count':int(empty.sum()),
            'max_before_probability_sum_error':float(np.max(np.abs(before.sum(axis=1)-1))),
            'replica_count':int(servers.selection_count.sum())}
    for previous,current in (('evaluation_task_assignments.csv','task_assignments_greedy.csv'),
                             ('sampled_evaluation_seed2026.csv','task_assignments_native_sample.csv')):
        a=pd.read_csv(prior/previous).sort_values(['episode','task_id']).reset_index(drop=True)
        b=pd.read_csv(root/current).sort_values(['episode','task_id']).reset_index(drop=True)
        assert a.equals(b)
    images=list(root.glob('*.png'))
    for path in images:
        with Image.open(path) as im: im.verify()
    metadata=json.loads((root/'run_metadata.json').read_text())
    assert metadata['checkpoint_sha256']['actor_final.pt']==sha256_file(prior/'actor_final.pt')
    assert metadata['external_arrival_spatial_risk_streams_identical']
    assert metadata['frozen_weights_unchanged']
    result={'matched_task_count':len(paired),'safe_set_sizes_identical_across_policies':True,
        'greedy_and_native_baselines_match_prior_4000_rows_exactly':True,
        'exogenous_streams_identical':True,'checkpoint_unchanged':True,
        'validated_png_count':len(images),'policies':checks}
    (root/'validation.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    return result

if __name__=='__main__':
    print(json.dumps(validate(),indent=2))
