import pandas as pd
import pytest
from diagnostics.async_task_credit_audit import _ranking_rows
from diagnostics.review_credit_diagnostics import pending_components


def test_top1_is_specific_action_not_any_best_set_intersection():
    row=_ranking_rows('s',[10,11],{'method':[2.,2.]},{'reference':[1.,3.]})[0]
    assert row['best_action_index']==10
    assert not row['top1_hit']
    assert row['best_set_intersects_reference']
    assert row['reference_regret']==2.


def test_pending_reward_variation_is_not_forced_to_zero():
    frame=pd.DataFrame({'task_role':['preexisting_pending','preexisting_pending','future_decision'],
                        'root_decision_time':[1.,1.,1.],'completion_time':[2.,3.,4.],
                        'task_reward':[10.,5.,20.]})
    p=pending_components(frame,.9)
    assert p.pending_event.tolist()==pytest.approx([9.,4.05])
