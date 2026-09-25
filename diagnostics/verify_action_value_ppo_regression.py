"""Replay legacy Masked PPO on seed 0 and compare its full action trace to Q sidecar."""
from __future__ import annotations
import json
import tempfile
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
import torch

from agents.masked_pair_ppo_agent import ReliabilityMaskedPairPPOAgent
from config.params import params
from diagnostics.run_action_value_critic import OUT
from diagnostics.run_masked_pair_ppo_10seed import (
    formal_spec, train as train_masked, OUT as PPO_OUT,
)
from Project_main import build_pair_correlations
from tools.paired_ppo_experiment import _agent_kwargs, scoped_environment_seeds
from tools.pair_policy_diagnostics import TASK_ASSIGNMENT_COLUMNS


def main():
    torch.set_num_threads(1)
    meta,plan=formal_spec(); trial=plan.iloc[0].to_dict()
    rho=np.asarray(build_pair_correlations()[1],dtype=float)
    torch.manual_seed(int(trial["Torch_Init_Seed"]))
    baseline=ReliabilityMaskedPairPPOAgent(**_agent_kwargs("pair_scoring",rho,int(trial["PPO_Minibatch_Seed"])))
    with tempfile.TemporaryDirectory(prefix="masked_ppo_regression_") as temp:
        run_dir=Path(temp)
        train_loop,curve=train_masked(baseline,"masked",trial,run_dir,meta)
        decisions=pd.DataFrame(baseline.selection_archive).sort_values(["episode","task_id"]).reset_index(drop=True)
        assignments=pd.DataFrame(train_loop.task_Assignments_info,columns=TASK_ASSIGNMENT_COLUMNS).sort_values(["episode","task_id"]).reset_index(drop=True)
        q_dir=OUT/"runs/trial_000/masked"
        q_decisions=pd.read_csv(q_dir/"q_training_action_trace.csv").sort_values(["episode","task_id"]).reset_index(drop=True)
        if len(decisions)!=60_000 or len(q_decisions)!=60_000:
            raise RuntimeError("Full training decision trace was not captured")
        if not np.array_equal(decisions[["episode","task_id","action_index"]].to_numpy(),q_decisions[["episode","task_id","action_index"]].to_numpy()):
            mismatch=np.flatnonzero(np.any(decisions[["episode","task_id","action_index"]].to_numpy()!=q_decisions[["episode","task_id","action_index"]].to_numpy(),axis=1))[:10]
            raise RuntimeError(f"Training actions differ from legacy Masked PPO at rows {mismatch.tolist()}")
        q_curve=pd.read_csv(q_dir/"training_curve.csv")
        if not np.allclose(curve.episode_reward,q_curve.episode_reward,rtol=0,atol=1e-9):
            raise RuntimeError("Per-episode training rewards differ")
        if not np.allclose(curve.episode_total_delay,q_curve.episode_total_delay,rtol=0,atol=1e-9):
            raise RuntimeError("Per-episode training delay differs")

    evaluation={}
    for mode in ("masked_stochastic","masked_greedy"):
        current=OUT/"runs/trial_000"/mode
        reference=PPO_OUT/"runs/trial_000"/mode
        a=pd.read_csv(current/"evaluation_episode_metrics.csv").sort_values("episode").reset_index(drop=True)
        b=pd.read_csv(reference/"evaluation_episode_metrics.csv").sort_values("episode").reset_index(drop=True)
        if a.columns.tolist()!=b.columns.tolist(): raise RuntimeError(f"{mode} evaluation metric schema changed")
        av=a.select_dtypes(include=[np.number]).to_numpy();bv=b.select_dtypes(include=[np.number]).to_numpy()
        if not np.allclose(av,bv,rtol=0,atol=1e-10,equal_nan=True):
            raise RuntimeError(f"{mode} evaluation metrics differ")
        evaluation[mode]={"episode_metrics_exact":True,"episodes":int(len(a))}
    result={"trial_id":0,"training_action_count":60000,"training_action_sequence_exact":True,
        "per_episode_reward_exact_with_atol_1e-9":True,"per_episode_delay_exact_with_atol_1e-9":True,
        "evaluation_episode_metrics":evaluation,"counterfactual_training_rows":0}
    (OUT/"ppo_regression_replay.json").write_text(json.dumps(result,indent=2))
    smoke=json.loads((OUT/"smoke_status.json").read_text())
    smoke["regression"]["per_decision_training_action_sequence_reference_available"]=True
    smoke["regression"]["training_action_sequence_exact"]=True
    smoke["evaluation_regression"]["evaluation_episode_metrics_exact"]=True
    (OUT/"smoke_status.json").write_text(json.dumps(smoke,indent=2))
    print(json.dumps(result,indent=2))


if __name__=="__main__":main()
