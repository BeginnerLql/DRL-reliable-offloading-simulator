"""One formal-config Pair PPO run with observation-only training/evaluation logging.

The subclass delegates all action sampling and optimizer updates to the
unchanged PPOAgent. Extra forwards use no_grad and do not consume randomness.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch
from torch.distributions import Categorical

from config.params import params
from core.main_loop import MainLoop
from tools.pair_policy_diagnostics import DiagnosticPPOAgent, TASK_ASSIGNMENT_COLUMNS
from tools.paired_ppo_experiment import (
    _agent_kwargs, _run_evaluation, _state_dict_snapshot, scoped_environment_seeds,
    sha256_file,
)
from Project_main import build_pair_correlations

FORMAL = ROOT / "diagnostics/paired_ppo_experiment/formal_10seed_300ep"
DEFAULT_OUT = ROOT / "diagnostics/results/policy_oracle_alignment"


class InstrumentedPairAgent(DiagnosticPPOAgent):
    """Record actor outputs and pre-update GAE without changing PPO operations."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.decision_archive = []
        self.rollout_archive = []
        self.episode_number = 0

    def select_action(self, state, epsilon=0.0, use_softmax=False, temperature=1.5):
        selected = super().select_action(state, epsilon, use_softmax, temperature)
        with torch.no_grad():
            logits = self.policy_old(self._to_tensor(state)).detach()
            distribution = Categorical(logits=logits)
            probabilities = distribution.probs.cpu().numpy()
            self.decision_archive.append({
                "state": np.asarray(state, dtype=np.float32).copy(),
                "sampled_action": int(selected),
                "greedy_action": int(torch.argmax(logits).item()),
                "raw_logits": logits.cpu().numpy().astype(float).tolist(),
                "probabilities": probabilities.astype(float).tolist(),
                "entropy": float(distribution.entropy().item()),
            })
        return selected

    def train_step(self):
        self.episode_number += 1
        n = len(self.states)
        assert n == len(self.decision_archive) - len(self.rollout_archive)
        if n:
            states = self._to_tensor(self.states)
            next_states = self._to_tensor(self.next_states)
            with torch.no_grad():
                values = self.value_net(states)
                next_values = self.value_net(next_states)
                rewards = torch.as_tensor(self.rewards, dtype=torch.float32, device=self.device) * float(self.reward_scale)
                dones = torch.as_tensor(self.dones, dtype=torch.float32, device=self.device)
                delta_times = torch.as_tensor(self.delta_times, dtype=torch.float32, device=self.device)
                gamma_k = torch.pow(torch.full_like(delta_times, float(self.gamma)), delta_times)
                nonterminal = 1.0 - dones
                deltas = rewards + gamma_k * next_values * nonterminal - values
                raw = torch.zeros_like(rewards)
                gae = torch.tensor(0.0, dtype=torch.float32, device=self.device)
                for k in reversed(range(n)):
                    gae = deltas[k] + gamma_k[k] * self.gae_lambda * nonterminal[k] * gae
                    raw[k] = gae
                returns = raw + values
                mean = raw.mean()
                std = raw.std(unbiased=False)
                normalized = raw - mean if std.item() < 1e-8 else (raw - mean) / (std + 1e-8)
                start = len(self.rollout_archive)
                for k in range(n):
                    decision = self.decision_archive[start + k]
                    assert int(decision["sampled_action"]) == int(self.actions[k])
                    assert np.array_equal(decision["state"], np.asarray(self.states[k], dtype=np.float32))
                    self.rollout_archive.append({
                        "Episode":self.episode_number,"Task_ID":int(self.task_ids[k]),
                        "timestep":start+k,"sampled_action":int(self.actions[k]),
                        "greedy_action":decision["greedy_action"],
                        "state":decision["state"].astype(float).tolist(),
                        "raw_logits":decision["raw_logits"],
                        "probabilities":decision["probabilities"],
                        "entropy":decision["entropy"],
                        "reward":float(self.rewards[k]),
                        "value":float(values[k].item()),"next_value":float(next_values[k].item()),
                        "return":float(returns[k].item()),
                        "raw_advantage":float(raw[k].item()),
                        "normalized_advantage":float(normalized[k].item()),
                        "old_log_probability":float(self.old_log_probs[k]),
                        "delta_t":float(self.delta_times[k]),"done":bool(self.dones[k]),
                    })
        return super().train_step()


def encode_lists(frame: pd.DataFrame) -> pd.DataFrame:
    result=frame.copy()
    for column in ("state","raw_logits","probabilities","masked_logits"):
        if column in result:
            result[column]=result[column].map(json.dumps)
    return result


def run(train_episodes:int, eval_episodes:int, output:Path, trial_id:int=0):
    metadata=json.loads((FORMAL/"metadata.json").read_text())
    if train_episodes != metadata["train_episodes"] and train_episodes != 2:
        raise ValueError("Only formal 300-episode run or 2-episode equivalence smoke is supported")
    if eval_episodes != metadata["eval_episodes"] and train_episodes != 2:
        raise ValueError("Formal evaluation uses 20 episodes")
    for name,key in (("server_info.xlsx","server_info_sha256"),("task_parameters.xlsx","task_parameters_sha256")):
        if sha256_file(ROOT/"data"/name)!=metadata[key]:
            raise RuntimeError(f"Input {name} differs from formal experiment")
    seed_plan=pd.read_csv(FORMAL/"seed_plan.csv").set_index("Trial_ID")
    trial=seed_plan.loc[trial_id]
    pairs,rho=build_pair_correlations()
    assert len(pairs)==params.num_actions==28 and params.num_states==35
    output.mkdir(parents=True,exist_ok=True)
    torch.manual_seed(int(trial.Torch_Init_Seed))
    agent=InstrumentedPairAgent(**_agent_kwargs("pair_scoring",np.asarray(rho,dtype=float),int(trial.PPO_Minibatch_Seed)))
    torch.manual_seed(int(trial.Train_Action_Seed))
    with scoped_environment_seeds(int(trial.Train_Arrival_Seed),int(trial.Train_Spatial_Seed)):
        loop=MainLoop(agent,train_episodes,metadata["tasks_per_episode"],params.num_states,params.num_actions)
        loop.EP()
    assert len(agent.rollout_archive)==train_episodes*metadata["tasks_per_episode"]
    assert len(loop.task_Assignments_info)==len(agent.rollout_archive)
    training=pd.DataFrame(agent.rollout_archive)
    assignments=pd.DataFrame(loop.task_Assignments_info,columns=TASK_ASSIGNMENT_COLUMNS)
    selected=assignments[["episode","task_id","Primary","Backup","Reliability_Requirement","Task_Reward",
        "Task_Delay","Base_Reward","Reliability_Violation_Log10","Reliability_Penalty"]]
    training=training.merge(selected,left_on=["Episode","Task_ID"],right_on=["episode","task_id"],
                            how="left",validate="one_to_one")
    assert len(training)==len(agent.rollout_archive) and training.Reliability_Requirement.notna().all()
    assert np.allclose(training.reward,training.Task_Reward,rtol=0,atol=1e-8)
    training["R_req"]=training.Reliability_Requirement.astype(float)
    training["normalized_R_req_feature"]=[float(np.asarray(s)[-1]) for s in training.state]
    training["masked_logits"]=training.raw_logits.map(list)  # no extra action mask exists
    encode_lists(training).to_csv(output/"training_rollout.csv.gz",index=False,compression="gzip")
    assignments.to_csv(output/"training_task_assignments.csv",index=False)
    pd.DataFrame(loop.log_data,columns=["Episode","Rolling_Reward","Episode_Reward","Rolling_Delay"]).to_csv(output/"training_curve.csv",index=False)
    actor_path=output/"actor_final.pt"; critic_path=output/"critic_final.pt"
    torch.save(agent.policy_old.state_dict(),actor_path)
    torch.save(agent.value_net.state_dict(),critic_path)
    train_hashes={name:sha256_file(output/name) for name in ("actor_final.pt","critic_final.pt")}
    # Evaluation deep-copies the agent; discard diagnostic-only archives first.
    agent.observation_archive=[]; agent.decision_archive=[]; agent.rollout_archive=[]
    frozen,eval_loop=_run_evaluation("pair_scoring",agent,trial,eval_episodes,metadata["tasks_per_episode"],rho)
    evaluation=pd.DataFrame(eval_loop.task_Assignments_info,columns=TASK_ASSIGNMENT_COLUMNS)
    evaluation.to_csv(output/"evaluation_task_assignments.csv",index=False)
    states=frozen.observation_archive
    assert len(states)==len(evaluation)==eval_episodes*metadata["tasks_per_episode"]
    records=[]
    for index,entry in enumerate(states):
        episode=index//metadata["tasks_per_episode"]+1
        task_id=int(entry["task_id"])
        assert task_id==index%metadata["tasks_per_episode"]+1
        state=np.asarray(entry["state"],dtype=np.float32)
        with torch.no_grad():
            logits=agent.policy_old(torch.as_tensor(state,dtype=torch.float32))
            probs=torch.softmax(logits,dim=-1)
        row=evaluation[(evaluation.episode==episode)&(evaluation.task_id==task_id)]
        assert len(row)==1 and int(row.iloc[0].action_index)==int(entry["action"])
        records.append({"Episode":episode,"Task_ID":task_id,"state":state.astype(float).tolist(),
            "R_req":float(row.iloc[0].Reliability_Requirement),
            "raw_logits":logits.cpu().numpy().astype(float).tolist(),
            "masked_logits":logits.cpu().numpy().astype(float).tolist(),
            "probabilities":probs.cpu().numpy().astype(float).tolist(),
            "greedy_action":int(torch.argmax(logits).item()),
            "sampled_action":None,"executed_action":int(entry["action"]),
            "sampling_mode":"greedy_no_sampling"})
    encode_lists(pd.DataFrame(records)).to_csv(output/"evaluation_actor_outputs.csv",index=False)
    pd.DataFrame(eval_loop.log_data,columns=["Episode","Rolling_Reward","Episode_Reward","Rolling_Delay"]).to_csv(output/"evaluation_curve.csv",index=False)
    result={"trial_id":trial_id,"train_episodes":train_episodes,"eval_episodes":eval_episodes,
            "tasks_per_episode":metadata["tasks_per_episode"],"formal_metadata_commit":metadata["git_commit"],
            "source_head_at_run":subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),"checkpoint_sha256":train_hashes,
            "training_rollout_rows":len(training),"evaluation_state_rows":len(records),
            "no_additional_action_mask":True,"evaluation_mode":"greedy_no_sampling",
            "formal_seed_row":{column:int(trial[column]) for column in trial.index}}
    (output/"run_metadata.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    return result


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--train-episodes",type=int,default=300)
    parser.add_argument("--eval-episodes",type=int,default=20)
    parser.add_argument("--trial-id",type=int,default=0)
    parser.add_argument("--output-dir",type=Path,default=DEFAULT_OUT)
    args=parser.parse_args()
    print(json.dumps(run(args.train_episodes,args.eval_episodes,args.output_dir,args.trial_id),indent=2))
