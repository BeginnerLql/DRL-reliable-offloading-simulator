"""Step 6: fixed-hyperparameter PPO training and paired frozen-policy evaluation."""

import argparse
import contextlib
import copy
import io
import os
from pathlib import Path
import subprocess
import sys
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import numpy as np
import pandas as pd
import torch
from scipy.stats import t as student_t

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agents.ppo_agent import PPOAgent
from config.configuration import parameters
from config.params import params
from config.paths import SERVER_INFO_PATH, TRAIN_TASK_PATH, EVALUATION_TASK_PATH, RESULTS_DIR
from core.main_loop import MainLoop
from core.spatial_risk import (
    build_distance_matrix, compute_baseline_correlation_length_km,
    build_spatial_correlation_matrix, sample_spatial_risk_field,
    map_spatial_risk_to_effective_failure_rates,
)
from tools.scan_random_load import make_environment, run_episode, collect_episode_metrics, TASK_METRICS, SERVER_METRICS
from tools.validate_spatial_reliability import markdown_table

RSR_COLUMNS = ["rsr_0_9", "rsr_0_99", "rsr_0_999", "rsr_0_9999"]
EVAL_METRICS = TASK_METRICS + ["episode_raw_reward", "overall_rsr", "mean_pair_reliability", "pair_hhi"] + RSR_COLUMNS
COMPARISON_METRICS = ["mean_latency_sec", "p95_latency_sec", "mean_cpu_waiting_time_sec", "episode_raw_reward", "overall_rsr"]


def derived_seed(master, key):
    return int(np.random.SeedSequence(master, spawn_key=key).generate_state(1)[0])


def environment_basis(master, key, correlation, base_rates):
    arrival_rng, risk_rng, random_rng = [
        np.random.default_rng(np.random.SeedSequence(master, spawn_key=(*key, stream)))
        for stream in range(3)
    ]
    normal = risk_rng.standard_normal(8)
    risk = sample_spatial_risk_field(correlation, standard_normal=normal)
    return {
        "arrival_base": arrival_rng.exponential(size=200),
        "standard_normal": normal, "risk": risk,
        "effective_rates": map_spatial_risk_to_effective_failure_rates(base_rates, risk, params.SPATIAL_RISK_BETA_P),
        "actions": random_rng.integers(28, size=200),
    }


def build_agent(agent_class, device):
    return agent_class(
        num_states=params.num_states, num_actions=params.num_actions,
        hidden_layers=params.hidden_layers_ppo, device=device,
        gamma=params.gamma_ppo, actor_lr=params.actor_lr_ppo, critic_lr=params.critic_lr_ppo,
        clip_eps=params.clip_eps_ppo, k_epochs=params.k_epochs_ppo,
        batch_size=params.batch_size_ppo, entropy_coef=params.entropy_coef_ppo,
        reward_scale=params.reward_scale_ppo, gae_lambda=params.gae_lambda_ppo,
        value_loss_coef=params.value_loss_coef_ppo, max_grad_norm=params.max_grad_norm_ppo,
        activation=params.af_ppo,
    )


def parameter_vector(network):
    return torch.cat([parameter.detach().flatten() for parameter in network.parameters()])


class TrackedPPO(PPOAgent):
    """Observe the existing update; do not replace its loss, sampling or optimizer."""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.policy_steps = self.value_steps = 0
        self.optimizer_policy.register_step_post_hook(self._policy_stepped)
        self.optimizer_value.register_step_post_hook(self._value_stepped)
        self.initial_actor = parameter_vector(self.policy_net).clone()
        self.last_update = {}

    def _policy_stepped(self, optimizer, args, kwargs):
        self.policy_steps += 1

    def _value_stepped(self, optimizer, args, kwargs):
        self.value_steps += 1

    def train_step(self):
        assert len(self.states) == len(self.actions) == len(self.rewards) == 200
        assert np.asarray(self.states).shape == np.asarray(self.next_states).shape == (200, 41)
        assert np.isfinite(self.states).all() and np.isfinite(self.next_states).all()
        assert all(0 <= action < 28 for action in self.actions)
        assert self.dones == [False] * 199 + [True]
        tasks = list(self.loop.env_state.tasks.values())
        arrivals = np.array([task.arrival_time for task in tasks])
        terminal_time = max(arrivals[-1], max(task.task_completion_time for task in tasks))
        np.testing.assert_allclose(self.delta_times[:-1], np.diff(arrivals), rtol=0, atol=1e-10)
        np.testing.assert_allclose(self.delta_times[-1], terminal_time-arrivals[-1], rtol=0, atol=1e-10)
        assert np.isfinite(self.delta_times).all() and min(self.delta_times) >= 0
        raw_reward = sum(self.rewards)
        np.testing.assert_allclose(raw_reward, -sum(task.task_completion_time-task.arrival_time for task in tasks), rtol=0, atol=1e-9)
        actor_before, critic_before = parameter_vector(self.policy_net).clone(), parameter_vector(self.value_net).clone()
        policy_steps, value_steps = self.policy_steps, self.value_steps
        super().train_step()
        expected_steps = params.k_epochs_ppo * int(np.ceil(200 / params.batch_size_ppo))
        assert self.policy_steps-policy_steps == self.value_steps-value_steps == expected_steps
        actor_delta = torch.linalg.vector_norm(parameter_vector(self.policy_net)-actor_before).item()
        critic_delta = torch.linalg.vector_norm(parameter_vector(self.value_net)-critic_before).item()
        assert actor_delta > 0 and critic_delta > 0
        for network in [self.policy_net, self.policy_old, self.value_net]:
            assert torch.isfinite(parameter_vector(network)).all()
        for network in [self.policy_net, self.value_net]:
            gradient_norm = torch.sqrt(sum(p.grad.square().sum() for p in network.parameters() if p.grad is not None)).item()
            assert np.isfinite(gradient_norm) and gradient_norm > 0
        for key, value in self.policy_net.state_dict().items():
            assert torch.equal(value, self.policy_old.state_dict()[key])
        assert all(not buffer for buffer in [self.states, self.actions, self.rewards, self.next_states, self.dones, self.old_log_probs, self.delta_times])
        self.last_update = {
            "transition_count": 200,
            "policy_optimizer_steps": self.policy_steps-policy_steps,
            "value_optimizer_steps": self.value_steps-value_steps,
            "actor_update_l2": actor_delta, "critic_update_l2": critic_delta,
            "actor_change_from_initial_l2": torch.linalg.vector_norm(parameter_vector(self.policy_net)-self.initial_actor).item(),
        }
        self.last_raw_reward = raw_reward


class TrainingLoop(MainLoop):
    """Inject Step 6 environment randomness without replacing MainLoop.Iteration."""
    def __init__(self, agent, arrival_rate, rate_index, seed_id, master, episodes, distances, correlation, base_rates, callback):
        self.arrival_rate, self.rate_index, self.seed_id, self.master = arrival_rate, rate_index, seed_id, master
        self.distances, self.correlation, self.base_rates = distances, correlation, base_rates
        super().__init__(agent, episodes, 200, params.num_states, params.num_actions, episode_callback=callback)
        assert self.model_name == "ppo"

    def _initialize_episode_spatial_risk(self):
        self.basis = environment_basis(self.master, (6, 2, self.rate_index, self.seed_id, self.this_episode), self.correlation, self.base_rates)
        self.env_state.set_episode_spatial_risk_context(
            list(self.server_profiles), self.distances, self.correlation,
            self.basis["risk"], self.basis["effective_rates"],
        )

    def _sample_interarrival_time(self):
        return self.basis["arrival_base"][self.taskCounter-1] / self.arrival_rate


def add_behavior_metrics(metrics, tasks, pairs):
    actions = [pairs.index((task.replica_A["server_id"], task.replica_B["server_id"])) for task in tasks]
    counts = np.bincount(actions, minlength=28)
    requirements = np.array([task.reliability_requirement for task in tasks])
    reliability = np.array([task.pair_reliability for task in tasks])
    assert np.isfinite(reliability).all() and ((reliability > 0) & (reliability <= 1)).all()
    satisfied = reliability >= requirements
    metrics.update({
        "episode_raw_reward": -sum(task.task_completion_time-task.arrival_time for task in tasks),
        "overall_rsr": satisfied.mean(), "mean_pair_reliability": reliability.mean(),
        "pair_hhi": np.sum((counts/200)**2),
        **{column: satisfied[requirements == level].mean() for column, level in zip(RSR_COLUMNS, params.TASK_RELIABILITY_REQUIREMENT_LEVELS)},
        **{f"action_count_{index}": count for index, count in enumerate(counts)},
    })
    return metrics


def checkpoint_snapshot(agent):
    return copy.deepcopy({
        "actor": agent.policy_net.state_dict(), "old_actor": agent.policy_old.state_dict(),
        "critic": agent.value_net.state_dict(), "policy_optimizer": agent.optimizer_policy.state_dict(),
        "value_optimizer": agent.optimizer_value.state_dict(),
        "rollout_lengths": [len(b) for b in [agent.states, agent.actions, agent.rewards, agent.next_states, agent.dones, agent.old_log_probs, agent.delta_times]],
    })


def assert_same_tree(left, right):
    if isinstance(left, torch.Tensor):
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            assert_same_tree(left[key], right[key])
    elif isinstance(left, (tuple, list)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            assert_same_tree(a, b)
    else:
        assert left == right


def assert_paired_environment(reference, tasks):
    assert len(reference) == len(tasks) == 200
    for a, b in zip(reference, tasks):
        assert (a.id, a.task_size, a.computation_demand, a.reliability_requirement, a.arrival_time) == (
            b.id, b.task_size, b.computation_demand, b.reliability_requirement, b.arrival_time
        )
    a_state, b_state = reference[0].env_state, tasks[0].env_state
    np.testing.assert_array_equal(a_state.spatial_risk_field, b_state.spatial_risk_field)
    assert a_state.effective_failure_rates == b_state.effective_failure_rates


def evaluate_checkpoint(path, method, args, rate, seed_id, training_seed, server_profiles, task_profiles, pairs, distances, correlation, base_rates, references):
    rows, server_rows = [], []
    devices = [] if args.device == "cpu" else [torch.device(args.device).index or 0]
    with torch.random.fork_rng(devices=devices):
        frozen = build_agent(PPOAgent, args.device)
        frozen.load_model(path)
        before = checkpoint_snapshot(frozen)
        for episode_id in range(1, args.eval_episodes+1):
            basis = environment_basis(args.master_seed, (6, 3, seed_id, episode_id), correlation, base_rates)
            torch.manual_seed(derived_seed(args.master_seed, (6, 3, seed_id, episode_id, 3)))

            def selector(state):
                assert state.shape == (41,) and np.isfinite(state).all()
                action = frozen.select_action(state, 0.)
                assert 0 <= action < 28
                return action

            metrics, server_metrics, tasks = run_episode(rate, episode_id, basis, server_profiles, task_profiles, pairs, distances, correlation, action_selector=selector)
            assert_paired_environment(references[episode_id-1], tasks)
            tags = {"seed_id": seed_id, "training_seed": training_seed, "method": method}
            rows.append({**tags, **add_behavior_metrics(metrics, tasks, pairs)})
            server_rows.extend({**tags, **record} for record in server_metrics)
        assert_same_tree(before, checkpoint_snapshot(frozen))
    return rows, server_rows


def run_experiment(args, rate_index, rate, seed_id, server_profiles, training_profiles, evaluation_profiles, pairs, distances, correlation, base_rates):
    training_seed = derived_seed(args.master_seed, (6, 0, seed_id))
    init_seed, action_seed, shuffle_seed = [derived_seed(args.master_seed, (6, 1, rate_index, seed_id, role)) for role in range(3)]
    np.random.seed(shuffle_seed)
    torch.manual_seed(init_seed)
    agent = build_agent(TrackedPPO, args.device)
    torch.manual_seed(action_seed)
    name = f"rate_{rate:g}_seed_{seed_id:02d}"
    initial_path = args.output_dir / "checkpoints" / f"{name}_initial.pt"
    final_path = args.output_dir / "checkpoints" / f"{name}_final.pt"
    agent.save_model(initial_path)
    train_rows = []
    started = time.monotonic()

    def record_episode(loop):
        tasks = list(loop.env_state.tasks.values())
        metrics, _, _ = collect_episode_metrics(rate, loop.this_episode, tasks, loop.env_state, loop.env.now)
        add_behavior_metrics(metrics, tasks, pairs)
        np.testing.assert_allclose(metrics["episode_raw_reward"], loop.episodic_reward, rtol=0, atol=1e-9)
        np.testing.assert_allclose(metrics["episode_raw_reward"], agent.last_raw_reward, rtol=0, atol=1e-9)
        train_rows.append({"seed_id": seed_id, "training_seed": training_seed, "method": "ppo_final", **metrics, **agent.last_update})
        loop.task_results.clear()  # Keep Episode aggregates; no persistent task-level training log.
        if loop.this_episode % 50 == 0 or loop.this_episode == args.train_episodes:
            print(f"rate={rate:g} seed={seed_id} train={loop.this_episode}/{args.train_episodes} latency={metrics['mean_latency_sec']:.4f} elapsed={time.monotonic()-started:.1f}s", file=sys.stderr, flush=True)

    loop = TrainingLoop(agent, rate, rate_index, seed_id, args.master_seed, args.train_episodes, distances, correlation, base_rates, record_episode)
    assert loop.task_profiles == training_profiles
    agent.loop = loop
    with contextlib.redirect_stdout(io.StringIO()):
        loop.EP()
    assert agent.policy_steps == agent.value_steps == args.train_episodes * 8
    agent.save_model(final_path)
    eval_rows, server_rows, references = [], [], []
    for episode_id in range(1, args.eval_episodes+1):
        basis = environment_basis(args.master_seed, (6, 3, seed_id, episode_id), correlation, base_rates)
        metrics, server_metrics, tasks = run_episode(rate, episode_id, basis, server_profiles, evaluation_profiles, pairs, distances, correlation)
        references.append(tasks)
        tags = {"seed_id": seed_id, "training_seed": training_seed, "method": "uniform_random"}
        eval_rows.append({**tags, **add_behavior_metrics(metrics, tasks, pairs)})
        server_rows.extend({**tags, **record} for record in server_metrics)
    for method, path in [("ppo_initial", initial_path), ("ppo_final", final_path)]:
        rows, records = evaluate_checkpoint(path, method, args, rate, seed_id, training_seed, server_profiles, evaluation_profiles, pairs, distances, correlation, base_rates, references)
        eval_rows.extend(rows)
        server_rows.extend(records)
    print(f"rate={rate:g} seed={seed_id} completed train/eval; elapsed={time.monotonic()-started:.1f}s", flush=True)
    return train_rows, eval_rows, server_rows


def seed_statistics(evaluation):
    return evaluation.groupby(["arrival_rate", "seed_id", "training_seed", "method"], sort=True)[EVAL_METRICS].mean().reset_index()


def paired_statistics(seed_summary):
    rows = []
    for rate, group in seed_summary.groupby("arrival_rate"):
        final = group[group.method == "ppo_final"].set_index("seed_id")
        for reference_method in ["uniform_random", "ppo_initial"]:
            reference = group[group.method == reference_method].set_index("seed_id")
            for metric in COMPARISON_METRICS:
                difference = final[metric]-reference[metric]
                count = len(difference)
                sd = difference.std(ddof=1)
                half = student_t.ppf(.975, count-1)*sd/np.sqrt(count) if count > 1 else np.nan
                lower_is_better = metric not in ["episode_raw_reward", "overall_rsr"]
                rows.append({
                    "arrival_rate": rate, "reference_method": reference_method, "metric": metric,
                    "seed_count": count, "ppo_mean": final[metric].mean(), "ppo_sd": final[metric].std(ddof=1),
                    "reference_mean": reference[metric].mean(), "reference_sd": reference[metric].std(ddof=1),
                    "paired_mean_difference": difference.mean(), "paired_difference_sd": sd,
                    "ci95_lower": difference.mean()-half, "ci95_upper": difference.mean()+half,
                    "ppo_better_seed_count": int((difference < 0).sum() if lower_is_better else (difference > 0).sum()),
                    "improvement_mean_pct": ((reference[metric]-final[metric])/reference[metric]*100).mean() if metric in TASK_METRICS[:2]+["mean_cpu_waiting_time_sec"] else np.nan,
                })
    return pd.DataFrame(rows)


def action_distribution(training, evaluation, pairs):
    rows = []
    for phase, frame in [("train", training), ("eval", evaluation)]:
        for (rate, seed_id, training_seed, method), group in frame.groupby(["arrival_rate", "seed_id", "training_seed", "method"], sort=True):
            counts = group[[f"action_count_{i}" for i in range(28)]].sum().to_numpy()
            denominator = len(group)*200
            tags = {"phase": phase, "arrival_rate": rate, "seed_id": seed_id, "training_seed": training_seed, "method": method}
            for index, ((a, b), count) in enumerate(zip(pairs, counts)):
                rows.append({**tags, "entity": "pair", "entity_id": index, "server_a": a, "server_b": b, "frequency": count, "probability": count/denominator})
            for server_id in range(1, 9):
                count = sum(counts[i] for i, pair in enumerate(pairs) if server_id in pair)
                rows.append({**tags, "entity": "server", "entity_id": server_id, "server_a": server_id, "server_b": None, "frequency": count, "probability": count/denominator})
    return pd.DataFrame(rows)


def save_tables(args, train_rows, eval_rows, server_rows, pairs):
    training, evaluation, servers = pd.DataFrame(train_rows), pd.DataFrame(eval_rows), pd.DataFrame(server_rows)
    seeds = seed_statistics(evaluation)
    paired = paired_statistics(seeds)
    actions = action_distribution(training, evaluation, pairs)
    for filename, frame in [
        ("train_episode_metrics.csv", training), ("eval_episode_metrics.csv", evaluation),
        ("eval_server_metrics.csv", servers), ("eval_seed_summary.csv", seeds),
        ("paired_comparison.csv", paired), ("action_distribution.csv", actions),
    ]:
        frame.to_csv(args.output_dir / filename, index=False, float_format="%.17g")
    return training, evaluation, servers, seeds, paired, actions


def make_report_and_plots(args, frames, server_frame, pairs):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    training, evaluation, servers, seeds, paired, actions = frames
    methods = ["ppo_initial", "ppo_final", "uniform_random"]
    for rate, data in training.groupby("arrival_rate"):
        fig, axes = plt.subplots(2, 2, figsize=(10, 7))
        for ax, metric in zip(axes.ravel(), ["episode_raw_reward", "mean_latency_sec", "p95_latency_sec", "mean_cpu_waiting_time_sec"]):
            raw = data.pivot(index="episode_id", columns="seed_id", values=metric)
            smooth = raw.rolling(20, min_periods=1).mean()
            ax.plot(raw.index, raw.mean(axis=1), alpha=.25, label="Raw seed mean")
            ax.plot(smooth.index, smooth.mean(axis=1), label="20-Episode rolling seed mean")
            if smooth.shape[1] > 1:
                mean, sd = smooth.mean(axis=1), smooth.std(axis=1, ddof=1)
                ax.fill_between(smooth.index, mean-sd, mean+sd, alpha=.2)
            ax.set(xlabel="Training Episode", ylabel=metric)
            ax.grid(alpha=.3)
        axes[0, 0].legend(fontsize=7)
        fig.suptitle(f"PPO training, arrival rate {rate:g}: seed mean +/- seed SD")
        fig.tight_layout(); fig.savefig(args.output_dir/f"training_rate_{rate:g}.png", dpi=150); plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, metric in zip(axes, ["mean_latency_sec", "p95_latency_sec", "mean_cpu_waiting_time_sec"]):
        for method_index, method in enumerate(methods):
            group = seeds[seeds.method == method].groupby("arrival_rate")[metric]
            means, sd = group.mean(), group.std(ddof=1).fillna(0.)
            ax.bar(np.arange(len(means))+(method_index-1)*.23, means, width=.23, yerr=sd, capsize=3, label=method)
        ax.set(xticks=[0, 1], xticklabels=["2.0", "3.0"], xlabel="Arrival rate (tasks/s)", ylabel=metric)
        ax.grid(axis="y", alpha=.3)
    axes[0].legend(fontsize=7); fig.suptitle("Frozen evaluation: means +/- SD across training seeds")
    fig.tight_layout(); fig.savefig(args.output_dir/"evaluation.png", dpi=150); plt.close(fig)
    fig, axes = plt.subplots(2, 2, figsize=(12, 7))
    server_seeds = servers.groupby(["arrival_rate", "seed_id", "method", "server_id"])[SERVER_METRICS].mean().reset_index()
    for row, rate in enumerate([2., 3.]):
        for ax, metric in zip(axes[row], ["cpu_utilization_arrival", "assigned_cpu_work_sec"]):
            for i, method in enumerate(methods):
                means = server_seeds[(server_seeds.arrival_rate == rate) & (server_seeds.method == method)].groupby("server_id")[metric].mean()
                ax.bar(np.arange(8)+(i-1)*.23, means, width=.23, label=method)
            ax.set(xticks=np.arange(8), xticklabels=range(1, 9), xlabel="Server_ID", ylabel=metric, title=f"arrival rate {rate:g}")
            ax.grid(axis="y", alpha=.3)
    axes[0, 0].legend(fontsize=7); fig.tight_layout()
    fig.savefig(args.output_dir/"servers.png", dpi=150); plt.close(fig)
    fig, axes = plt.subplots(2, 1, figsize=(12, 7))
    for ax, rate in zip(axes, [2., 3.]):
        for method in methods:
            group = actions[(actions.phase == "eval") & (actions.arrival_rate == rate) & (actions.method == method) & (actions.entity == "pair")]
            ax.plot(range(28), group.groupby("entity_id").probability.mean(), marker=".", label=method)
        ax.set(xticks=range(28), xticklabels=[f"{a}-{b}" for a, b in pairs], ylabel="Pair selection probability", title=f"arrival rate {rate:g}")
        ax.tick_params(axis="x", labelsize=7); ax.grid(alpha=.3)
    axes[0].legend(fontsize=7); fig.tight_layout()
    fig.savefig(args.output_dir/"actions.png", dpi=150); plt.close(fig)
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True).strip()
    branch = subprocess.check_output(["git", "branch", "--show-current"], cwd=PROJECT_ROOT, text=True).strip()
    seed_mapping = []
    for rate_index, rate in enumerate([2., 3.]):
        for seed_id in range(1, args.seed_count+1):
            seed_mapping.append({"arrival_rate": rate, "seed_id": seed_id, "training_seed": derived_seed(args.master_seed, (6, 0, seed_id)), **{name: derived_seed(args.master_seed, (6, 1, rate_index, seed_id, role)) for role, name in enumerate(["init_seed", "action_seed", "shuffle_seed"])}})
    summary_rows = []
    for (rate, method), group in seeds.groupby(["arrival_rate", "method"]):
        summary_rows.append({"arrival_rate": rate, "method": method, **{metric: f"{group[metric].mean():.8g} ± {group[metric].std(ddof=1):.8g}" for metric in EVAL_METRICS}})
    updates = training.groupby(["arrival_rate", "seed_id"]).agg(
        episodes=("episode_id", "count"), transitions=("transition_count", "sum"),
        policy_optimizer_steps=("policy_optimizer_steps", "sum"), value_optimizer_steps=("value_optimizer_steps", "sum"),
        actor_change_from_initial_l2=("actor_change_from_initial_l2", "last"),
        min_actor_update_l2=("actor_update_l2", "min"), min_critic_update_l2=("critic_update_l2", "min"),
    ).reset_index()
    cpu = torch.cuda.get_device_name(0) if args.device.startswith("cuda") else "CPU"
    lines = [
        "# Step 6：PPO 与 Uniform Random 延迟学习能力对比", "",
        f"基准：`{branch}@{head}`。本目录为 {'smoke 路径验证，不作性能结论' if args.smoke else '完整正式实验'}。",
        "", "## 实验设置", "",
        f"主种子={args.master_seed}；到达率 2.0/3.0；每率 {args.seed_count} 独立训练种子，每种子 {args.train_episodes} 训练 Episode 和 {args.eval_episodes} 评估 Episode。每 Episode 200 个任务、400 副本。",
        f"训练用 data/task_parameters.xlsx；评估仅用 data/task_parameters_evaluation.xlsx；服务器固定为 data/server_info.xlsx。设备={args.device}（{cpu}），CPU threads={args.threads}。",
        f"Python {sys.version.split()[0]}；PyTorch {torch.__version__}，CUDA build {torch.version.cuda}；NumPy {np.__version__}，pandas {pd.__version__}，matplotlib {matplotlib.__version__}。",
        "torch.use_deterministic_algorithms(True)，cuDNN benchmark=False，CUBLAS_WORKSPACE_CONFIG=:4096:8。GPU 逐位复现的实测范围是同设备/同依赖下的重复 smoke；没有把 manual_seed 本身作为 GPU 确定性的证明。",
        "每个 arrival_rate×seed 都新建 PPO；网络初始化、Torch 动作、NumPy shuffle 分别设置以下独立派生种子。没有继承另一到达率或种子的权重。",
        "", markdown_table(pd.DataFrame(seed_mapping)), "",
        "训练环境由 (6,2,rate_index,seed_id,episode,role) 派生，role=0 到达、1 风险；未消费 MainLoop 默认环境 RNG。",
        "评估由 (6,3,seed_id,episode,role) 派生，role=0 到达、1 风险、2 Random 动作、3 PPO 动作种子。相同编号环境也在两到达率之间 CRN 配对，再按速率缩放。",
        "初始/最终 PPO 使用相同冻结采样种子作诊断；Random 动作用独立 NumPy 流，与 PPO 的 Torch 流不同。",
        "PPO 超参数和归一化尺度均保留：41 状态、28 动作、[64,32]/tanh、actor LR=1e-4、critic LR=5e-4、clip=0.2、epochs=2、batch=64、entropy=0.01、gamma=0.90、GAE=0.95、value_coef=0.5、grad_norm=0.5、reward_scale=1。",
        "gamma_k=gamma**delta_t；训练使用原 MainLoop completion-event interval reward。每轮原始奖励和=-全部任务延迟和；保留终止 transition 与原 PPO 更新流程。",
        "每个种子保存初始与最后训练更新后的 checkpoint；正式主结果使用 ppo_final stochastic sampling，不选 best checkpoint、不调参。",
        "", "## 训练更新验证", "", markdown_table(updates), "",
        "每轮 200 transitions、8 次 actor 和 8 次 critic optimizer steps（2 epochs×4 minibatches），一次 train_step。检查非零/有限梯度、参数变化、policy_old 同步及 buffer 清空。",
        "原始训练数据与 28 动作频数保留在 train_episode_metrics.csv；平滑窗口固定 20，仅用于绘图，不修改学习。",
        "训练表 method=ppo_final 标识最终 checkpoint 所属的训练运行；训练期间策略持续更新，不代表使用冻结最终策略训练。",
        "", "![训练2.0](training_rate_2.png)", "", "![训练3.0](training_rate_3.png)", "",
        "## 独立冻结评估：跨训练种子 Mean ± SD", "",
        "先对每种子 20 个 Episode 的指标求平均，再汇总 10 个种子。P95 始终先在单 Episode 的 200 任务内计算。",
        "", markdown_table(pd.DataFrame(summary_rows)), "",
        "## 配对比较：ppo_final − reference", "",
        "每个 seed 内同环境配对；95% t CI 使用种子数量作自由度，正式 n=10、df=9，绝不将 200 Episodes 当作 200 个独立训练种子。",
        "时延改善率先算各 seed 的 (reference-PPO)/reference×100，再平均；奖励越大越好，时延越小越好。非时延指标的改善率栏不适用。",
        "", markdown_table(paired.fillna("—")), "",
        "![冻结评估](evaluation.png)", "", "## 服务器与动作行为", "",
        "每台服务器被选中 probability 为含该服务器的任务比例（分母为任务数），8 个概率合计为 2；pair probability 合计为 1。",
        "", "![服务器](servers.png)", "", "![动作](actions.png)", "",
    ]
    for rate in [2., 3.]:
        group = paired[(paired.arrival_rate == rate) & (paired.metric == "mean_latency_sec")]
        random = group[group.reference_method == "uniform_random"].iloc[0]
        initial = group[group.reference_method == "ppo_initial"].iloc[0]
        lines.extend([
            f"### 到达率 {rate:g}", "",
            f"最终−Random 平均延迟差 {random.paired_mean_difference:.8g} s，95% CI [{random.ci95_lower:.8g}, {random.ci95_upper:.8g}]；优于 Random 种子数 {random.ppo_better_seed_count}/{args.seed_count}。",
            f"最终−初始策略差 {initial.paired_mean_difference:.8g} s，95% CI [{initial.ci95_lower:.8g}, {initial.ci95_upper:.8g}]。权重变化本身不等于性能提升。",
        ])
        for method in methods:
            pair_group = actions[(actions.phase == "eval") & (actions.arrival_rate == rate) & (actions.method == method) & (actions.entity == "pair")]
            top = pair_group.groupby(["server_a", "server_b"]).probability.mean().sort_values(ascending=False).head(5).reset_index()
            lines.extend([f"{method} 最常选的服务器对：", "", markdown_table(top), ""])
        server_group = servers[servers.arrival_rate == rate].groupby(["seed_id", "method", "server_id"])[SERVER_METRICS].mean().reset_index()
        server_table = server_group.groupby(["method", "server_id"])[SERVER_METRICS].mean().reset_index()
        lines.extend(["服务器指标先按种子汇总后的总体均值；完整每 Episode 数值见 eval_server_metrics.csv。", "", markdown_table(server_table), ""])
        final_servers = server_table[server_table.method == "ppo_final"].set_index("server_id")
        random_servers = server_table[server_table.method == "uniform_random"].set_index("server_id")
        work_cv = lambda frame: frame.assigned_cpu_work_sec.std(ddof=0) / frame.assigned_cpu_work_sec.mean()
        lines.append(f"8 台服务器平均分配 CPU 工作量的横截面 CV（总体 SD/均值）由 Random 的 {work_cv(random_servers):.4f} 变为 PPO 的 {work_cv(final_servers):.4f}；这是描述性负载指标，不是种子级显著性检验。")
        for server_id in [6, 8]:
            before, after = random_servers.loc[server_id], final_servers.loc[server_id]
            lines.append(f"server{server_id}：CPU 工作量 {before.assigned_cpu_work_sec:.3f}→{after.assigned_cpu_work_sec:.3f} s，时间加权等待队列 {before.mean_waiting_queue_length:.3f}→{after.mean_waiting_queue_length:.3f}。")
        final_pairs = actions[(actions.phase == "eval") & (actions.arrival_rate == rate) & (actions.method == "ppo_final") & (actions.entity == "pair")]
        maximum = final_pairs.groupby("entity_id").probability.mean().max()
        lines.extend([f"跨种子汇总最高服务器对频率 {maximum:.2%}；结合逐种子动作分布与 HHI，未见所有种子共同塌缩到单一服务器对。每 Episode 200 次均匀抽样的经验 HHI 期望为 1/28+(1−1/28)/200≈0.04054，不能把有限样本 HHI 高于 1/28 本身视为学习证据。", ""])
        if not args.smoke:
            if random.ci95_upper < 0:
                conclusion = "最终策略在本轮种子级配对独立评估中优于 Random。"
            elif random.ci95_lower > 0:
                conclusion = "最终策略在本轮独立评估中劣于 Random，不能把训练更新解释为有效延迟学习。"
            else:
                conclusion = "本轮最终策略相对 Random 的种子级置信区间包含零，未得到明确改善证据。"
            learning = "最终策略的平均延迟相对初始策略有改善证据。" if initial.ci95_upper < 0 else "最终策略相对初始策略的平均延迟尚无明确改善证据。"
            lines.extend([conclusion, learning + "该诊断不用于 checkpoint 选择；非均匀动作和参数变化也不能单独证明有效学习。", ""])
    lines.extend([
        "## 验证边界与遗留事项", "",
        "冻结评估仅用 select_action 与共享 Step 5 物理过程；不 store_transition、不 train_step、不 backward。模型/optimizer/rollout 快照前后完全一致，Torch 动作 RNG 由 fork_rng 保存恢复。",
        "逐场景检查 PPO/Random/初始策略的任务属性、到达时间、风险场与有效故障率一致。全部副本排空，按同一时间戳口径统计 CPU 忙碌区间和队列。",
        "RSR 是理论可靠性要求满足率，不是实际失败事件成功率；本轮不使用可靠性 mask/penalty/constrained PPO、不校准可靠性参数、不研究独立风险场泛化。",
        "若未改善，只提出下一轮单因素诊断：区间奖励的动作信用分配、遗漏在途副本的部分可观测性、动作集中与 CPU 工作量分配。此次不修改这些因素，也不根据评估结果调参。",
    ])
    (args.output_dir/"report.md").write_text("\n".join(lines)+"\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--master-seed", type=int, default=parameters.EXPERIMENT_SEED)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=Path(RESULTS_DIR)/"step6_ppo_vs_random")
    args = parser.parse_args()
    args.seed_count, args.train_episodes, args.eval_episodes = (1, 5, 2) if args.smoke else (10, 300, 20)
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "checkpoints").mkdir(exist_ok=True)
    sf = pd.read_excel(SERVER_INFO_PATH, sheet_name="Servers", dtype={"Site_ID": "string"}).sort_values("Server_ID").reset_index(drop=True)
    train = pd.read_excel(TRAIN_TASK_PATH).sort_values("Task_ID").reset_index(drop=True)
    evaluation = pd.read_excel(EVALUATION_TASK_PATH).sort_values("Task_ID").reset_index(drop=True)
    server_profiles = MainLoop._build_server_profiles(sf)
    training_profiles, evaluation_profiles = MainLoop._build_task_profiles(train), MainLoop._build_task_profiles(evaluation)
    assert len(training_profiles) == len(evaluation_profiles) == 200 and training_profiles != evaluation_profiles
    _, state = make_environment(server_profiles)
    _, distances = build_distance_matrix([info["server_object"] for info in state.servers.values()])
    correlation = build_spatial_correlation_matrix(distances, compute_baseline_correlation_length_km(distances))
    base_rates = sf.Base_Failure_Rate.to_numpy()
    pairs = MainLoop.generate_combinations()
    train_rows, eval_rows, server_rows = [], [], []
    for rate_index, rate in enumerate([2., 3.]):
        for seed_id in range(1, args.seed_count+1):
            rows, evals, servers = run_experiment(args, rate_index, rate, seed_id, server_profiles, training_profiles, evaluation_profiles, pairs, distances, correlation, base_rates)
            train_rows.extend(rows); eval_rows.extend(evals); server_rows.extend(servers)
            frames = save_tables(args, train_rows, eval_rows, server_rows, pairs)
    make_report_and_plots(args, frames, sf, pairs)
    print(f"Complete: training={len(train_rows)}, final evaluation={sum(r['method']=='ppo_final' for r in eval_rows)}, Random evaluation={sum(r['method']=='uniform_random' for r in eval_rows)}", flush=True)


if __name__ == "__main__":
    main()
