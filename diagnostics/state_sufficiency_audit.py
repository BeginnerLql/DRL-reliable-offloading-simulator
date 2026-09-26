"""Offline state-sufficiency audit helpers and read-only upload telemetry."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd
import torch
from scipy.spatial import cKDTree
import core.main_loop as main_loop_module
import core.task as task_module
from core.env_state import EnvironmentState
from diagnostics.value_gae_identifiability import (
    ROOT, OUT as VALUE_OUT, CHECKPOINT, EXCEL, FrozenAuditAgent, make_agent,
    run_episode, model_hash, smdp_mc, production_gae, train_normalization,
    normalize_target, inverse_target, value_metrics, params, HORIZONS,
    episode_seed_plan, split_episodes, matched_indices, sha256_file,
)
from agents.ppo_agent import PPOValueNetwork

OUT = ROOT / 'diagnostics/results/state_sufficiency_audit'
VALUE_MANIFEST = VALUE_OUT / 'frozen_policy_dataset_manifest.json'
VARIANT_SEED = 2026092701


class InFlightAuditEnvironmentState(EnvironmentState):
    """Tracks upload lifecycle without changing SimPy scheduling or queues."""
    def __init__(self):
        super().__init__()
        self.audit_inflight = {}
        self.audit_upload_history = []

    def register_waiting_replica(self, server_id, task, selection, service_time):
        key = (int(task.id), str(selection))
        entry = self.audit_inflight.pop(key, None)
        if entry is None:
            raise RuntimeError(f'Missing diagnostic upload start for {key}')
        now = float(task.env.now)
        expected = float(entry['upload_start_time'] + entry['expected_upload_duration'])
        if not np.isclose(now, expected, rtol=0.0, atol=1e-8):
            raise RuntimeError(f'Upload lifecycle timing mismatch for {key}: {now} vs {expected}')
        if int(server_id) != int(entry['destination_server_id']):
            raise RuntimeError(f'Upload destination mismatch for {key}')
        self.audit_upload_history.append({**entry, 'cpu_queue_entry_time': now})
        return super().register_waiting_replica(server_id, task, selection, service_time)


def instrumented_upload_duration(task, server_object, original):
    duration, output_duration = original(task, server_object)
    if not isinstance(task.env_state, InFlightAuditEnvironmentState):
        raise RuntimeError('Expected diagnostic-only environment state')
    if task.primaryNode is not None and int(server_object.server_id) == int(task.primaryNode.server_id):
        label = 'primary'
    elif task.backupNode is not None and int(server_object.server_id) == int(task.backupNode.server_id):
        label = 'backup'
    else:
        raise RuntimeError('Cannot identify replica destination')
    key = (int(task.id), label)
    if key in task.env_state.audit_inflight:
        raise RuntimeError(f'Duplicate diagnostic upload start for {key}')
    task.env_state.audit_inflight[key] = {
        'task_id': int(task.id), 'replica': label,
        'destination_server_id': int(server_object.server_id),
        'upload_start_time': float(task.env.now),
        'expected_upload_duration': float(duration),
        'input_data_size_mb': float(task.input_data_size_mb),
        'computation_demand': float(task.computation_demand),
        'cpu_service_time': float(task.computation_demand / server_object.processing_frequency),
    }
    return duration, output_duration


def _remaining_upload(entry, now):
    return max(float(entry['upload_start_time'] + entry['expected_upload_duration'] - now), 0.0)


def snapshot_decision_state(task, env, decision):
    """Return only currently observable simulator metadata at this decision."""
    server_ids = sorted(env.servers)
    n = len(server_ids)
    flight_count = np.zeros(n, dtype=np.float64)
    flight_work = np.zeros(n, dtype=np.float64)
    upload_min = np.zeros(n, dtype=np.float64)
    upload_mean = np.zeros(n, dtype=np.float64)
    upload_max = np.zeros(n, dtype=np.float64)
    flight_records = [[] for _ in server_ids]
    by_server = [[] for _ in server_ids]
    now = float(task.env.now)
    for entry in env.audit_inflight.values():
        j = server_ids.index(int(entry['destination_server_id']))
        rem = _remaining_upload(entry, now)
        item = {**entry, 'remaining_upload_time': rem}
        flight_records[j].append(item)
        by_server[j].append(item)
    for j, records in enumerate(by_server):
        if records:
            remaining = np.asarray([r['remaining_upload_time'] for r in records], dtype=float)
            flight_count[j] = len(records)
            flight_work[j] = sum(float(r['cpu_service_time']) for r in records)
            upload_min[j], upload_mean[j], upload_max[j] = remaining.min(), remaining.mean(), remaining.max()

    running_remaining = np.zeros(n, dtype=np.float64)
    waiting_work = np.zeros(n, dtype=np.float64)
    waiting_count = np.zeros(n, dtype=np.float64)
    running_count = np.zeros(n, dtype=np.float64)
    pending_replicas = np.zeros(n, dtype=np.float64)
    backlog = np.zeros(n, dtype=np.float64)
    for j, sid in enumerate(server_ids):
        info = env.servers[sid]
        running = info['running_replica']
        if running is not None:
            elapsed = max(now - float(running['service_start_time']), 0.0)
            running_remaining[j] = max(float(running['service_time']) - elapsed, 0.0)
            running_count[j] = 1.0
        waiting = info['waiting_replicas']
        waiting_count[j] = len(waiting)
        waiting_work[j] = sum(max(float(w['service_time']), 0.0) for w in waiting)
        backlog[j] = float(env.get_server_backlog_time(sid, now))
        pending_replicas[j] = flight_count[j] + waiting_count[j] + running_count[j]

    reliabilities = np.asarray(decision['reliabilities'], dtype=np.float64).copy()
    safe = np.asarray(decision['safe_mask'], dtype=bool).copy()
    effective = np.asarray(decision['effective_mask'], dtype=bool).copy()
    if not (reliabilities.shape == safe.shape == effective.shape == (28,)):
        raise RuntimeError('Unexpected pair reliability/mask vector size')
    safe_values = reliabilities[safe]
    best = float(decision['best_achievable_reliability'])
    rel_summary = np.asarray([
        float(safe.sum()), float(bool(decision['safe_set_empty'])), best,
        max(float(decision['requirement']) - best, 0.0),
        float(safe_values.mean()) if len(safe_values) else 0.0,
        float(safe_values.min()) if len(safe_values) else 0.0,
        float(safe_values.max()) if len(safe_values) else 0.0,
    ], dtype=np.float64)
    flight_features = np.concatenate([flight_count, flight_work, upload_mean, upload_min, upload_max])
    s2_cpu = np.concatenate([running_remaining, waiting_work, waiting_count, running_count,
                             pending_replicas, np.asarray([len(env.tasks)], dtype=float)])
    effective_rates = np.asarray([env.get_active_failure_rate(sid) for sid in server_ids], dtype=float)
    risk_field = np.asarray(env.spatial_risk_field if env.spatial_risk_field is not None else np.zeros(n), dtype=float)
    return {
        'time': now, 'task_id': int(task.id), 'pending_task_count': int(len(env.tasks)),
        'active_uploads': int(len(env.audit_inflight)),
        'inflight_count': flight_count, 'inflight_workload': flight_work,
        'upload_remaining_min': upload_min, 'upload_remaining_mean': upload_mean,
        'upload_remaining_max': upload_max,
        'flight_records_json': json.dumps(flight_records, sort_keys=True, separators=(',', ':')),
        'running_remaining': running_remaining, 'waiting_service': waiting_work,
        'waiting_count': waiting_count, 'running_count': running_count,
        'pending_replica_count': pending_replicas, 'cpu_backlog_seconds': backlog,
        'effective_failure_rates': effective_rates, 'spatial_risk_field': risk_field,
        'pair_reliability': reliabilities, 'safe_mask': safe.astype(np.float64),
        'effective_mask': effective.astype(np.float64), 'reliability_summary': rel_summary,
        'safe_set_empty': bool(decision['safe_set_empty']),
        'flight_features': flight_features, 'cpu_population_features': s2_cpu,
        'reliability_features': np.concatenate([reliabilities, safe.astype(float), effective.astype(float), rel_summary]),
        'state1': np.concatenate([np.asarray(decision['state'], dtype=float), reliabilities,
                                  effective.astype(float), safe.astype(float), rel_summary]),
        'state2': np.concatenate([np.asarray(decision['state'], dtype=float), flight_features]),
        'state3': np.concatenate([np.asarray(decision['state'], dtype=float), reliabilities,
                                  effective.astype(float), safe.astype(float), rel_summary, flight_features]),
        'state4': np.concatenate([np.asarray(decision['state'], dtype=float), reliabilities,
                                  effective.astype(float), safe.astype(float), rel_summary, flight_features,
                                  running_remaining, waiting_work, waiting_count, running_count,
                                  pending_replicas, effective_rates, risk_field]),
    }


def set_up_agent():
    agent, rho = make_agent()
    agent.__class__ = StateSufficiencyAgent
    return agent, rho


class StateSufficiencyAgent(FrozenAuditAgent):
    def select_action(self, state, *args, **kwargs):
        task, env = self.audit_context
        latent = snapshot_decision_state(task, env, self.current_decision)
        action = super().select_action(state, *args, **kwargs)
        if not self.details:
            raise RuntimeError('Full telemetry collection must record each decision')
        self.records[-1].update(latent)
        return action


def collect_replay_episode(agent, seedrow, max_tasks=200):
    """Replay exact prior frozen-policy seeds while instrumenting only callbacks."""
    original_upload = task_module.Task.calc_input_output_delay
    def upload(task, server_object):
        return instrumented_upload_duration(task, server_object, original_upload)
    agent.reset_audit(details=True)
    main_loop_module.EnvironmentState = InFlightAuditEnvironmentState
    try:
        with patch.object(task_module.Task, 'calc_input_output_delay', upload):
            data, loop = run_episode(agent, seedrow, details=True, max_tasks=max_tasks)
    finally:
        main_loop_module.EnvironmentState = EnvironmentState
    if not isinstance(loop.env_state, InFlightAuditEnvironmentState):
        raise RuntimeError('Read-only telemetry did not run')
    if loop.env_state.audit_inflight:
        raise RuntimeError('Unfinished uploads remain after natural episode drain')
    if len(loop.env_state.audit_upload_history) != 2 * max_tasks:
        raise RuntimeError('Expected two upload lifecycle records per task')
    # MainLoop's existing TaskAssignments record is the authoritative realized
    # task delay. Join by task id because completion order is asynchronous.
    delays = {int(row[1]): float(row[22]) for row in loop.task_Assignments_info}
    task_ids = np.asarray(data['task_ids'], dtype=int)
    if len(delays) != max_tasks or set(delays) != set(task_ids.tolist()):
        raise RuntimeError('Resolved task-delay records do not match frozen decisions')
    data['task_delay'] = np.asarray([delays[int(tid)] for tid in task_ids], dtype=np.float64)
    return data, loop


def make_variants(data):
    s0 = np.asarray(data['states'], dtype=np.float32)
    variants = {
        'S0': s0,
        'S1': np.concatenate([s0, data['reliability_features']], axis=1).astype(np.float32),
        'S2': np.concatenate([s0, data['flight_features']], axis=1).astype(np.float32),
        'S3': np.concatenate([s0, data['reliability_features'], data['flight_features']], axis=1).astype(np.float32),
        'S4': np.concatenate([s0, data['reliability_features'], data['flight_features'],
                              data['cpu_population_features'], data['effective_failure_rates'],
                              data['spatial_risk_field']], axis=1).astype(np.float32),
    }
    return variants


def nearest_indices(features, train_mask, test_mask, k=20):
    train = np.asarray(features[train_mask], dtype=np.float64)
    test = np.asarray(features[test_mask], dtype=np.float64)
    mean = train.mean(axis=0); scale = train.std(axis=0)
    active = scale > 1e-8
    if not active.any():
        raise ValueError('No varying state features')
    train_z = (train[:, active] - mean[active]) / scale[active]
    test_z = (test[:, active] - mean[active]) / scale[active]
    tree = cKDTree(train_z)
    distance, idx = tree.query(test_z, k=k, workers=1)
    return distance, idx


def build_value_model(input_dim, seed=VARIANT_SEED):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(seed))
        return PPOValueNetwork(input_dim, params.hidden_layers_ppo, activation=params.af_ppo)


def target_from_saved_value_gae(value, next_value, data, episode):
    ids = np.flatnonzero(np.asarray(data['episode']) == int(episode))
    n = len(ids)
    nexts = np.asarray(next_value[ids], dtype=np.float32).copy()
    nexts[:-1] = np.asarray(value[ids[1:]], dtype=np.float32)
    rewards = np.asarray(data['rewards'][ids], dtype=float) * params.reward_scale_ppo
    return production_gae(rewards, data['delta_t'][ids], data['done'][ids],
                          value[ids], nexts, params.gamma_ppo, params.gae_lambda_ppo)[3]


def compatibility_pair(primary, backup):
    return tuple(sorted((int(primary), int(backup))))


def seed_dict_from_plan(frame, episode):
    return frame.set_index('episode').loc[int(episode)].to_dict()
