"""Read-only feasibility scan before training; never changes task workbooks."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import simpy

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config.params import params
from config.paths import SERVER_INFO_PATH, TRAIN_TASK_PATH, EVALUATION_TASK_PATH
from core.env_state import EnvironmentState
from core.main_loop import MainLoop
from core.server import Server
from core.task import Task


def scan(scales, requirements=None):
    profiles = MainLoop._build_server_profiles(pd.read_excel(SERVER_INFO_PATH, sheet_name='Servers'))
    rows = []
    for split, path in (('train', TRAIN_TASK_PATH), ('evaluation', EVALUATION_TASK_PATH)):
        tasks = MainLoop._build_task_profiles(pd.read_excel(path))
        for scale in scales:
            env = simpy.Environment()
            state = EnvironmentState(failure_rate_scale=scale)
            for sid, profile in profiles.items():
                state.add_server_and_init_environment(Server(
                    env, sid, profile['Site_ID'], profile['Processing_Frequency'],
                    profile['Base_Failure_Rate'], profile['Transmission_Rate'],
                    profile['Latitude'], profile['Longitude']))
            reliability = np.stack([state.compute_task_reliabilities(Task(env, state, tid, profile))[1]
                                    for tid, profile in tasks.items()])
            thresholds = [None] if requirements is None else requirements
            for threshold in thresholds:
                required = (np.array([p['Reliability_Requirement'] for p in tasks.values()])
                            if threshold is None else np.full(len(tasks), threshold))
                counts = (reliability >= required[:, None]).sum(axis=1)
                rows.append(dict(split=split, failure_rate_scale=scale,
                    requirement='workbook' if threshold is None else threshold,
                    task_count=len(tasks), no_safe_action_count=int((counts == 0).sum()),
                    single_safe_action_count=int((counts == 1).sum()),
                    multi_safe_action_count=int((counts > 1).sum()),
                    mean_safe_pair_count=float(counts.mean()),
                    max_pair_reliability=float(reliability.max())))
    return dict(objective='mean effective QoS over all tasks', gamma=params.gamma_ppo,
                reward_assignment='completion interval', scenarios=rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--failure-rate-scales', type=float, nargs='+', default=[1., .5, .1, .05])
    parser.add_argument('--requirements', type=float, nargs='+',
                        help='In-memory threshold scan; default uses the actual workbook requirements')
    args = parser.parse_args()
    if args.requirements is not None and any(not np.isfinite(r) or not 0 < r <= 1 for r in args.requirements):
        parser.error('Requirements must be finite and in (0, 1]')
    print(json.dumps(scan(args.failure_rate_scales, args.requirements), indent=2))


if __name__ == '__main__':
    main()
