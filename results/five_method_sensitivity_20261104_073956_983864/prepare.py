"""Generate one new scenario and nested evaluation subsets; preserve old inputs."""
import hashlib
import json
from pathlib import Path
import shutil
import sys
from unittest.mock import patch

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
from config.configuration import parameters
from config.paths import SERVER_INFO_PATH, TRAIN_TASK_PATH, EVALUATION_TASK_PATH
from tools.generate_server_and_task_parameters import generate_server_info, generate_task_params


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    inputs = OUT / "inputs"
    assert not (OUT / "manifest.json").exists(), "Completed inputs must not be overwritten"
    inputs.mkdir(exist_ok=True)
    assert parameters.PROFILE_SEED == 2029
    assert parameters.TASK_RELIABILITY_REQUIREMENT_RANGE == (.92, .999)
    assert parameters.taskno == 200 and parameters.NUM_SERVERS == 8
    servers = generate_server_info(inputs / "server_info.xlsx")
    train = generate_task_params(inputs / "train200.xlsx", dataset="train")
    evaluation = generate_task_params(inputs / "evaluation200.xlsx", dataset="evaluation")
    with patch.object(parameters, "taskno", 600):
        pool = generate_task_params(inputs / "evaluation_pool600.xlsx", dataset="evaluation")
    # Excel stores decimal floats; define all scenarios from the same saved values.
    train = pd.read_excel(inputs / "train200.xlsx")
    evaluation = pd.read_excel(inputs / "evaluation200.xlsx")
    pool = pd.read_excel(inputs / "evaluation_pool600.xlsx")
    pd.testing.assert_frame_equal(pool.head(200), evaluation, check_exact=True)
    assert not train.equals(evaluation)
    for frame in (train, pool):
        assert frame.Reliability_Requirement.between(.92, .999).all()
        assert frame.Task_ID.tolist() == list(range(1, len(frame) + 1))
    for source, destination in ((inputs / "server_info.xlsx", SERVER_INFO_PATH),
                                (inputs / "train200.xlsx", TRAIN_TASK_PATH),
                                (inputs / "evaluation200.xlsx", EVALUATION_TASK_PATH)):
        shutil.copy2(source, destination)
        assert digest(source) == digest(destination)
    scenarios = {"baseline": dict(id="baseline", arrival_rate=2., demand_multiplier=1., task_count=200,
                                   task_path=str(inputs / "evaluation200.xlsx"))}
    groups = {}
    for group, levels in (("arrival_rate", [.5, 1., 1.5, 2., 2.5, 3.]),
                          ("demand_multiplier", [.5, 1., 1.5, 2., 2.5, 3.]),
                          ("task_count", [200, 300, 400, 500, 600])):
        members = []
        for level in levels:
            base = {"arrival_rate": 2., "demand_multiplier": 1., "task_count": 200}
            if level == base[group]:
                scenario_id = "baseline"
            else:
                scenario_id = f"{group}_{level:g}"
                scenario = dict(base, id=scenario_id)
                scenario[group] = level
                frame = pool.head(int(scenario["task_count"])).copy()
                frame["Computation_Demand"] *= scenario["demand_multiplier"]
                task_path = inputs / f"{scenario_id}.xlsx"
                frame.to_excel(task_path, index=False)
                reread = pd.read_excel(task_path)
                expected = pool.head(int(scenario["task_count"]))
                pd.testing.assert_frame_equal(reread.drop(columns="Computation_Demand"),
                                              expected.drop(columns="Computation_Demand"), check_exact=True)
                np.testing.assert_allclose(reread.Computation_Demand,
                                           expected.Computation_Demand * scenario["demand_multiplier"], rtol=1e-15)
                scenario["task_path"] = str(task_path)
                scenarios[scenario_id] = scenario
            members.append(dict(value=level, scenario_id=scenario_id))
        groups[group] = members
    # One unit-rate exponential stream per episode; prefixes align across task counts.
    arrivals = np.random.default_rng(2027).exponential(size=(500, 600))
    np.save(inputs / "evaluation_unit_interarrivals.npy", arrivals)
    manifest = dict(profile_seed=2029, reliability_requirement_range=[.92, .999],
                    server_count=8, topology="Existing eight EUA sites, new hardware and failure-rate samples",
                    training_episodes=8000, training_tasks_per_episode=200, training_arrival_rate=2., training_seed=2026,
                    evaluation_episodes=500, evaluation_seed=2027,
                    arrival_protocol="PCG64(seed=2027) exponential(1), shape=(500,600); prefix N per episode divided by arrival rate",
                    frozen_observation_normalization="Retain baseline size, demand and server feature normalization at every sweep point",
                    groups=groups, scenarios=list(scenarios.values()),
                    input_sha256={str(path.relative_to(OUT)): digest(path) for path in inputs.iterdir()},
                    all_task_prefixes_verified=True, unchanged_input_columns_verified=True)
    with (OUT / "manifest.json").open("x", encoding="utf-8") as output:
        json.dump(manifest, output, indent=2)
    print("New data generated. Unique scenarios:", len(scenarios), flush=True)
    print("Baseline requirement range:", evaluation.Reliability_Requirement.min(), evaluation.Reliability_Requirement.max())
    print(servers.to_string(index=False))


if __name__ == "__main__":
    main()
