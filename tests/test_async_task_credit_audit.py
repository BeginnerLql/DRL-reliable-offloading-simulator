import numpy as np

from diagnostics.async_task_credit_audit import (
    classify_provenance,
    common_randomness_mismatches,
    compare_original_branch,
    centered_scores,
    decision_event_alignment,
    full_terminal_components,
    optimal_set,
    select_relevant_provenance,
)


def _record(index, decision_time, completion_time, reward):
    return {
        "decision_index": index,
        "task_id": index,
        "decision_time": decision_time,
        "completion_time": completion_time,
        "reward": reward,
    }


def test_reward_provenance_classification_uses_decision_order_and_resolution():
    assert classify_provenance(5, 5, 8.0, 7.0) == "current_root"
    assert classify_provenance(5, 6, 9.0, 7.0) == "future_decision"
    assert classify_provenance(5, 4, 8.0, 7.0) == "preexisting_pending"
    assert classify_provenance(5, 4, 6.5, 7.0) == "preexisting_resolved"


def test_preexisting_pending_rewards_are_excluded_from_future_component():
    records = [_record(4, 6.0, 8.0, 100.0), _record(5, 7.0, 8.0, 4.0),
               _record(6, 8.0, 10.0, 2.0)]
    selected = select_relevant_provenance(records, 5, 7.0)
    assert [row["task_role"] for row in selected] == ["preexisting_pending", "current_root", "future_decision"]
    components = full_terminal_components(selected, 5, 7.0, gamma=0.5)
    assert components["future_decision"] == 0.5 * 2.0
    assert components["future_event"] == 0.5**3 * 2.0
    assert np.isclose(components["total_event"], 2.25)
    assert np.isclose(components["total_decision"], 5.0)


def test_own_future_event_decomposition_identity():
    records = [_record(3, 2.0, 3.0, 4.0), _record(4, 2.5, 4.5, -2.0),
               _record(5, 3.0, 5.0, 1.0)]
    result = full_terminal_components(records, 3, 2.0, gamma=0.9)
    assert abs(result["total_event"] - result["own_event"] - result["future_event"]) < 1e-12
    assert abs(result["total_decision"] - result["own_decision"] - result["future_decision"]) < 1e-12


def test_decision_time_discount_uses_task_decision_time_lag():
    result = full_terminal_components(
        [_record(2, 4.0, 5.0, 4.0), _record(3, 7.0, 9.0, 2.0)],
        2, 4.0, gamma=0.5,
    )
    assert result["own_decision"] == 4.0
    assert result["future_decision"] == 0.5**3 * 2.0
    assert result["total_decision"] == 4.25


def test_event_time_discount_uses_reward_resolution_lag_including_root():
    result = full_terminal_components(
        [_record(2, 4.0, 5.0, 4.0), _record(3, 7.0, 8.0, 2.0)],
        2, 4.0, gamma=0.5,
    )
    assert result["own_event"] == 0.5 * 4.0
    assert result["future_event"] == 0.5**4 * 2.0
    assert result["total_event"] == 2.125


def test_policy_centering_has_zero_weighted_residual():
    centered, residual = centered_scores([1.0, 3.0, -2.0], [0.2, 0.5, 0.3])
    assert np.allclose(centered, np.asarray([1.0, 3.0, -2.0]) - np.dot([0.2, 0.5, 0.3], [1.0, 3.0, -2.0]))
    assert residual < 1e-14


def test_original_branch_replay_compares_state_action_reward_delay_arrays():
    reference = {
        "states": np.asarray([[1.0, 2.0]]), "next_states": np.asarray([[2.0, 3.0]]),
        "actions": np.asarray([4]), "rewards": np.asarray([2.5]),
        "delta_t": np.asarray([0.2]), "done": np.asarray([True]),
    }
    mismatches, maximum = compare_original_branch(reference, reference)
    assert not any(mismatches.values())
    assert maximum == 0.0
    changed = {key: np.array(value, copy=True) for key, value in reference.items()}
    changed["rewards"][0] += 1e-4
    mismatches, maximum = compare_original_branch(reference, changed)
    assert mismatches["rewards"] == 1
    assert np.isclose(maximum, 1e-4, rtol=0.0, atol=1e-15)


def test_common_randomness_requires_same_arrival_and_torch_streams():
    reference = {"rng_trace": np.asarray(["a", "b"])}
    exact = {"rng_trace": np.asarray(["a", "b"]), "arrival_trace": np.asarray([.1, .2])}
    assert common_randomness_mismatches(reference, exact, np.asarray([.1, .2])) == {
        "arrival_stream_mismatch": 0, "torch_stream_mismatch": 0,
    }
    changed = {"rng_trace": np.asarray(["a", "c"]), "arrival_trace": np.asarray([.1, .3])}
    assert common_randomness_mismatches(reference, changed, np.asarray([.1, .2])) == {
        "arrival_stream_mismatch": 1, "torch_stream_mismatch": 1,
    }


def test_tie_aware_optimal_sets_support_conflict_and_overlap_checks():
    own = np.asarray([3.0, 3.0, 2.0])
    total = np.asarray([1.0, 2.0, 4.0])
    own_set, total_set = set(optimal_set(own)), set(optimal_set(total))
    assert own_set == {0, 1}
    assert total_set == {2}
    assert own_set.isdisjoint(total_set)
    assert decision_event_alignment(own, total)["optimal_set_jaccard"] == 0.0


def test_full_terminal_future_return_uses_every_following_decision():
    records = [_record(7, 7.0, 8.0, 1.0)]
    records.extend(_record(index, float(index), float(index + 1), 1.0) for index in range(8, 15))
    result = full_terminal_components(records, 7, 7.0, gamma=0.5)
    expected = sum(0.5 ** (index - 7) for index in range(8, 15))
    assert np.isclose(result["future_decision"], expected)
    assert np.isclose(result["total_decision"], 1.0 + expected)
    assert result["future_decision"] > sum(0.5 ** n for n in range(1, 4))
