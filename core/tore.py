"""TORE: score-ordered incremental replication with at most two replicas.

This adaptation scores nodes by predicted QoS without availability estimation.
Add the highest-scoring unselected
node, check reliability, and stop on success or at the replica cap.
"""

from dataclasses import dataclass

import numpy as np

from config.params import params
from core.env_state import predict_server_completion_times, select_latency_greedy_action
from core.task import compute_task_qos


@dataclass(frozen=True)
class TOREDecision:
    server_ids: tuple
    action_index: int | None  # Existing pair index; None for a single replica.
    is_fallback: bool
    independent_reliability: float
    predicted_qos: float
    predicted_latency_s: float
    greedy_server_ids: tuple  # Addition order, before any fallback.
    greedy_reliability: float


def select_tore_action(state, pairs, marginal_reliabilities, independent_pair_reliabilities,
                       reliability_requirement, deadline):
    """Add nodes in descending QoS order; check after each addition.

    Node-score ties use predicted completion then server ID. A lower-ranked
    feasible singleton is not tried in isolation, and alternative pairs are
    not searched. The two-replica cap is an adaptation of TORE-NET.
    Pair reliabilities must use the same marginals as the correlated evaluator.
    No safe mask is consumed, so a cached correlated mask cannot affect TORE.
    """
    server_ids = sorted({sid for pair in pairs for sid in pair})
    predicted = predict_server_completion_times(state)
    marginals = np.asarray(marginal_reliabilities, dtype=float)
    independent = np.asarray(independent_pair_reliabilities, dtype=float)
    if (len(server_ids) != len(predicted) or marginals.shape != (len(server_ids),)
            or independent.shape != (len(pairs),)):
        raise ValueError("TORE requires matching server, pair and reliability orders")
    if (not np.isfinite(reliability_requirement) or not 0 <= reliability_requirement <= 1
            or not all(np.isfinite(p).all() and np.all((p >= 0) & (p <= 1))
                       for p in (marginals, independent))):
        raise ValueError("TORE reliabilities and requirement must be probabilities")
    qos = compute_task_qos(predicted, deadline, grace_multiplier=params.QOS_GRACE_MULTIPLIER)
    index = {sid: i for i, sid in enumerate(server_ids)}
    ranked = sorted(server_ids, key=lambda sid: (-float(qos[index[sid]]),
                                                float(predicted[index[sid]]), sid))
    selected = []
    for sid in ranked[:2]:
        selected.append(sid)
        canonical = tuple(sorted(selected))
        action = pairs.index(canonical) if len(selected) == 2 else None
        reliability = (float(independent[action]) if action is not None
                       else float(marginals[index[sid]]))
        latency = float(min(predicted[index[s]] for s in selected))
        task_qos = float(max(qos[index[s]] for s in selected))
        if reliability >= reliability_requirement:
            return TOREDecision(canonical, action, False, reliability, task_qos, latency,
                                tuple(selected), reliability)

    # Same unmasked SML fallback as the existing comparison, including its ties.
    action = select_latency_greedy_action(state, pairs)
    pair = tuple(pairs[action])
    a, b = (index[sid] for sid in pair)
    return TOREDecision(pair, action, True, float(independent[action]),
                        float(max(qos[a], qos[b])), float(min(predicted[a], predicted[b])),
                        tuple(selected), reliability)
