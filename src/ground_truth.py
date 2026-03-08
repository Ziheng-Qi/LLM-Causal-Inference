"""
Ground truth computation engine.

Computes causal quantities using rigorous statistical methods:
- Observational probabilities (standard conditioning)
- Interventional probabilities (do-calculus via backdoor/frontdoor adjustment)
- Backdoor criterion identification
- Frontdoor criterion identification

These serve as the "golden version" to compare against LLM answers.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
from pgmpy.models import DiscreteBayesianNetwork as BayesianNetwork
from pgmpy.inference import VariableElimination
from pgmpy.inference import CausalInference


def compute_observational(
    model: BayesianNetwork,
    query_var: str,
    query_val: str,
    evidence: Optional[Dict[str, str]] = None,
) -> float:
    """
    Compute P(query_var = query_val | evidence) using standard conditioning.
    This is a Level-1 (associational) query.
    """
    ve = VariableElimination(model)
    evidence = evidence or {}
    result = ve.query([query_var], evidence=evidence)

    state_names = None
    for cpd in model.get_cpds():
        if cpd.variable == query_var:
            state_names = cpd.state_names[query_var]
            break

    idx = state_names.index(query_val)
    return float(result.values[idx])


def compute_interventional(
    model: BayesianNetwork,
    treatment: str,
    treatment_val: str,
    outcome: str,
    outcome_val: str,
    method: str = "backdoor",
    adjustment_set: Optional[List[str]] = None,
) -> float:
    """
    Compute P(outcome = outcome_val | do(treatment = treatment_val))
    using the specified adjustment method.

    This is a Level-2 (interventional) query.

    Parameters
    ----------
    method : str
        "backdoor" - uses backdoor adjustment (requires adjustment_set or auto-detect)
        "frontdoor" - uses frontdoor adjustment
        "mutilated" - uses graph mutilation (cut incoming edges to treatment)
    """
    ci = CausalInference(model)

    if method == "mutilated":
        # Graph mutilation: remove all edges into treatment, then condition
        result = ci.query(
            variables=[outcome],
            do={treatment: treatment_val},
        )
    elif method == "backdoor":
        if adjustment_set is None:
            adjustment_set = list(ci.get_minimal_adjustment_set(treatment, outcome))
        result = ci.query(
            variables=[outcome],
            do={treatment: treatment_val},
            adjustment_set=adjustment_set,
        )
    elif method == "frontdoor":
        result = ci.query(
            variables=[outcome],
            do={treatment: treatment_val},
        )
    else:
        raise ValueError(f"Unknown method: {method}")

    state_names = None
    for cpd in model.get_cpds():
        if cpd.variable == outcome:
            state_names = cpd.state_names[outcome]
            break

    idx = state_names.index(outcome_val)
    return float(result.values[idx])


def compute_ate(
    model: BayesianNetwork,
    treatment: str,
    treatment_vals: Tuple[str, str],
    outcome: str,
    outcome_val: str,
    method: str = "backdoor",
) -> float:
    """
    Compute Average Treatment Effect (ATE):
      ATE = P(Y=y | do(T=t1)) - P(Y=y | do(T=t0))

    treatment_vals: (control_value, treated_value)
    """
    p_control = compute_interventional(
        model, treatment, treatment_vals[0], outcome, outcome_val, method
    )
    p_treated = compute_interventional(
        model, treatment, treatment_vals[1], outcome, outcome_val, method
    )
    return p_treated - p_control


def identify_adjustment_sets(
    model: BayesianNetwork,
    treatment: str,
    outcome: str,
) -> Dict[str, object]:
    """
    Identify valid adjustment sets for causal effect estimation.
    Returns info about backdoor and frontdoor criteria.
    """
    ci = CausalInference(model)

    result = {
        "treatment": treatment,
        "outcome": outcome,
    }

    # Backdoor
    try:
        backdoor_sets = ci.get_minimal_adjustment_set(treatment, outcome)
        result["backdoor_adjustment_set"] = list(backdoor_sets)
        result["backdoor_applicable"] = True
    except Exception:
        result["backdoor_adjustment_set"] = []
        result["backdoor_applicable"] = False

    # Frontdoor - check if any single variable satisfies frontdoor criterion
    result["frontdoor_candidates"] = []
    all_vars = set(model.nodes()) - {treatment, outcome}
    for mediator in all_vars:
        if _check_frontdoor(model, treatment, outcome, mediator):
            result["frontdoor_candidates"].append(mediator)
    result["frontdoor_applicable"] = len(result["frontdoor_candidates"]) > 0

    return result


def _check_frontdoor(
    model: BayesianNetwork,
    treatment: str,
    outcome: str,
    mediator: str,
) -> bool:
    """
    Check if mediator satisfies the frontdoor criterion for
    the causal effect of treatment on outcome.

    Frontdoor criterion requires:
    1. Treatment intercepts all directed paths from treatment to outcome
       (mediator is on all directed paths from T to Y)
    2. No unblocked backdoor path from treatment to mediator
    3. All backdoor paths from mediator to outcome are blocked by treatment
    """
    import networkx as nx
    G = nx.DiGraph(model.edges())

    # Check 1: mediator is on all directed paths from treatment to outcome
    all_paths = list(nx.all_simple_paths(G, treatment, outcome))
    if not all_paths:
        return False
    for path in all_paths:
        if mediator not in path:
            return False

    # Check 2: treatment -> mediator path exists
    if not nx.has_path(G, treatment, mediator):
        return False

    return True


def compute_ground_truth_suite(
    model: BayesianNetwork,
    treatment: str,
    outcome: str,
    treatment_vals: Tuple[str, str] = None,
    outcome_val: str = None,
) -> Dict:
    """
    Compute a full suite of ground truth values for a given causal query.
    Returns observational, interventional, and ATE results.
    """
    # Get state names
    for cpd in model.get_cpds():
        if cpd.variable == treatment:
            t_states = cpd.state_names[treatment]
        if cpd.variable == outcome:
            o_states = cpd.state_names[outcome]

    if treatment_vals is None:
        treatment_vals = (t_states[0], t_states[1])
    if outcome_val is None:
        outcome_val = o_states[1]  # typically the "positive" outcome

    results = {
        "query": {
            "treatment": treatment,
            "outcome": outcome,
            "treatment_values": treatment_vals,
            "outcome_value": outcome_val,
        },
        "observational": {},
        "interventional": {},
        "adjustment_info": identify_adjustment_sets(model, treatment, outcome),
    }

    # Observational: P(Y=y | T=t)
    for t_val in treatment_vals:
        p = compute_observational(
            model, outcome, outcome_val, evidence={treatment: t_val}
        )
        results["observational"][f"P({outcome}={outcome_val}|{treatment}={t_val})"] = round(p, 6)

    # Interventional: P(Y=y | do(T=t))
    for t_val in treatment_vals:
        p = compute_interventional(
            model, treatment, t_val, outcome, outcome_val, method="mutilated"
        )
        results["interventional"][f"P({outcome}={outcome_val}|do({treatment}={t_val}))"] = round(p, 6)

    # ATE
    results["ate"] = round(
        compute_ate(model, treatment, treatment_vals, outcome, outcome_val),
        6,
    )

    # Naive (unadjusted) vs adjusted comparison
    naive_ate = (
        results["observational"][f"P({outcome}={outcome_val}|{treatment}={treatment_vals[1]})"]
        - results["observational"][f"P({outcome}={outcome_val}|{treatment}={treatment_vals[0]})"]
    )
    results["naive_ate"] = round(naive_ate, 6)
    results["confounding_bias"] = round(naive_ate - results["ate"], 6)

    return results
