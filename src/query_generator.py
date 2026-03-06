"""
Query generator: converts causal graphs + ground truth into natural language
questions for LLM evaluation.

Generates three levels of causal queries (Pearl's Ladder):
  Level 1 (Association): P(Y | X)
  Level 2 (Intervention): P(Y | do(X))
  Level 3 (Counterfactual): P(Y_x | X', Y')

Also generates structural questions:
  - "Is X a cause of Y?"
  - "What variables should you adjust for?"
  - "Does the backdoor criterion apply?"
"""

from typing import Dict, List, Optional
from pgmpy.models import BayesianNetwork


def _dag_to_text(model: BayesianNetwork) -> str:
    """Convert a DAG to a human-readable text description."""
    edges = list(model.edges())
    lines = ["The causal graph has the following edges (directed):"]
    for src, dst in edges:
        lines.append(f"  {src} -> {dst}")
    lines.append(f"Variables: {', '.join(sorted(model.nodes()))}")
    return "\n".join(lines)


def _cpd_to_text(model: BayesianNetwork) -> str:
    """Convert CPDs to human-readable text."""
    lines = ["Conditional Probability Distributions:"]
    for cpd in model.get_cpds():
        lines.append(f"\n{cpd}")
    return "\n".join(lines)


def generate_query(
    model: BayesianNetwork,
    query_type: str,
    treatment: str,
    outcome: str,
    treatment_val: str,
    outcome_val: str,
    include_cpds: bool = True,
    include_dag: bool = True,
    extra_context: str = "",
) -> str:
    """
    Generate a natural language causal reasoning question.

    Parameters
    ----------
    query_type : str
        One of: "associational", "interventional", "backdoor_identify",
        "frontdoor_identify", "ate", "structural"
    include_cpds : bool
        Whether to include the full CPD tables in the prompt
    include_dag : bool
        Whether to include the DAG structure in the prompt
    """
    # Build context
    context_parts = []
    if extra_context:
        context_parts.append(extra_context)
    if include_dag:
        context_parts.append(_dag_to_text(model))
    if include_cpds:
        context_parts.append(_cpd_to_text(model))
    context = "\n\n".join(context_parts)

    # Build question based on type
    templates = {
        "associational": (
            f"Given the causal model below, compute the conditional probability "
            f"P({outcome} = {outcome_val} | {treatment} = {treatment_val}).\n\n"
            f"Show your step-by-step calculation.\n\n"
            f"{context}"
        ),
        "interventional": (
            f"Given the causal model below, compute the interventional probability "
            f"P({outcome} = {outcome_val} | do({treatment} = {treatment_val})).\n\n"
            f"Explain which adjustment method (backdoor, frontdoor, or graph mutilation) "
            f"you are using and why. Show your step-by-step calculation.\n\n"
            f"{context}"
        ),
        "backdoor_identify": (
            f"Given the causal model below, determine:\n"
            f"1. Does the backdoor criterion apply for estimating the causal effect "
            f"of {treatment} on {outcome}?\n"
            f"2. If yes, what is the minimal sufficient adjustment set?\n"
            f"3. Write out the backdoor adjustment formula.\n\n"
            f"Explain your reasoning step by step.\n\n"
            f"{context}"
        ),
        "frontdoor_identify": (
            f"Given the causal model below, determine:\n"
            f"1. Does the frontdoor criterion apply for estimating the causal effect "
            f"of {treatment} on {outcome}?\n"
            f"2. If yes, identify the mediating variable(s) that satisfy the criterion.\n"
            f"3. Write out the frontdoor adjustment formula.\n\n"
            f"Explain your reasoning step by step.\n\n"
            f"{context}"
        ),
        "ate": (
            f"Given the causal model below, compute the Average Treatment Effect (ATE) "
            f"of {treatment} on {outcome} = {outcome_val}.\n\n"
            f"ATE = P({outcome}={outcome_val} | do({treatment}={treatment_val})) - "
            f"P({outcome}={outcome_val} | do({treatment}=<control>))\n\n"
            f"Show your complete calculation, including any adjustments for confounders.\n\n"
            f"{context}"
        ),
        "structural": (
            f"Given the causal model below, answer the following:\n"
            f"1. Is {treatment} a direct cause of {outcome}?\n"
            f"2. List all directed paths from {treatment} to {outcome}.\n"
            f"3. Are there any confounders between {treatment} and {outcome}? "
            f"If so, name them.\n"
            f"4. Are there any mediators between {treatment} and {outcome}? "
            f"If so, name them.\n\n"
            f"{context}"
        ),
    }

    if query_type not in templates:
        raise ValueError(f"Unknown query_type: {query_type}. "
                         f"Choose from: {list(templates.keys())}")

    return templates[query_type]


def generate_experiment_queries(
    model: BayesianNetwork,
    graph_config: dict,
) -> List[dict]:
    """
    Generate a full set of experiment queries for a given graph.

    Returns a list of dicts, each containing:
      - query_type: the type of question
      - prompt: the natural language question
      - expected_method: what method should be used
    """
    treatment = graph_config["backdoor_test"]["treatment"]
    outcome = graph_config["backdoor_test"]["outcome"]

    # Get state names
    for cpd in model.get_cpds():
        if cpd.variable == treatment:
            t_states = cpd.state_names[treatment]
        if cpd.variable == outcome:
            o_states = cpd.state_names[outcome]

    queries = []

    # Level 1: Associational
    queries.append({
        "query_type": "associational",
        "prompt": generate_query(
            model, "associational",
            treatment, outcome, t_states[1], o_states[1],
        ),
        "expected_method": "variable_elimination",
    })

    # Level 2: Interventional
    queries.append({
        "query_type": "interventional",
        "prompt": generate_query(
            model, "interventional",
            treatment, outcome, t_states[1], o_states[1],
        ),
        "expected_method": "backdoor_adjustment_or_graph_mutilation",
    })

    # Backdoor identification
    queries.append({
        "query_type": "backdoor_identify",
        "prompt": generate_query(
            model, "backdoor_identify",
            treatment, outcome, t_states[1], o_states[1],
        ),
        "expected_method": "backdoor_criterion",
    })

    # Frontdoor identification
    if "frontdoor_test" in graph_config:
        queries.append({
            "query_type": "frontdoor_identify",
            "prompt": generate_query(
                model, "frontdoor_identify",
                treatment, outcome, t_states[1], o_states[1],
            ),
            "expected_method": "frontdoor_criterion",
        })

    # ATE
    queries.append({
        "query_type": "ate",
        "prompt": generate_query(
            model, "ate",
            treatment, outcome, t_states[1], o_states[1],
        ),
        "expected_method": "interventional_calculus",
    })

    # Structural
    queries.append({
        "query_type": "structural",
        "prompt": generate_query(
            model, "structural",
            treatment, outcome, t_states[1], o_states[1],
        ),
        "expected_method": "graph_analysis",
    })

    return queries
