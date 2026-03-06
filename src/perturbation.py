"""
DAG perturbation strategies for adversarial evaluation.

Creates "wrong" versions of causal graphs to test whether LLMs
truly reason about causal structure vs. relying on memorized knowledge.

Perturbation types:
1. Edge reversal   - reverse the direction of a causal edge
2. Edge addition   - add a spurious edge that doesn't exist
3. Edge removal    - remove a real causal edge
4. Confounder hide - remove a confounder to see if LLM still adjusts for it
5. Collider create - add an edge that creates a collider bias
"""

import copy
import random
from typing import List, Optional, Tuple

import numpy as np
from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD


def _rebuild_cpd_for_new_parents(
    model: BayesianNetwork,
    variable: str,
    new_parents: List[str],
) -> TabularCPD:
    """Create a random CPD for a variable with new parents."""
    old_cpd = model.get_cpds(variable)
    var_card = old_cpd.variable_card
    state_names = {variable: old_cpd.state_names[variable]}

    if not new_parents:
        values = np.random.dirichlet(np.ones(var_card)).reshape(-1, 1)
        return TabularCPD(variable, var_card, values, state_names=state_names)

    parent_cards = []
    evidence_state_names = {}
    for p in new_parents:
        p_cpd = model.get_cpds(p)
        parent_cards.append(p_cpd.variable_card)
        evidence_state_names[p] = p_cpd.state_names[p]

    state_names.update(evidence_state_names)
    n_cols = int(np.prod(parent_cards))
    values = np.column_stack([
        np.random.dirichlet(np.ones(var_card)) for _ in range(n_cols)
    ])

    return TabularCPD(
        variable, var_card, values,
        evidence=new_parents, evidence_card=parent_cards,
        state_names=state_names,
    )


def reverse_edge(
    model: BayesianNetwork,
    edge: Tuple[str, str],
) -> BayesianNetwork:
    """
    Reverse a single edge in the DAG.
    Returns a new model with the reversed edge and re-fitted CPDs.
    """
    src, dst = edge
    new_edges = [e for e in model.edges() if e != (src, dst)]

    # Check that reversal doesn't create a cycle
    new_edges.append((dst, src))
    new_model = BayesianNetwork(new_edges)

    # Copy CPDs for unaffected nodes, rebuild for affected ones
    for node in new_model.nodes():
        new_parents = list(new_model.get_parents(node))
        old_parents = list(model.get_parents(node))
        if set(new_parents) == set(old_parents):
            new_model.add_cpds(copy.deepcopy(model.get_cpds(node)))
        else:
            new_cpd = _rebuild_cpd_for_new_parents(model, node, new_parents)
            new_model.add_cpds(new_cpd)

    assert new_model.check_model()
    return new_model


def add_edge(
    model: BayesianNetwork,
    edge: Tuple[str, str],
) -> BayesianNetwork:
    """
    Add a spurious edge to the DAG.
    Returns a new model with the added edge.
    """
    src, dst = edge
    if (src, dst) in model.edges():
        raise ValueError(f"Edge {edge} already exists")

    new_edges = list(model.edges()) + [(src, dst)]
    new_model = BayesianNetwork(new_edges)

    for node in new_model.nodes():
        new_parents = list(new_model.get_parents(node))
        old_parents = list(model.get_parents(node))
        if set(new_parents) == set(old_parents):
            new_model.add_cpds(copy.deepcopy(model.get_cpds(node)))
        else:
            new_cpd = _rebuild_cpd_for_new_parents(model, node, new_parents)
            new_model.add_cpds(new_cpd)

    assert new_model.check_model()
    return new_model


def remove_edge(
    model: BayesianNetwork,
    edge: Tuple[str, str],
) -> BayesianNetwork:
    """
    Remove an edge from the DAG.
    Returns a new model without the specified edge.
    """
    new_edges = [e for e in model.edges() if e != edge]
    new_model = BayesianNetwork(new_edges)

    for node in new_model.nodes():
        new_parents = list(new_model.get_parents(node))
        old_parents = list(model.get_parents(node))
        if set(new_parents) == set(old_parents):
            new_model.add_cpds(copy.deepcopy(model.get_cpds(node)))
        else:
            new_cpd = _rebuild_cpd_for_new_parents(model, node, new_parents)
            new_model.add_cpds(new_cpd)

    assert new_model.check_model()
    return new_model


def hide_confounder(
    model: BayesianNetwork,
    confounder: str,
) -> BayesianNetwork:
    """
    Remove a confounder node entirely from the DAG.
    This tests whether the LLM recognizes that adjustment is needed
    even when the confounder is not mentioned.
    """
    new_edges = [e for e in model.edges()
                 if e[0] != confounder and e[1] != confounder]
    remaining_nodes = set(model.nodes()) - {confounder}
    new_model = BayesianNetwork(new_edges)
    # Ensure all nodes are present even if isolated
    for node in remaining_nodes:
        if node not in new_model.nodes():
            new_model.add_node(node)

    for node in new_model.nodes():
        new_parents = list(new_model.get_parents(node))
        old_parents = list(model.get_parents(node))
        if set(new_parents) == set(old_parents):
            new_model.add_cpds(copy.deepcopy(model.get_cpds(node)))
        else:
            new_cpd = _rebuild_cpd_for_new_parents(model, node, new_parents)
            new_model.add_cpds(new_cpd)

    assert new_model.check_model()
    return new_model


# ---------------------------------------------------------------------------
# Perturbation registry: describes all perturbation strategies
# ---------------------------------------------------------------------------

PERTURBATION_TYPES = {
    "edge_reversal": {
        "function": reverse_edge,
        "description": "Reverse the direction of a causal edge",
        "tests": "Does LLM detect incorrect causal direction?",
    },
    "edge_addition": {
        "function": add_edge,
        "description": "Add a spurious causal edge",
        "tests": "Does LLM reject non-existent causal relationships?",
    },
    "edge_removal": {
        "function": remove_edge,
        "description": "Remove a real causal edge",
        "tests": "Does LLM notice missing causal paths?",
    },
    "confounder_hide": {
        "function": hide_confounder,
        "description": "Remove a confounder from the graph",
        "tests": "Does LLM recognize hidden confounding?",
    },
}


def generate_perturbations(
    model: BayesianNetwork,
    graph_name: str,
    seed: int = 42,
) -> List[dict]:
    """
    Generate a set of perturbed graphs from the original model.
    Returns list of dicts with perturbation info and the perturbed model.
    """
    random.seed(seed)
    np.random.seed(seed)
    edges = list(model.edges())
    nodes = list(model.nodes())
    perturbations = []

    # Edge reversals (try each edge)
    for edge in edges:
        try:
            perturbed = reverse_edge(model, edge)
            perturbations.append({
                "graph_name": graph_name,
                "type": "edge_reversal",
                "detail": f"Reversed {edge[0]} -> {edge[1]} to {edge[1]} -> {edge[0]}",
                "model": perturbed,
            })
        except Exception:
            pass  # reversal may create cycle

    # Edge removal (each edge)
    for edge in edges:
        try:
            perturbed = remove_edge(model, edge)
            perturbations.append({
                "graph_name": graph_name,
                "type": "edge_removal",
                "detail": f"Removed edge {edge[0]} -> {edge[1]}",
                "model": perturbed,
            })
        except Exception:
            pass

    # Edge addition (sample a few non-existing edges)
    existing = set(edges)
    for src in nodes:
        for dst in nodes:
            if src != dst and (src, dst) not in existing:
                try:
                    perturbed = add_edge(model, (src, dst))
                    perturbations.append({
                        "graph_name": graph_name,
                        "type": "edge_addition",
                        "detail": f"Added spurious edge {src} -> {dst}",
                        "model": perturbed,
                    })
                except Exception:
                    pass

    return perturbations
