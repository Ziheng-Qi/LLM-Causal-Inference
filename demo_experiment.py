"""
Demo experiment: Does the LLM truly reason with the causal graph?

Three experiments:
1. Graph vs No-Graph: Does providing a DAG change the LLM's answer?
2. Identification Strategy: Can the LLM correctly identify BD/FD adjustment sets?
3. Perturbation Sensitivity: Does the LLM's strategy change when the graph changes?

This tests CAUSAL REASONING (identification strategy), not arithmetic.
"""

import json
import os
from datetime import datetime
from src.llm_client import LLMClient
from src.causal_graphs import GRAPH_REGISTRY
from src.mimic_dag import build_mimic_dag, load_mimic_data
from src.perturbation import reverse_edge, remove_edge, add_edge, hide_confounder
from pgmpy.models import DiscreteBayesianNetwork as BayesianNetwork
from pgmpy.inference import CausalInference


# ── Helpers ──────────────────────────────────────────────────────────────

MIMIC_VARIABLE_DESCRIPTIONS = {
    "Age":       "patient age group (young: <=65, old: >65)",
    "Gender":    "patient sex (M/F)",
    "Severity":  "illness severity at ICU admission (low/high), proxied by number of diagnoses",
    "ICU_type":  "type of ICU the patient was admitted to (medical/surgical)",
    "LOS_long":  "ICU length of stay (short: <3 days, long: >=3 days)",
    "Mortality": "in-hospital mortality outcome (survived/died)",
}

MIMIC_CONTEXT = """This causal graph is derived from MIMIC-IV, a real ICU dataset from Beth Israel Deaconess Medical Center (2008-2019), containing 94,458 ICU admissions.

Variable descriptions:
""" + "\n".join(f"  - {k}: {v}" for k, v in MIMIC_VARIABLE_DESCRIPTIONS.items())


def dag_to_text(model: BayesianNetwork, include_mimic_context: bool = False) -> str:
    """Convert DAG to human-readable text (no CPD values)."""
    edges = list(model.edges())
    lines = []
    if include_mimic_context:
        lines.append(MIMIC_CONTEXT)
        lines.append("")
    lines.append("Causal graph (directed edges):")
    for src, dst in edges:
        lines.append(f"  {src} -> {dst}")
    lines.append(f"Variables: {', '.join(sorted(model.nodes()))}")
    return "\n".join(lines)


def get_ground_truth_adjustment(model: BayesianNetwork, treatment: str, outcome: str) -> dict:
    """Compute ground truth: is effect identifiable? What's the adjustment set?"""
    ci = CausalInference(model)

    result = {"treatment": treatment, "outcome": outcome}

    # Backdoor
    try:
        bd_set = ci.get_minimal_adjustment_set(treatment, outcome)
        result["backdoor_set"] = sorted(list(bd_set))
        result["backdoor_applicable"] = True
    except Exception:
        result["backdoor_set"] = []
        result["backdoor_applicable"] = False

    # Check if treatment directly causes outcome
    result["direct_edge"] = (treatment, outcome) in model.edges()

    # All directed paths
    import networkx as nx
    G = nx.DiGraph(model.edges())
    try:
        all_paths = list(nx.all_simple_paths(G, treatment, outcome))
        result["directed_paths"] = [" -> ".join(p) for p in all_paths]
    except nx.NetworkXError:
        result["directed_paths"] = []

    return result


# ── Experiment 1: Graph vs No-Graph ──────────────────────────────────────

def experiment_1_graph_vs_no_graph(llm: LLMClient, model: BayesianNetwork,
                                    treatment: str, outcome: str) -> dict:
    """
    Test: Does providing the DAG change the LLM's identification strategy?

    If answers are the same with and without the graph,
    the LLM is using memorized knowledge, not the graph.
    """
    print("\n" + "="*60)
    print("EXPERIMENT 1: Graph vs No-Graph")
    print("="*60)

    variables = sorted(model.nodes())

    # Condition A: WITH graph
    prompt_with_graph = f"""You are given the following causal graph:

{dag_to_text(model, include_mimic_context=True)}

Question: To estimate the causal effect of {treatment} on {outcome},
i.e., E[{outcome} | do({treatment})]:

1. Is this causal effect identifiable from observational data?
2. What identification strategy should be used? (backdoor adjustment, frontdoor adjustment, or other?)
3. What is the minimal sufficient adjustment set?
4. Write out the identification formula.

Answer based ONLY on the graph structure provided above. Do NOT use any domain knowledge."""

    # Condition B: WITHOUT graph (only variable names)
    prompt_no_graph = f"""Consider the following variables in a medical study:
{', '.join(variables)}

Question: To estimate the causal effect of {treatment} on {outcome},
i.e., E[{outcome} | do({treatment})]:

1. Is this causal effect identifiable from observational data?
2. What identification strategy should be used? (backdoor adjustment, frontdoor adjustment, or other?)
3. What is the minimal sufficient adjustment set?
4. Write out the identification formula.

Note: You are NOT given a causal graph. Answer based on your best judgment."""

    print("\n  Querying LLM WITH graph...")
    resp_with = llm.query(prompt_with_graph)
    print(f"  Response length: {len(resp_with['response'])} chars")

    print("  Querying LLM WITHOUT graph...")
    resp_without = llm.query(prompt_no_graph)
    print(f"  Response length: {len(resp_without['response'])} chars")

    # Ground truth
    gt = get_ground_truth_adjustment(model, treatment, outcome)

    return {
        "experiment": "graph_vs_no_graph",
        "ground_truth": gt,
        "with_graph": {
            "prompt": prompt_with_graph,
            "response": resp_with["response"],
        },
        "without_graph": {
            "prompt": prompt_no_graph,
            "response": resp_without["response"],
        },
    }


# ── Experiment 2: Identification Strategy Correctness ────────────────────

def experiment_2_identification(llm: LLMClient, model: BayesianNetwork,
                                 treatment: str, outcome: str) -> dict:
    """
    Test: Can the LLM correctly identify the adjustment set from the graph?

    This is pure causal reasoning - no arithmetic involved.
    """
    print("\n" + "="*60)
    print("EXPERIMENT 2: Identification Strategy Correctness")
    print("="*60)

    prompt = f"""You are given the following causal graph:

{dag_to_text(model, include_mimic_context=True)}

Answer the following questions about estimating the causal effect of {treatment} on {outcome}:

1. List ALL backdoor paths from {treatment} to {outcome}. (A backdoor path is a path that starts with an arrow INTO {treatment}.)
2. Is the backdoor criterion satisfied? If yes, what is the minimal sufficient adjustment set?
3. Are there any variables that should NOT be adjusted for (e.g., colliders, mediators)? Explain why.
4. Write the backdoor adjustment formula for E[{outcome} | do({treatment})].

Be precise and base your answers ONLY on the graph structure above."""

    print("\n  Querying LLM...")
    resp = llm.query(prompt)

    gt = get_ground_truth_adjustment(model, treatment, outcome)

    print(f"\n  Ground truth adjustment set: {gt['backdoor_set']}")
    print(f"  Ground truth BD applicable: {gt['backdoor_applicable']}")
    print(f"  Directed paths: {gt['directed_paths']}")

    return {
        "experiment": "identification_correctness",
        "ground_truth": gt,
        "prompt": prompt,
        "response": resp["response"],
    }


# ── Experiment 3: Perturbation Sensitivity ───────────────────────────────

def experiment_3_perturbation(llm: LLMClient, model: BayesianNetwork,
                               treatment: str, outcome: str) -> dict:
    """
    Test: When the graph structure changes, does the LLM's
    identification strategy change accordingly?

    Key: we only use perturbations that PROVABLY change the adjustment set.
    """
    print("\n" + "="*60)
    print("EXPERIMENT 3: Perturbation Sensitivity")
    print("="*60)

    gt_original = get_ground_truth_adjustment(model, treatment, outcome)

    base_question = """Answer the following about estimating the causal effect of {treatment} on {outcome}:
1. What is the minimal sufficient adjustment set for the backdoor criterion?
2. Is the backdoor criterion applicable? Why or why not?

Base your answer ONLY on the graph structure provided."""

    # Query on original graph
    prompt_original = f"""You are given the following causal graph:

{dag_to_text(model, include_mimic_context=True)}

{base_question.format(treatment=treatment, outcome=outcome)}"""

    print("\n  Querying on ORIGINAL graph...")
    resp_original = llm.query(prompt_original)

    # Generate perturbations and find ones that change the adjustment set
    perturbation_results = []
    edges = list(model.edges())

    perturbations_to_try = []

    # Try removing each edge
    for edge in edges:
        try:
            perturbed = remove_edge(model, edge)
            gt_perturbed = get_ground_truth_adjustment(perturbed, treatment, outcome)
            if gt_perturbed["backdoor_set"] != gt_original["backdoor_set"] or \
               gt_perturbed["backdoor_applicable"] != gt_original["backdoor_applicable"]:
                perturbations_to_try.append({
                    "type": "edge_removal",
                    "detail": f"Removed {edge[0]} -> {edge[1]}",
                    "model": perturbed,
                    "gt": gt_perturbed,
                })
        except Exception:
            pass

    # Try reversing each edge
    for edge in edges:
        try:
            perturbed = reverse_edge(model, edge)
            gt_perturbed = get_ground_truth_adjustment(perturbed, treatment, outcome)
            if gt_perturbed["backdoor_set"] != gt_original["backdoor_set"] or \
               gt_perturbed["backdoor_applicable"] != gt_original["backdoor_applicable"]:
                perturbations_to_try.append({
                    "type": "edge_reversal",
                    "detail": f"Reversed {edge[0]} -> {edge[1]}",
                    "model": perturbed,
                    "gt": gt_perturbed,
                })
        except Exception:
            pass

    # Try hiding confounders
    for node in model.nodes():
        if node in (treatment, outcome):
            continue
        try:
            perturbed = hide_confounder(model, node)
            gt_perturbed = get_ground_truth_adjustment(perturbed, treatment, outcome)
            if gt_perturbed["backdoor_set"] != gt_original["backdoor_set"]:
                perturbations_to_try.append({
                    "type": "confounder_hidden",
                    "detail": f"Removed confounder node {node}",
                    "model": perturbed,
                    "gt": gt_perturbed,
                })
        except Exception:
            pass

    print(f"  Found {len(perturbations_to_try)} answer-changing perturbations")

    # Test up to 5 perturbations
    for i, pert in enumerate(perturbations_to_try[:5]):
        print(f"\n  Testing perturbation {i+1}: {pert['detail']}")
        print(f"    Original adjustment set: {gt_original['backdoor_set']}")
        print(f"    New adjustment set:      {pert['gt']['backdoor_set']}")

        prompt_perturbed = f"""You are given the following causal graph:

{dag_to_text(pert['model'], include_mimic_context=True)}

{base_question.format(treatment=treatment, outcome=outcome)}"""

        resp = llm.query(prompt_perturbed)

        perturbation_results.append({
            "perturbation_type": pert["type"],
            "perturbation_detail": pert["detail"],
            "original_adjustment_set": gt_original["backdoor_set"],
            "perturbed_adjustment_set": pert["gt"]["backdoor_set"],
            "prompt": prompt_perturbed,
            "response": resp["response"],
        })

    return {
        "experiment": "perturbation_sensitivity",
        "ground_truth_original": gt_original,
        "original_response": resp_original["response"],
        "perturbation_results": perturbation_results,
    }


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Demo: LLM Causal Reasoning Evaluation")
    parser.add_argument("--graph", default="mimic_icu", choices=list(GRAPH_REGISTRY.keys()))
    parser.add_argument("--provider", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--experiment", default="all", choices=["1", "2", "3", "all"])
    args = parser.parse_args()

    llm = LLMClient(provider=args.provider, model=args.model)
    print(f"LLM: {llm.provider} / {llm.model}")

    # Build graph
    config = GRAPH_REGISTRY[args.graph]
    model = config["builder"]()
    treatment = config["backdoor_test"]["treatment"]
    outcome = config["backdoor_test"]["outcome"]

    print(f"\nGraph: {args.graph}")
    print(f"Treatment: {treatment}, Outcome: {outcome}")
    print(f"Edges: {list(model.edges())}")

    results = {"graph": args.graph, "llm": f"{llm.provider}/{llm.model}", "experiments": []}

    if args.experiment in ("1", "all"):
        r = experiment_1_graph_vs_no_graph(llm, model, treatment, outcome)
        results["experiments"].append(r)

    if args.experiment in ("2", "all"):
        r = experiment_2_identification(llm, model, treatment, outcome)
        results["experiments"].append(r)

    if args.experiment in ("3", "all"):
        r = experiment_3_perturbation(llm, model, treatment, outcome)
        results["experiments"].append(r)

    # Save
    os.makedirs("results", exist_ok=True)
    path = f"results/demo_{datetime.now():%Y%m%d_%H%M%S}.json"
    with open(path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to: {path}")


if __name__ == "__main__":
    main()
