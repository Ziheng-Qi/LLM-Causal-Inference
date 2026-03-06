"""
Main experiment runner.

Runs the full evaluation pipeline:
1. Build causal graphs (correct + perturbed)
2. Compute ground truth for each graph
3. Generate natural language queries
4. Send queries to LLM
5. Evaluate responses against ground truth
6. Generate summary report

Usage:
    python run_experiment.py                        # placeholder mode (no API key)
    python run_experiment.py --graph smoking_cancer  # specific graph
    python run_experiment.py --provider openai       # use OpenAI
"""

import argparse
import json
import os
import sys
from datetime import datetime
from typing import List

from src.causal_graphs import GRAPH_REGISTRY
from src.ground_truth import compute_ground_truth_suite
from src.perturbation import generate_perturbations
from src.query_generator import generate_experiment_queries, generate_query
from src.evaluator import (
    run_evaluation_suite,
    evaluate_perturbation_robustness,
)
from src.llm_client import LLMClient


def run_single_graph_experiment(
    graph_name: str,
    llm_client: LLMClient,
    max_perturbations: int = 3,
) -> dict:
    """Run the full experiment for a single graph."""

    config = GRAPH_REGISTRY[graph_name]
    print(f"\n{'='*60}")
    print(f"Graph: {graph_name} - {config['description']}")
    print(f"{'='*60}")

    # 1. Build graph
    model = config["builder"]()
    print(f"  Nodes: {list(model.nodes())}")
    print(f"  Edges: {list(model.edges())}")

    # 2. Compute ground truth
    treatment = config["backdoor_test"]["treatment"]
    outcome = config["backdoor_test"]["outcome"]
    ground_truth = compute_ground_truth_suite(model, treatment, outcome)
    print(f"\n  Ground Truth:")
    print(f"    Observational: {ground_truth['observational']}")
    print(f"    Interventional: {ground_truth['interventional']}")
    print(f"    ATE: {ground_truth['ate']}")
    print(f"    Naive ATE: {ground_truth['naive_ate']}")
    print(f"    Confounding Bias: {ground_truth['confounding_bias']}")
    print(f"    Adjustment Info: {ground_truth['adjustment_info']}")

    # 3. Generate queries for CORRECT graph
    queries = generate_experiment_queries(model, config)
    print(f"\n  Generated {len(queries)} queries for correct graph")

    # 4. Evaluate on correct graph
    print(f"\n  --- Evaluating on CORRECT graph ---")
    correct_results = run_evaluation_suite(queries, ground_truth, llm_client)

    # 5. Generate perturbations and evaluate
    print(f"\n  --- Generating perturbations ---")
    perturbations = generate_perturbations(model, graph_name)
    print(f"  Generated {len(perturbations)} perturbations")

    # Limit perturbations for efficiency
    perturbations = perturbations[:max_perturbations]

    perturbed_results = []
    for i, pert in enumerate(perturbations):
        print(f"\n  --- Evaluating perturbation {i+1}/{len(perturbations)}: "
              f"{pert['type']} ---")
        print(f"      {pert['detail']}")

        # Generate the same interventional query but with perturbed graph
        perturbed_model = pert["model"]
        perturbed_prompt = generate_query(
            perturbed_model, "interventional",
            treatment, outcome,
            ground_truth["query"]["treatment_values"][1],
            ground_truth["query"]["outcome_value"],
        )

        # Query LLM with perturbed graph
        perturbed_llm = llm_client.query(perturbed_prompt)

        # Compare with original response
        original_response = ""
        for r in correct_results:
            if r["query_type"] == "interventional":
                original_response = r["llm_response"]
                break

        robustness = evaluate_perturbation_robustness(
            original_response,
            perturbed_llm["response"],
            pert,
        )
        robustness["perturbed_llm_response"] = perturbed_llm["response"]
        perturbed_results.append(robustness)

    # 6. Compile results
    return {
        "graph_name": graph_name,
        "description": config["description"],
        "ground_truth": ground_truth,
        "correct_graph_results": correct_results,
        "perturbation_results": perturbed_results,
    }


def generate_summary(all_results: List[dict]) -> dict:
    """Generate a summary report across all graphs."""
    summary = {
        "total_graphs": len(all_results),
        "total_queries": 0,
        "correct_numerical": 0,
        "correct_method": 0,
        "total_perturbations": 0,
        "possible_memorization": 0,
        "per_graph": {},
    }

    for result in all_results:
        graph_name = result["graph_name"]
        correct_results = result["correct_graph_results"]
        pert_results = result["perturbation_results"]

        n_queries = len(correct_results)
        n_num_correct = sum(
            1 for r in correct_results
            if r.get("numerical", {}).get("correct", False)
        )
        n_method_correct = sum(
            1 for r in correct_results
            if r.get("method", {}).get("method_correct", False)
        )
        n_memorization = sum(
            1 for r in pert_results
            if r.get("possible_memorization", False)
        )

        summary["total_queries"] += n_queries
        summary["correct_numerical"] += n_num_correct
        summary["correct_method"] += n_method_correct
        summary["total_perturbations"] += len(pert_results)
        summary["possible_memorization"] += n_memorization

        summary["per_graph"][graph_name] = {
            "queries": n_queries,
            "numerical_accuracy": f"{n_num_correct}/{n_queries}",
            "method_accuracy": f"{n_method_correct}/{n_queries}",
            "perturbations_tested": len(pert_results),
            "possible_memorization_cases": n_memorization,
        }

    # Overall rates
    if summary["total_queries"] > 0:
        summary["overall_numerical_accuracy"] = round(
            summary["correct_numerical"] / summary["total_queries"], 3
        )
        summary["overall_method_accuracy"] = round(
            summary["correct_method"] / summary["total_queries"], 3
        )
    if summary["total_perturbations"] > 0:
        summary["memorization_rate"] = round(
            summary["possible_memorization"] / summary["total_perturbations"], 3
        )

    return summary


def main():
    parser = argparse.ArgumentParser(
        description="LLM Causal Reasoning Evaluation"
    )
    parser.add_argument(
        "--graph", type=str, default=None,
        help="Graph name to test (default: all)",
        choices=list(GRAPH_REGISTRY.keys()),
    )
    parser.add_argument(
        "--provider", type=str, default=None,
        help="LLM provider: openai, anthropic, or placeholder",
    )
    parser.add_argument(
        "--model", type=str, default=None,
        help="LLM model name",
    )
    parser.add_argument(
        "--max-perturbations", type=int, default=3,
        help="Max perturbations per graph (default: 3)",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output JSON file path",
    )
    args = parser.parse_args()

    # Initialize LLM client
    llm_client = LLMClient(provider=args.provider, model=args.model)
    print(f"LLM Provider: {llm_client.provider}")
    print(f"LLM Model: {llm_client.model}")

    # Select graphs
    graph_names = [args.graph] if args.graph else list(GRAPH_REGISTRY.keys())

    # Run experiments
    all_results = []
    for graph_name in graph_names:
        result = run_single_graph_experiment(
            graph_name, llm_client, args.max_perturbations
        )
        all_results.append(result)

    # Generate summary
    summary = generate_summary(all_results)

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(json.dumps(summary, indent=2))

    # Save results
    output_path = args.output or f"results/experiment_{datetime.now():%Y%m%d_%H%M%S}.json"
    os.makedirs(os.path.dirname(output_path) or "results", exist_ok=True)

    output_data = {
        "timestamp": datetime.now().isoformat(),
        "llm_provider": llm_client.provider,
        "llm_model": llm_client.model,
        "summary": summary,
        "detailed_results": all_results,
    }
    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2, default=str)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
