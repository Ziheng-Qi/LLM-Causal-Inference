"""
Evaluator: compares LLM responses against ground truth.

Evaluation dimensions:
1. Numerical accuracy  - Is the computed probability correct (within tolerance)?
2. Method correctness  - Did the LLM use the right causal inference method?
3. Structural accuracy - Did the LLM correctly identify graph properties?
4. Robustness          - Does the LLM give different answers for perturbed graphs?
"""

import re
import json
from typing import Dict, List, Optional, Tuple

from .llm_client import LLMClient


def extract_numerical_answer(response: str) -> Optional[float]:
    """
    Extract the final numerical probability from an LLM response.
    Looks for patterns like "= 0.35", "is 0.35", "approximately 0.35".
    """
    # Try to find explicit "final answer" patterns
    patterns = [
        r"(?:final answer|result|therefore|thus|so)[:\s]*(?:is\s+)?(?:approximately\s+)?(\d+\.?\d*)",
        r"P\([^)]+\)\s*=\s*(\d+\.?\d*)",
        r"(?:=|≈)\s*(\d+\.?\d*)\s*$",
        r"(\d+\.?\d*)\s*$",  # last number in the response
    ]
    for pattern in patterns:
        matches = re.findall(pattern, response, re.IGNORECASE | re.MULTILINE)
        if matches:
            val = float(matches[-1])
            if 0 <= val <= 1:
                return val
    return None


def extract_method_used(response: str) -> List[str]:
    """
    Detect which causal inference methods the LLM mentioned/used.
    """
    method_keywords = {
        "backdoor": ["backdoor", "back-door", "back door"],
        "frontdoor": ["frontdoor", "front-door", "front door"],
        "do_calculus": ["do-calculus", "do calculus", "intervention", "do("],
        "graph_mutilation": ["mutilat", "truncat", "removing incoming edges"],
        "variable_elimination": ["variable elimination", "marginalization", "sum over"],
        "bayes_rule": ["bayes", "conditional probability", "conditioning"],
        "adjustment_formula": ["adjustment formula", "adjustment set", "adjust for"],
        "naive_conditioning": ["condition on", "given that", "P(Y|X)"],
    }

    found = []
    response_lower = response.lower()
    for method, keywords in method_keywords.items():
        if any(kw in response_lower for kw in keywords):
            found.append(method)
    return found


def evaluate_numerical(
    llm_answer: Optional[float],
    ground_truth: float,
    tolerance: float = 0.05,
) -> dict:
    """Evaluate numerical accuracy of the LLM's answer."""
    if llm_answer is None:
        return {
            "correct": False,
            "error": None,
            "reason": "Could not extract numerical answer from response",
        }

    error = abs(llm_answer - ground_truth)
    return {
        "correct": error <= tolerance,
        "error": round(error, 6),
        "llm_answer": round(llm_answer, 6),
        "ground_truth": round(ground_truth, 6),
        "tolerance": tolerance,
    }


def evaluate_method(
    methods_used: List[str],
    expected_methods: List[str],
) -> dict:
    """Evaluate whether the LLM used the correct method."""
    used_set = set(methods_used)
    expected_set = set(expected_methods)
    correct_methods = used_set & expected_set
    wrong_methods = used_set - expected_set

    return {
        "correct_methods_used": list(correct_methods),
        "wrong_methods_used": list(wrong_methods),
        "expected_methods": expected_methods,
        "method_correct": len(correct_methods) > 0,
    }


def evaluate_single_query(
    query_info: dict,
    ground_truth_value: Optional[float],
    llm_response: str,
    expected_methods: List[str],
    tolerance: float = 0.05,
) -> dict:
    """
    Full evaluation of a single query.
    """
    numerical_answer = extract_numerical_answer(llm_response)
    methods_used = extract_method_used(llm_response)

    result = {
        "query_type": query_info["query_type"],
        "methods_detected": methods_used,
    }

    # Numerical evaluation (if applicable)
    if ground_truth_value is not None:
        result["numerical"] = evaluate_numerical(
            numerical_answer, ground_truth_value, tolerance
        )
    else:
        result["numerical"] = {"skipped": True, "reason": "No numerical ground truth"}

    # Method evaluation
    result["method"] = evaluate_method(methods_used, expected_methods)

    # Overall score
    num_correct = result["numerical"].get("correct", True)
    method_correct = result["method"]["method_correct"]
    result["overall_correct"] = num_correct and method_correct

    return result


def evaluate_perturbation_robustness(
    original_response: str,
    perturbed_response: str,
    perturbation_info: dict,
) -> dict:
    """
    Evaluate whether the LLM's answer changes appropriately
    when given a perturbed graph.

    Key insight: if the graph structure changes but the LLM gives
    the same answer, it might be relying on memorized knowledge
    rather than actually reasoning from the graph.
    """
    orig_answer = extract_numerical_answer(original_response)
    perturbed_answer = extract_numerical_answer(perturbed_response)
    orig_methods = extract_method_used(original_response)
    perturbed_methods = extract_method_used(perturbed_response)

    # Check if the answer changed
    if orig_answer is not None and perturbed_answer is not None:
        answer_changed = abs(orig_answer - perturbed_answer) > 0.01
    else:
        answer_changed = None

    method_changed = set(orig_methods) != set(perturbed_methods)

    return {
        "perturbation_type": perturbation_info["type"],
        "perturbation_detail": perturbation_info["detail"],
        "original_answer": orig_answer,
        "perturbed_answer": perturbed_answer,
        "answer_changed": answer_changed,
        "method_changed": method_changed,
        "original_methods": orig_methods,
        "perturbed_methods": perturbed_methods,
        # If the graph changed but answer didn't, LLM may be using memorization
        "possible_memorization": answer_changed is False,
    }


def run_evaluation_suite(
    queries: List[dict],
    ground_truths: Dict,
    llm_client: LLMClient,
) -> List[dict]:
    """
    Run the full evaluation suite: send all queries to LLM and evaluate.
    """
    results = []

    for query_info in queries:
        print(f"  Evaluating: {query_info['query_type']}...")

        # Query LLM
        llm_result = llm_client.query(query_info["prompt"])

        # Determine ground truth value for this query type
        gt_value = None
        expected_methods = []

        if query_info["query_type"] == "associational":
            # Get the first observational value as ground truth
            obs_vals = list(ground_truths.get("observational", {}).values())
            gt_value = obs_vals[-1] if obs_vals else None
            expected_methods = ["variable_elimination", "bayes_rule", "naive_conditioning"]

        elif query_info["query_type"] == "interventional":
            int_vals = list(ground_truths.get("interventional", {}).values())
            gt_value = int_vals[-1] if int_vals else None
            expected_methods = ["backdoor", "frontdoor", "do_calculus", "graph_mutilation"]

        elif query_info["query_type"] == "ate":
            gt_value = ground_truths.get("ate")
            expected_methods = ["backdoor", "do_calculus", "adjustment_formula"]

        elif query_info["query_type"] == "backdoor_identify":
            expected_methods = ["backdoor", "adjustment_formula"]

        elif query_info["query_type"] == "frontdoor_identify":
            expected_methods = ["frontdoor"]

        elif query_info["query_type"] == "structural":
            expected_methods = []  # structural questions don't need specific methods

        # Evaluate
        eval_result = evaluate_single_query(
            query_info, gt_value, llm_result["response"],
            expected_methods,
        )
        eval_result["llm_response"] = llm_result["response"]
        eval_result["llm_model"] = llm_result["model"]
        eval_result["llm_latency_s"] = llm_result["latency_s"]
        results.append(eval_result)

    return results
