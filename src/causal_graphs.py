"""
Causal graph definitions with Conditional Probability Distributions (CPDs).

Provides example DAGs for evaluation. When MIMIC access is available,
add real clinical DAGs here.
"""

import numpy as np
from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD


# ---------------------------------------------------------------------------
# Example 1: Simple medical DAG (Smoking -> Cancer -> Dyspnea)
#   with confounder (Age) and mediator (Tar Deposit)
#
#   DAG:
#       Age ──────────┐
#        │            ▼
#        ▼         Cancer ──► Dyspnea
#     Smoking ──► TarDeposit
#        │            │
#        └────────────┘  (Smoking -> Cancer via TarDeposit = frontdoor)
#
#   This graph allows testing:
#   - Backdoor adjustment (Age confounds Smoking -> Cancer)
#   - Frontdoor adjustment (TarDeposit mediates Smoking -> Cancer)
# ---------------------------------------------------------------------------

def build_smoking_cancer_graph() -> BayesianNetwork:
    """
    A small medical causal DAG designed to test backdoor & frontdoor criteria.
    All variables are binary (0/1).
    """
    model = BayesianNetwork([
        ("Age", "Smoking"),
        ("Age", "Cancer"),
        ("Smoking", "TarDeposit"),
        ("TarDeposit", "Cancer"),
        ("Cancer", "Dyspnea"),
    ])

    # --- CPDs (chosen to create non-trivial causal effects) ---

    # P(Age=old) = 0.4
    cpd_age = TabularCPD("Age", 2, [[0.6], [0.4]],
                         state_names={"Age": ["young", "old"]})

    # P(Smoking | Age)
    cpd_smoking = TabularCPD(
        "Smoking", 2,
        [[0.8, 0.5],   # P(Smoking=no | Age)
         [0.2, 0.5]],  # P(Smoking=yes | Age)
        evidence=["Age"], evidence_card=[2],
        state_names={"Smoking": ["no", "yes"], "Age": ["young", "old"]},
    )

    # P(TarDeposit | Smoking)
    cpd_tar = TabularCPD(
        "TarDeposit", 2,
        [[0.95, 0.2],   # P(Tar=no | Smoking)
         [0.05, 0.8]],  # P(Tar=yes | Smoking)
        evidence=["Smoking"], evidence_card=[2],
        state_names={"TarDeposit": ["no", "yes"], "Smoking": ["no", "yes"]},
    )

    # P(Cancer | Age, TarDeposit)
    cpd_cancer = TabularCPD(
        "Cancer", 2,
        # columns: (Age=young,Tar=no), (Age=young,Tar=yes),
        #          (Age=old,Tar=no),   (Age=old,Tar=yes)
        [[0.95, 0.6, 0.8, 0.3],   # Cancer=no
         [0.05, 0.4, 0.2, 0.7]],  # Cancer=yes
        evidence=["Age", "TarDeposit"], evidence_card=[2, 2],
        state_names={
            "Cancer": ["no", "yes"],
            "Age": ["young", "old"],
            "TarDeposit": ["no", "yes"],
        },
    )

    # P(Dyspnea | Cancer)
    cpd_dyspnea = TabularCPD(
        "Dyspnea", 2,
        [[0.9, 0.3],   # Dyspnea=no
         [0.1, 0.7]],  # Dyspnea=yes
        evidence=["Cancer"], evidence_card=[2],
        state_names={"Dyspnea": ["no", "yes"], "Cancer": ["no", "yes"]},
    )

    model.add_cpds(cpd_age, cpd_smoking, cpd_tar, cpd_cancer, cpd_dyspnea)
    assert model.check_model()
    return model


# ---------------------------------------------------------------------------
# Example 2: ICU Treatment DAG (placeholder for MIMIC-style graph)
#
#   Severity ──────────┐
#      │               ▼
#      ▼          Mortality
#   Treatment ────────┘
#      │
#      ▼
#   LabResult ──► Mortality
#
#   Severity confounds Treatment -> Mortality (backdoor)
#   LabResult mediates Treatment -> Mortality (frontdoor-like)
# ---------------------------------------------------------------------------

def build_icu_treatment_graph() -> BayesianNetwork:
    """
    A simplified ICU treatment DAG mimicking MIMIC-style clinical reasoning.
    All variables are binary.
    """
    model = BayesianNetwork([
        ("Severity", "Treatment"),
        ("Severity", "Mortality"),
        ("Treatment", "LabResult"),
        ("LabResult", "Mortality"),
    ])

    # P(Severity=high) = 0.35
    cpd_severity = TabularCPD(
        "Severity", 2, [[0.65], [0.35]],
        state_names={"Severity": ["low", "high"]},
    )

    # P(Treatment | Severity) - sicker patients more likely to get treatment
    cpd_treatment = TabularCPD(
        "Treatment", 2,
        [[0.8, 0.3],
         [0.2, 0.7]],
        evidence=["Severity"], evidence_card=[2],
        state_names={"Treatment": ["no", "yes"], "Severity": ["low", "high"]},
    )

    # P(LabResult | Treatment) - treatment improves lab results
    cpd_lab = TabularCPD(
        "LabResult", 2,
        [[0.7, 0.3],
         [0.3, 0.7]],
        evidence=["Treatment"], evidence_card=[2],
        state_names={"LabResult": ["abnormal", "normal"], "Treatment": ["no", "yes"]},
    )

    # P(Mortality | Severity, LabResult)
    cpd_mortality = TabularCPD(
        "Mortality", 2,
        # (Sev=low,Lab=abn), (Sev=low,Lab=norm), (Sev=high,Lab=abn), (Sev=high,Lab=norm)
        [[0.85, 0.95, 0.4, 0.7],
         [0.15, 0.05, 0.6, 0.3]],
        evidence=["Severity", "LabResult"], evidence_card=[2, 2],
        state_names={
            "Mortality": ["survived", "died"],
            "Severity": ["low", "high"],
            "LabResult": ["abnormal", "normal"],
        },
    )

    model.add_cpds(cpd_severity, cpd_treatment, cpd_lab, cpd_mortality)
    assert model.check_model()
    return model


# ---------------------------------------------------------------------------
# Placeholder: MIMIC-derived DAG
# ---------------------------------------------------------------------------

def build_mimic_graph() -> BayesianNetwork:
    """
    TODO: Build a DAG from actual MIMIC-IV data once access is granted.
    Steps:
      1. Select clinically relevant variables (e.g., ventilation, vasopressors,
         SOFA score, lab values, mortality)
      2. Define DAG structure based on clinical knowledge / literature
      3. Learn CPDs from MIMIC data using pgmpy's MaximumLikelihoodEstimator
    """
    raise NotImplementedError(
        "MIMIC DAG requires dataset access. Use build_icu_treatment_graph() "
        "as a stand-in for now."
    )


# ---------------------------------------------------------------------------
# Registry: all available graphs
# ---------------------------------------------------------------------------

GRAPH_REGISTRY = {
    "smoking_cancer": {
        "builder": build_smoking_cancer_graph,
        "description": "Smoking -> TarDeposit -> Cancer with Age confounder",
        "backdoor_test": {
            "treatment": "Smoking",
            "outcome": "Cancer",
            "confounders": ["Age"],
        },
        "frontdoor_test": {
            "treatment": "Smoking",
            "outcome": "Cancer",
            "mediator": "TarDeposit",
        },
    },
    "icu_treatment": {
        "builder": build_icu_treatment_graph,
        "description": "ICU Treatment -> LabResult -> Mortality with Severity confounder",
        "backdoor_test": {
            "treatment": "Treatment",
            "outcome": "Mortality",
            "confounders": ["Severity"],
        },
        "frontdoor_test": {
            "treatment": "Treatment",
            "outcome": "Mortality",
            "mediator": "LabResult",
        },
    },
}
