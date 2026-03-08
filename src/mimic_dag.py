"""
Build a causal DAG from MIMIC-IV data.

DAG structure (based on clinical knowledge):

    Age ──────────────┐
     │                │
     ▼                ▼
  Severity ────► Treatment ────► Mortality
     │                │              ▲
     │                ▼              │
     └──────────► LabResult ────────┘

Variables (all discretized to binary):
  - Age:        young (<=65) / old (>65)
  - Gender:     M / F
  - Severity:   low / high  (based on number of diagnoses as proxy)
  - ICU_type:   medical / surgical (first_careunit)
  - LOS_long:   short (<3 days) / long (>=3 days)
  - Mortality:  survived / died

The DAG:
  Age → Severity  (older patients tend to be sicker)
  Age → Mortality (age is independent risk factor)
  Gender → Severity
  Severity → ICU_type  (sicker patients go to specific ICUs)
  Severity → LOS_long  (sicker patients stay longer)
  Severity → Mortality (direct effect)
  ICU_type → Mortality (care setting affects outcome)
  LOS_long → Mortality (longer stay associated with outcome)

This gives us:
  - Backdoor: Severity confounds ICU_type → Mortality
  - Frontdoor-like: LOS_long mediates Severity → Mortality
"""

import pandas as pd
import numpy as np
from pathlib import Path
from pgmpy.models import DiscreteBayesianNetwork as BayesianNetwork
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.factors.discrete import TabularCPD

DATA_DIR = Path(__file__).parent.parent / "data" / "mimic"


def load_mimic_data() -> pd.DataFrame:
    """Load and merge MIMIC tables into a flat analysis table."""
    patients = pd.read_csv(DATA_DIR / "patients.csv")
    admissions = pd.read_csv(DATA_DIR / "admissions.csv")
    icustays = pd.read_csv(DATA_DIR / "icustays.csv")
    diagnoses = pd.read_csv(DATA_DIR / "diagnoses_icd.csv")

    # Count diagnoses per admission as severity proxy
    diag_counts = (
        diagnoses.groupby("hadm_id")["icd_code"]
        .count()
        .reset_index()
        .rename(columns={"icd_code": "n_diagnoses"})
    )

    # Merge: icustays ← admissions ← patients ← diag_counts
    df = icustays.merge(admissions, on=["subject_id", "hadm_id"], how="inner")
    df = df.merge(patients, on="subject_id", how="inner")
    df = df.merge(diag_counts, on="hadm_id", how="left")
    df["n_diagnoses"] = df["n_diagnoses"].fillna(0)

    # --- Discretize variables ---

    # Age: young (<=65) vs old (>65)
    df["Age"] = np.where(df["anchor_age"] > 65, "old", "young")

    # Gender
    df["Gender"] = df["gender"]

    # Severity: based on median split of diagnosis count
    median_diag = df["n_diagnoses"].median()
    df["Severity"] = np.where(df["n_diagnoses"] > median_diag, "high", "low")

    # ICU type: medical vs surgical (simplify careunit names)
    df["ICU_type"] = df["first_careunit"].apply(_classify_icu)

    # LOS: short (<3 days) vs long (>=3 days)
    df["LOS_long"] = np.where(df["los"] >= 3, "long", "short")

    # Mortality
    df["Mortality"] = np.where(
        df["hospital_expire_flag"] == 1, "died", "survived"
    )

    return df[["Age", "Gender", "Severity", "ICU_type", "LOS_long", "Mortality"]]


def _classify_icu(careunit: str) -> str:
    """Classify ICU type into medical vs surgical."""
    careunit = str(careunit).lower()
    surgical_keywords = ["surg", "csru", "tsicu", "trauma"]
    if any(kw in careunit for kw in surgical_keywords):
        return "surgical"
    return "medical"


def build_mimic_dag(data: pd.DataFrame = None) -> BayesianNetwork:
    """
    Build and fit a causal DAG from MIMIC data using MLE.
    """
    if data is None:
        data = load_mimic_data()

    print(f"  MIMIC data: {len(data)} ICU stays")
    print(f"  Mortality rate: {(data['Mortality'] == 'died').mean():.3f}")
    print(f"  Age distribution: {data['Age'].value_counts().to_dict()}")

    # Define DAG structure based on clinical knowledge
    edges = [
        ("Age", "Severity"),
        ("Age", "Mortality"),
        ("Gender", "Severity"),
        ("Severity", "ICU_type"),
        ("Severity", "LOS_long"),
        ("Severity", "Mortality"),
        ("ICU_type", "Mortality"),
        ("LOS_long", "Mortality"),
    ]

    model = BayesianNetwork(edges)

    # Learn CPDs from data using Maximum Likelihood Estimation
    model.fit(data, estimator=MaximumLikelihoodEstimator)

    assert model.check_model()
    print(f"  DAG edges: {list(model.edges())}")
    print(f"  CPDs learned from {len(data)} samples")

    return model


# --- Register in the graph registry ---

def get_mimic_config() -> dict:
    """Return config dict compatible with GRAPH_REGISTRY."""
    return {
        "builder": build_mimic_dag,
        "description": "MIMIC-IV ICU: Severity → ICU_type/LOS → Mortality with Age confounder",
        "backdoor_test": {
            "treatment": "ICU_type",
            "outcome": "Mortality",
            "confounders": ["Severity"],
        },
        "frontdoor_test": {
            "treatment": "Severity",
            "outcome": "Mortality",
            "mediator": "LOS_long",
        },
    }


if __name__ == "__main__":
    print("Loading MIMIC data...")
    data = load_mimic_data()
    print(f"\nDataset shape: {data.shape}")
    print(f"\nValue counts:")
    for col in data.columns:
        print(f"\n{col}:")
        print(data[col].value_counts())

    print("\nBuilding DAG...")
    model = build_mimic_dag(data)

    print("\nCPDs:")
    for cpd in model.get_cpds():
        print(f"\n{cpd}")
