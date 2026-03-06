# LLM Causal Reasoning Evaluation Framework

## Research Question

Can LLMs perform **genuine causal reasoning** (DAG-based, using proper do-calculus),
or do they rely on **memorized knowledge / common sense** to answer causal questions?

## Pipeline Overview

```
┌─────────────┐     ┌──────────────┐     ┌────────────────┐
│ Causal Graph │────►│ Ground Truth │────►│ Query Generator│
│  (DAG + CPD) │     │   Engine     │     │  (NL prompts)  │
└──────┬──────┘     └──────────────┘     └───────┬────────┘
       │                                         │
       ▼                                         ▼
┌─────────────┐                          ┌──────────────┐
│ Perturbation│──► Wrong DAGs ──────────►│  LLM Client  │
│   Module    │                          └───────┬──────┘
└─────────────┘                                  │
                                                 ▼
                                         ┌──────────────┐
                                         │  Evaluator   │
                                         │ (compare vs  │
                                         │ ground truth)│
                                         └──────────────┘
```

## Project Structure

```
├── run_experiment.py        # Main entry point
├── requirements.txt         # Dependencies
├── .env.example             # API key template
├── src/
│   ├── causal_graphs.py     # DAG definitions with CPDs
│   ├── ground_truth.py      # Exact causal inference computation
│   ├── perturbation.py      # DAG perturbation strategies
│   ├── query_generator.py   # Natural language query templates
│   ├── llm_client.py        # LLM API wrapper (OpenAI/Anthropic)
│   └── evaluator.py         # Response evaluation & scoring
└── results/                 # Experiment outputs (gitignored)
```

## Experiment Design

### 1. Causal Graphs

Two built-in example graphs (expandable to MIMIC later):

| Graph | Variables | Tests |
|-------|-----------|-------|
| `smoking_cancer` | Age, Smoking, TarDeposit, Cancer, Dyspnea | Backdoor (Age), Frontdoor (TarDeposit) |
| `icu_treatment` | Severity, Treatment, LabResult, Mortality | Backdoor (Severity), Frontdoor-like (LabResult) |

### 2. Query Types (Pearl's Ladder)

| Level | Type | Example |
|-------|------|---------|
| L1 | Associational | P(Cancer=yes \| Smoking=yes) |
| L2 | Interventional | P(Cancer=yes \| do(Smoking=yes)) |
| L2 | Backdoor ID | "What's the adjustment set for Smoking→Cancer?" |
| L2 | Frontdoor ID | "Does TarDeposit satisfy the frontdoor criterion?" |
| L2 | ATE | Compute average treatment effect |
| Structural | Graph analysis | "Is Smoking a direct cause of Cancer?" |

### 3. Perturbation Strategies (Adversarial)

| Type | What it does | What it tests |
|------|-------------|---------------|
| Edge reversal | Flip A→B to B→A | Does LLM detect wrong direction? |
| Edge addition | Add spurious edge | Does LLM reject false relationships? |
| Edge removal | Remove real edge | Does LLM notice missing paths? |
| Confounder hide | Remove confounder node | Does LLM still account for confounding? |

### 4. Evaluation Dimensions

- **Numerical accuracy**: Is the computed probability within ±0.05 of ground truth?
- **Method correctness**: Did the LLM use backdoor/frontdoor/do-calculus (not just P(Y|X))?
- **Robustness to perturbation**: Does the answer change when the graph structure changes?
  - If graph changes but answer stays same → **possible memorization**
  - If graph changes and answer changes appropriately → **genuine reasoning**

## How to Run

```bash
# Install dependencies
pip install -r requirements.txt

# Run with placeholder (no API key needed, tests the pipeline)
python run_experiment.py

# Run with a real LLM
cp .env.example .env
# Edit .env with your API key
python run_experiment.py --provider openai --model gpt-4o

# Run specific graph only
python run_experiment.py --graph smoking_cancer
```

## Next Steps (TODO)

- [ ] Get MIMIC-IV access and build clinical DAG
- [ ] Add more LLM models for comparison (GPT-4o, Claude, Llama, etc.)
- [ ] Add counterfactual (Level 3) queries
- [ ] Add visualization of DAGs and results
- [ ] Statistical significance testing across multiple runs
- [ ] Prompt engineering experiments (CoT vs. direct, etc.)
