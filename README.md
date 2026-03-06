# LLM Causal Inference Evaluation

Evaluating whether LLMs can perform genuine causal reasoning (using do-calculus, backdoor/frontdoor adjustment) vs. relying on memorized knowledge.

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env  # fill in your API key
```

## Run

```bash
python run_experiment.py                          # placeholder mode, no API key needed
python run_experiment.py --provider openai         # with OpenAI
python run_experiment.py --graph smoking_cancer    # specific graph only
```

## How it works

1. Define a causal DAG with known CPDs
2. Compute ground truth using exact inference (backdoor/frontdoor adjustment)
3. Generate natural language causal questions from the DAG
4. Perturb the DAG (reverse/add/remove edges, hide confounders) to create "wrong" versions
5. Query LLM with both correct and perturbed graphs
6. Compare: if the graph changes but the LLM's answer doesn't, it's likely memorizing rather than reasoning

See [PLAN.md](PLAN.md) for detailed experiment design.
