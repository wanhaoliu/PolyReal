# PolyReal

PolyReal is a polymer-domain multimodal evaluation benchmark with inference and scoring scripts.

> **Dataset**: [weidawang/PolyReal on Hugging Face](https://huggingface.co/datasets/weidawang/PolyReal)  
> **Code**: [wanhaoliu/PolyReal on GitHub](https://github.com/wanhaoliu/PolyReal)

## Getting Started

### 1. Clone the code

```bash
git clone git@github.com:wanhaoliu/PolyReal.git
cd PolyReal
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Download the dataset from Hugging Face

```bash
pip install huggingface_hub

python - <<'EOF'
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="weidawang/PolyReal",
    repo_type="dataset",
    local_dir=".",        # download into the project root
)
EOF
```

After downloading, the project root should contain:

```
PolyReal.json      # Main dataset (545 questions)
ref/               # Reference images (PNG) and CSV files
```

### 4. Set environment variables

```bash
export POLYREAL_API_BASE_URL="https://your-api-host.example.com"
export POLYREAL_API_KEY="your-api-key"
```

> If you are using `--model intern-s1`, also set:
> ```bash
> export INTERN_S1_API_BASE_URL="https://your-intern-s1-host.example.com/api"
> export INTERN_S1_API_KEY="your-intern-s1-api-key"
> ```

The `POLYREAL_API_BASE_URL` must point to an **OpenAI-compatible** `/v1/chat/completions` endpoint (e.g. OpenAI, Together, OpenRouter, or a local vLLM instance).

---

## Running the Evaluation Pipeline

### Step 1 — Inference

```bash
python test.py --model gpt-4o
```

Results are saved to `result/gpt-4o/results_gpt-4o.jsonl`. Already-processed items are automatically skipped on re-run (resume support).

| Argument | Default | Description |
|----------|---------|-------------|
| `--model` | `gpt-4o` | Model name sent to the API |
| `--workers` | `10` | Number of concurrent threads |
| `--input_file` | `PolyReal.json` | Dataset path |
| `--image_dir` | `ref/` | Directory with reference images and CSVs |
| `--output_dir` | `result/` | Root output directory |

### Step 2 — Evaluate precision

```bash
python eval_precision.py --model gpt-4o --eval_model gemini-2.5-flash
```

Uses a second LLM to count true/false positives and compute `Precision = TP / (TP + FP)`.  
Output: `result/gpt-4o/precision_gpt-4o.jsonl`

### Step 3 — Evaluate recall

```bash
python eval_recall.py --model gpt-4o --eval_model gemini-2.5-flash
```

Uses a second LLM to score coverage and quality of each key scoring point.  
Output: `result/gpt-4o/recall_gpt-4o.jsonl`

### Step 4 — Evaluate ranking tasks

```bash
python eval_ranking.py
```

Computes strict accuracy, pairwise accuracy, and F1 for the 34 ranking questions across all model folders under `result/`.  
Output: `result/{model}/ranking_{model}.jsonl`

---

## Repository Layout

```
PolyReal.json          # Dataset — download from Hugging Face (see above)
ref/                   # Reference images/CSVs — download from Hugging Face
test.py                # Inference
eval_precision.py      # Precision evaluation
eval_recall.py         # Recall / completeness evaluation
eval_ranking.py        # Ranking task evaluation
open_source_config.py  # Shared path and API config helpers
requirements.txt       # Python dependencies
result/                # Inference & eval outputs (git-ignored)
logs/                  # Log files (git-ignored)
```

## Notes

- API keys are read from environment variables only — never stored in the repository.
- `eval_precision.py` and `eval_recall.py` skip ranking questions automatically; use `eval_ranking.py` for those.

Please add your preferred open-source license before publishing.
