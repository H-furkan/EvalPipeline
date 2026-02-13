# EvalPipeline

A modular evaluation pipeline for benchmarking AI-generated presentations against baseline methods. It compares a primary method ("ours") against multiple baselines across content, design, coherence, and fluency dimensions using both LLM-as-judge and traditional NLP metrics.

## Project Structure

```
EvalPipeline/
├── pipeline.py              # Main orchestrator — discovers papers, runs metrics, prints summary
├── run_eval.sh              # Shell wrapper with logging (forwards args to pipeline.py)
├── constants.py             # Single source of truth for all paths, models, and thresholds
├── data_prep/
│   └── convert_to_text.py   # PDF → markdown and PPTX (extracted.json) → plain text conversion
├── llm/
│   └── client.py            # Unified LLM client (vLLM / OpenAI) with retry and vision support
├── metrics/
│   ├── narrative_flow.py    # Pairwise: narrative structure preservation
│   ├── content_pairwise.py  # Pairwise: text quality, structure, organization, conciseness
│   ├── design_pairwise.py   # Pairwise: layout, readability, density, aesthetics, consistency (image-based)
│   ├── coherence_pairwise.py# Pairwise: logical flow, consistency, completeness
│   ├── quiz_eval.py         # Q&A information coverage from source paper
│   ├── rouge_eval.py        # ROUGE-L text overlap
│   ├── perplexity_eval.py   # Language fluency via Llama-3-8B
│   └── ppt_score_eval.py    # VLM-as-judge per-slide scoring (1–5)
├── prompts/                 # Prompt templates for each LLM-based metric
├── utils/
│   ├── result_utils.py      # JSON result persistence, aggregation, and console summaries
│   └── image_utils.py       # PPTX → PNG conversion (LibreOffice), resizing, slide sampling
└── results/                 # Output directory for metric JSON files and processed data
```

## Requirements

- Python 3.10+
- A running **vLLM** server (default) or an **OpenAI API** key
- **LibreOffice** (for PPTX → PNG conversion in design metrics)

### Python Dependencies

- `requests`
- `openai` (if using OpenAI backend)
- `pymupdf4llm` (PDF → markdown conversion)
- `Pillow` (image resizing)
- `transformers`, `torch` (perplexity evaluation with Llama-3-8B)
- `rouge-score` (ROUGE-L metric)

## Configuration

All settings are centralized in `constants.py`:

| Setting | Description |
|---|---|
| `GENERATED_SAMPLES_DIR` | Root directory of generated presentations (one sub-folder per method) |
| `BENCHMARK_DATA_DIR` | Directory containing original paper PDFs |
| `LLM_BACKEND` | `"vllm"` (local server) or `"openai"` (API) |
| `TEXT_MODEL` | Text-only LLM (default: `Qwen/Qwen2.5-32B-Instruct`) |
| `VISION_MODEL` | Vision-language model (default: `Qwen/Qwen2.5-VL-32B-Instruct`) |
| `OURS_METHOD` | Name of the primary method being evaluated |
| `BASELINE_METHODS` | List of baseline method names to compare against |
| `ENABLED_METRICS` | Which metrics to run by default |

## Usage

### Run all metrics on all papers

```bash
bash run_eval.sh
```

### Run with Python directly

```bash
python pipeline.py
```

### Common options

```bash
# Skip data preparation (if processed_data/ is already up-to-date)
python pipeline.py --skip-data-prep

# Run specific metrics only
python pipeline.py --metrics narrative_flow rouge_eval

# Evaluate specific papers
python pipeline.py --papers attn camera graph

# Use OpenAI backend instead of vLLM
python pipeline.py --backend openai

# Random sample of N papers
python pipeline.py --num-papers 5

# Force re-conversion of data files
python pipeline.py --force-conversion
```

## Metrics

### Pairwise (LLM-as-judge)

These metrics present slides from the primary method and a baseline side-by-side (with randomized A/B positions) and ask the LLM to choose a winner.

| Metric | Attributes Evaluated |
|---|---|
| `narrative_flow` | Preservation of the source paper's narrative structure |
| `content_pairwise` | Text quality, structure, organization, conciseness |
| `design_pairwise` | Layout, readability, text density, aesthetics, consistency |
| `coherence_pairwise` | Logical flow, consistency, completeness |

### Scalar

| Metric | Description |
|---|---|
| `quiz_eval` | Generates quiz questions from the source paper and measures how well slide content answers them |
| `rouge_eval` | ROUGE-L overlap between slide text and source paper |
| `perplexity_eval` | Language fluency scored via Llama-3-8B perplexity |
| `ppt_score_eval` | VLM rates each slide on content, style, and logic (1–5 scale) |

## Output

Results are saved as JSON files in the `results/` directory, one file per metric. Each file includes metadata (model, backend, timestamp) and per-paper breakdowns. The pipeline prints a consolidated summary to the console at the end of each run.

Logs are saved to `logs/` with timestamps when using `run_eval.sh`.
