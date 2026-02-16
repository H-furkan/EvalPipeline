# EvalPipeline

A modular evaluation pipeline for benchmarking AI-generated presentations against baseline methods. It compares a primary method ("ours") against multiple baselines across content, design, coherence, and fluency dimensions using both LLM-as-judge and traditional NLP metrics.

## What This Repository Does

This project provides a comprehensive framework for evaluating the quality of automatically generated slide presentations from scientific papers. Given:

- **Source papers** (PDF format)
- **Generated presentations** from multiple methods (PPTX or PDF format)

The pipeline converts all inputs to markdown, strips supplementary material from papers, and runs **7 evaluation metrics** covering:

- **Pairwise comparisons** (LLM-as-judge): Which method produces better slides?
- **Scalar metrics** (absolute scores): How good are each method's slides independently?
- **Statistical metrics** (no LLM needed): Text overlap, language fluency, structural stats, visual similarity

## Project Structure

```
EvalPipeline/
├── pipeline.py              # Main orchestrator - discovers papers, runs metrics, prints summary
├── run_eval.sh              # Shell wrapper with logging (forwards args to pipeline.py)
├── constants.py             # Single source of truth for all paths, models, and thresholds
├── data_prep/
│   └── convert_to_text.py   # PPTX/PDF → markdown (direct text extraction)
│                            # Includes supplementary material stripping from papers
├── llm/
│   └── client.py            # Unified LLM client (vLLM / OpenAI) with retry and vision support
├── metrics/
│   ├── narrative_flow.py    # Pairwise: narrative structure preservation
│   ├── design_pairwise.py   # Pairwise: layout, readability, density, aesthetics, consistency
│   ├── quiz_eval.py         # Quiz-based information coverage from source paper
│   ├── rouge_eval.py        # ROUGE-L text overlap
│   ├── perplexity_eval.py   # Language fluency via causal LM
│   ├── ppt_score_eval.py    # VLM-as-judge per-slide scoring (1-5 scale)
│   └── general_stats_eval.py# General stats: page count, character count, figure count
├── prompts/                 # Prompt templates for all LLM-based metrics
├── utils/
│   ├── result_utils.py      # JSON result persistence, aggregation, console summaries, file helpers
│   └── image_utils.py       # PPTX/PDF → PNG conversion (LibreOffice + pymupdf), resizing, sampling
└── results/                 # Output directory for metric JSON files and processed data
```

## Requirements

### System Dependencies

- **Python 3.10+**
- **LibreOffice** (for PPTX/PDF → PNG conversion in design and FID metrics)
- **CUDA GPU** (recommended for perplexity evaluation and vLLM serving)

### Python Dependencies

Install all required packages:

```bash
pip install requests pymupdf4llm pymupdf Pillow jinja2 transformers torch torchvision scipy
pip install rouge-score evaluate python-pptx
pip install openai  # Only if using OpenAI backend
pip install vllm    # For serving open-source models locally
```

### LLM Server

The pipeline uses a single **vision-language (VL) model** for all evaluation tasks (both text-only and image-based). You need one vLLM server:

**Option A: vLLM (recommended for open-source models)**

```bash
# Serve a Qwen2.5-VL model (handles both text and vision tasks)
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-VL-32B-Instruct \
    --port 7001 \
    --dtype float16 \
    --trust-remote-code \
    --tensor-parallel-size 2  # Use 2 GPUs
```

**Option B: OpenAI API**

```bash
export OPENAI_API_KEY="sk-..."
```

## Supported Models

All evaluation tasks use a single VL model. The same model handles text-only prompts (content, coherence, narrative, quiz) and multimodal prompts (design, ppt_score).

| Alias | Full Model ID | Notes |
|---|---|---|
| `qwen2.5-vl-7b` | `Qwen/Qwen2.5-VL-7B-Instruct` | Lightweight, single GPU |
| `qwen2.5-vl-32b` | `Qwen/Qwen2.5-VL-32B-Instruct` | Default model |
| `qwen2.5-vl-72b` | `Qwen/Qwen2.5-VL-72B-Instruct` | Most capable, needs multi-GPU |
| `gpt-4o` | `gpt-4o-2024-08-06` | Requires OpenAI API key |

### Perplexity Model

- `facebook/opt-125m` (loaded locally via HuggingFace transformers, no server needed)

## Configuration

All settings are centralized in `constants.py`:

| Setting | Description | Default |
|---|---|---|
| `GENERATED_SAMPLES_DIR` | Root directory of generated presentations | - |
| `BENCHMARK_DATA_DIR` | Directory containing original paper PDFs | - |
| `QUIZ_DATA_DIR` | Where quiz JSON files are cached | - |
| `PROCESSED_DATA_DIR` | Output for converted markdown files | - |
| `RESULTS_DIR` | Where metric result JSON files are saved | - |
| `LLM_BACKEND` | `"vllm"` or `"openai"` | `"vllm"` |
| `VLLM_API_BASE` | vLLM server URL | `http://localhost:7001/v1` |
| `MODEL` | VL model identifier (used for all tasks) | `Qwen/Qwen2.5-VL-32B-Instruct` |
| `OURS_METHOD` | Name of the primary method being evaluated | `ours_rst-4o` |
| `BASELINE_METHODS` | List of baseline method names | See constants.py |
| `ENABLED_METRICS` | Which metrics to run by default | All 10 metrics |
| `TEMPERATURE` | LLM sampling temperature | `0.0` (deterministic) |
| `MAX_TOKENS` | Max tokens per LLM response | `2048` |
| `MAX_SLIDES_FOR_VISUAL` | Max slides sampled for image-based evaluation | `4` |
| `IMAGE_TARGET_WIDTH` | Resize slide images to this width (px) | `480` |

### Expected Data Directory Layout

Presentations can be in **PPTX or PDF format** (mixed across methods is fine). The pipeline automatically detects and handles both formats.

```
GENERATED_SAMPLES_DIR/
  ours_rst-4o/
    {paper_name}/
      final.pptx          # or final.pdf — any PPTX or PDF file
  paper2poster_orig-4o/
    {paper_name}/...
  slidegen/
    {paper_name}/...
  gt/                     # Ground-truth presentations
    {paper_name}/...

BENCHMARK_DATA_DIR/
  {paper_name}/
    original.pdf          # Source paper PDF
```

## Step-by-Step Usage Guide

### Step 1: Configure Paths

Edit `constants.py` to set your data paths:

```python
GENERATED_SAMPLES_DIR = "/path/to/generated_samples_final"
BENCHMARK_DATA_DIR = "/path/to/final_benchmark_data/science/pdf"
QUIZ_DATA_DIR = "/path/to/final_benchmark_data/science/quiz"
PROCESSED_DATA_DIR = "/path/to/EvalPipeline/results/processed_data"
RESULTS_DIR = "/path/to/EvalPipeline/results"
PROMPTS_DIR = "/path/to/EvalPipeline/prompts"
```

### Step 2: Start the LLM Server

```bash
# Start a single VL model server (handles both text and vision tasks)
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-VL-32B-Instruct \
    --port 7001 \
    --dtype float16 \
    --trust-remote-code
```

### Step 3: Run the Pipeline

```bash
# Run all metrics on all papers
python pipeline.py

# Or use the shell wrapper (includes logging)
bash run_eval.sh
```

### Step 4: Check Results

Results are saved as JSON files in the `results/` directory:

```
results/
  narrative_flow.json
  design_pairwise.json
  quiz_eval.json
  rouge_eval.json
  perplexity_eval.json
  ppt_score_eval.json
  general_stats_eval.json
  processed_data/           # Converted markdown files
```

## Common Usage Examples

```bash
# Skip data preparation (if markdown files already exist)
python pipeline.py --skip-data-prep

# Run only specific metrics
python pipeline.py --metrics narrative_flow rouge_eval perplexity_eval

# Evaluate specific papers
python pipeline.py --papers attn camera graph

# Use OpenAI backend instead of vLLM
python pipeline.py --backend openai

# Random sample of 5 papers
python pipeline.py --num-papers 5

# Force re-conversion of all data files
python pipeline.py --force-conversion

# Use a different model
python pipeline.py --model qwen2.5-vl-72b

# Override vLLM server URL
python pipeline.py --vllm-api-base http://localhost:8000/v1

# Combine options
python pipeline.py --skip-data-prep --metrics rouge_eval quiz_eval --num-papers 10 --model qwen2.5-vl-72b
```

## Metrics Reference

### Pairwise Metrics (LLM-as-judge)

These metrics present slides from the primary method and a baseline side-by-side (with randomized A/B positions to prevent bias) and ask the LLM to choose a winner.

| Metric | Attributes Evaluated | Input Type |
|---|---|---|
| `narrative_flow` | Preservation of the source paper's narrative structure | Text |
| `design_pairwise` | Layout, readability, text density, aesthetics, consistency (5 attributes) | Images |

### Scalar Metrics

| Metric | Description | Model Used |
|---|---|---|
| `quiz_eval` | Generates 50 simple + 50 detail quiz questions from the source paper, then tests how well slide content can answer them | VL model |
| `rouge_eval` | ROUGE-L F1 overlap between slide text and source paper | None (text-based) |
| `perplexity_eval` | Language fluency measured via causal LM perplexity (lower = more fluent) | OPT-125M (local) |
| `ppt_score_eval` | VLM rates each slide on content (1-5), style (1-5), and logic (1-5) | VL model |
| `general_stats_eval` | Counts pages, characters, and figures per presentation | None (PPTX/PDF parsing) |

## Data Preparation

### Presentation Text Extraction

The pipeline extracts text directly from presentation files:
- **PPTX files**: Text is extracted using `python-pptx` (reads text from all shapes/frames)
- **PDF files**: Text is extracted using `pymupdf` (reads text from each page)

Both formats are auto-detected. Each method's output directory is searched for PPTX first, then PDF.

### Slide Image Conversion

For image-based metrics (design, FID, ppt_score), presentations are converted to PNG images:
- **PPTX/PDF → PNG**: Via LibreOffice headless, with pymupdf as fallback for PDFs
- Images are cached in `IMAGES_CACHE_DIR` to avoid re-conversion

### Supplementary Material Stripping

When converting source paper PDFs to markdown, the pipeline automatically strips supplementary material (appendices, supporting information) that appears **after** the references section.

**Safety guarantees:**
- The references section is **always preserved** intact
- Content **before** references is **never** removed
- Only content **after** references with clear supplementary headings (e.g., "Appendix A", "Supplementary Material") is removed
- If no references section or no supplementary material is found, the text is returned unchanged

## Resumability

The pipeline supports incremental evaluation. If a run is interrupted:

- Already-evaluated papers/methods are **automatically skipped** on the next run
- Partial results are saved after each paper via atomic JSON writes
- Use `--force-conversion` to re-process data files from scratch
- Delete individual metric JSON files from `results/` to re-run specific metrics

## Output Format

Each metric produces a JSON file with this structure:

```json
{
  "metadata": {
    "metric": "metric_name",
    "timestamp": "2025-01-01T00:00:00",
    "model": "Qwen/Qwen2.5-VL-32B-Instruct",
    "backend": "vllm",
    "ours_method": "ours_rst-4o",
    "baseline_methods": ["paper2poster_orig-4o", "slidegen", ...],
    "total_papers": 20
  },
  "overall_summary": { ... },
  "per_method_summary": {
    "method_name": { "mean_score": 0.85, "papers_evaluated": 20 }
  },
  "per_paper": {
    "paper_name": { "method_name": { ... detailed results ... } }
  }
}
```

Logs are saved to `logs/` with timestamps when using `run_eval.sh`.
