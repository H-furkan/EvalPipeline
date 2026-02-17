# EvalPipeline

A modular evaluation pipeline for benchmarking AI-generated scientific presentations. It compares a primary method ("ours") against multiple baselines across content, design, coherence, and fluency dimensions using 7 metrics — a mix of LLM-as-judge, vision-language model scoring, and traditional NLP measures.

---

## Table of Contents

- [What This Pipeline Does](#what-this-pipeline-does)
- [Folder Structure](#folder-structure)
- [Requirements](#requirements)
- [Configuration](#configuration)
- [How to Run](#how-to-run)
- [Data Preparation](#data-preparation)
- [Metrics Overview](#metrics-overview)
  - [1. Narrative Flow (Pairwise)](#1-narrative-flow-pairwise)
  - [2. Design Pairwise (Pairwise)](#2-design-pairwise-pairwise)
  - [3. Quiz Eval (Scalar)](#3-quiz-eval-scalar)
  - [4. ROUGE Eval (Scalar)](#4-rouge-eval-scalar)
  - [5. Perplexity Eval (Scalar)](#5-perplexity-eval-scalar)
  - [6. PPT Score Eval (Scalar)](#6-ppt-score-eval-scalar)
  - [7. General Stats Eval (Scalar)](#7-general-stats-eval-scalar)
- [Supporting Modules](#supporting-modules)
- [Output Format](#output-format)
- [Resumability](#resumability)
- [Full Configuration Reference](#full-configuration-reference)

---

## What This Pipeline Does

Given a set of **source scientific papers** (PDF) and **generated slide presentations** from multiple methods (PPTX or PDF), the pipeline:

1. Converts all inputs to markdown text (with optional GPT-4o vision fallback for image-only slides).
2. Runs **7 evaluation metrics** covering narrative quality, visual design, information coverage, text overlap, language fluency, per-slide quality, and structural statistics.
3. Outputs JSON result files with per-paper details, per-method summaries, and overall aggregated scores.

The metrics fall into two categories:

| Category | Metrics | What They Measure |
|---|---|---|
| **Pairwise** (LLM picks a winner) | Narrative Flow, Design Pairwise | Which method's slides are better? |
| **Scalar** (absolute score per method) | Quiz Eval, ROUGE, Perplexity, PPT Score, General Stats | How good are each method's slides independently? |

---

## Folder Structure

```
EvalPipeline/
├── pipeline.py                  # Main entry point — discovers papers, runs metrics, prints summary
├── constants.py                 # Single source of truth for all paths, models, and thresholds
├── run_eval.sh                  # Shell wrapper with logging (forwards CLI args to pipeline.py)
├── requirements.txt             # Python dependencies
│
├── data_prep/
│   ├── __init__.py
│   └── convert_to_text.py       # PDF/PPTX → markdown conversion (with GPT-4o vision fallback)
│
├── metrics/
│   ├── __init__.py
│   ├── narrative_flow.py        # Pairwise: narrative structure preservation (text-based)
│   ├── design_pairwise.py       # Pairwise: overall visual design quality (image-based)
│   ├── quiz_eval.py             # Scalar: quiz-based information coverage
│   ├── rouge_eval.py            # Scalar: ROUGE-L text overlap
│   ├── perplexity_eval.py       # Scalar: language fluency via causal LM
│   ├── ppt_score_eval.py        # Scalar: VLM-as-judge content / style / logic scores (1–5)
│   └── general_stats_eval.py    # Scalar: page count, character count, figure count
│
├── llm/
│   ├── __init__.py
│   └── client.py                # Unified LLM client (vLLM / OpenAI) with retry and vision support
│
├── utils/
│   ├── __init__.py
│   ├── result_utils.py          # JSON persistence, aggregation, console summaries, file helpers
│   └── image_utils.py           # Slide → PNG conversion, resizing, and sampling
│
├── prompts/                     # Prompt templates (Jinja2 / string-format)
│   ├── narrative_flow.txt
│   ├── design_pairwise.txt
│   ├── quiz_generate_simple.txt
│   ├── quiz_generate_detail.txt
│   ├── quiz_taker_text.txt
│   ├── quiz_taker_image.txt
│   ├── ppteval_describe_content.txt
│   ├── ppteval_content.txt
│   ├── ppteval_describe_style.txt
│   ├── ppteval_style.txt
│   ├── ppteval_coherence.txt
│   ├── ppteval_extract.txt
│   └── vision_extract_text.txt
│
├── Dataset/                     # (External) Input data
│   ├── generated_samples_final/ # One sub-folder per method, each with per-paper presentations
│   └── final_benchmark_data/    # Original papers (PDF) and cached quiz JSON files
│
└── results/                     # All output
    ├── narrative_flow.json
    ├── design_pairwise.json
    ├── quiz_eval.json
    ├── rouge_eval.json
    ├── perplexity_eval.json
    ├── ppt_score_eval.json
    ├── general_stats_eval.json
    ├── processed_data/          # Converted markdown text files
    └── slide_images/            # Cached slide PNG images
```

### Expected Input Data Layout

```
Dataset/
├── generated_samples_final/
│   ├── ours_rst-4o_newv2/           # Primary method
│   │   ├── {paper_name}/
│   │   │   └── final.pptx           # or final.pdf — any PPTX/PDF
│   │   └── ...
│   ├── paper2poster_orig-4o/        # Baseline 1
│   │   └── {paper_name}/...
│   ├── paper2slides/                # Baseline 2
│   ├── slidegen/                    # Baseline 3
│   ├── pptagent_template/           # Baseline 4
│   └── gt/                          # Ground truth presentations
│
└── final_benchmark_data/
    └── science/
        ├── pdf/
        │   └── {paper_name}/
        │       ├── original.pdf     # Source paper
        │       └── source.md        # (Optional) pre-converted markdown
        └── quiz/
            └── {paper_name}/
                ├── quiz_simple.json # (Generated on first run, then cached)
                └── quiz_detail.json
```

---

## Requirements

### System Dependencies

- **Python 3.10+**
- **LibreOffice** — for PPTX/PDF to PNG conversion (used by image-based metrics)
- **CUDA GPU** — recommended for perplexity evaluation and vLLM model serving

### Python Dependencies

```bash
pip install requests pymupdf4llm pymupdf Pillow jinja2 transformers torch torchvision scipy
pip install rouge-score evaluate python-pptx
pip install openai    # Only if using OpenAI backend or GPT-4o vision extraction
pip install vllm      # For serving open-source VL models locally
```

### LLM Server

The pipeline uses a single **vision-language model** for all LLM-based evaluation tasks (both text-only and image-based prompts).

**Option A — vLLM (recommended for open-source models):**

```bash
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-VL-7B-Instruct \
    --port 7001 \
    --dtype float16 \
    --trust-remote-code \
    --tensor-parallel-size 1   # increase for larger models
```

**Option B — OpenAI API:**

```bash
export OPENAI_API_KEY="sk-..."
# Then set LLM_BACKEND = "openai" in constants.py
```

---

## Configuration

All settings live in `constants.py`. Key ones to set before your first run:

| Setting | Default | What It Controls |
|---|---|---|
| `GENERATED_SAMPLES_DIR` | `Dataset/generated_samples_final` | Where generated presentations live |
| `BENCHMARK_DATA_DIR` | `Dataset/final_benchmark_data/science/pdf` | Where source papers live |
| `OURS_METHOD` | `"ours_rst-4o_newv3"` | The primary method being evaluated |
| `BASELINE_METHODS` | 5 baselines (see constants.py) | Which baselines to compare against |
| `LLM_BACKEND` | `"openai"` | `"vllm"` or `"openai"` |
| `MODEL` | `"gpt-5"` | Model for all evaluation tasks |
| `PPL_MODEL_PATH` | `"meta-llama/Meta-Llama-3-8B"` | Local causal LM for perplexity |
| `ENABLED_METRICS` | All 7 metrics | Which metrics to run by default |

See [Full Configuration Reference](#full-configuration-reference) at the bottom for the complete list.

---

## How to Run

```bash
# Run everything (data prep + all 7 metrics on all papers)
python pipeline.py

# Or use the shell wrapper (includes timestamped logging)
bash run_eval.sh

# Skip data preparation if markdown files already exist
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
python pipeline.py --skip-data-prep --metrics rouge_eval quiz_eval --num-papers 10
```

### CLI Arguments

| Argument | Description |
|---|---|
| `--metrics METRIC ...` | Specific metrics to run (default: all enabled) |
| `--papers PAPER ...` | Specific paper names to evaluate (default: auto-discover) |
| `--baselines METHOD ...` | Baseline methods (default: from constants.py) |
| `--backend {vllm,openai}` | Override LLM backend |
| `--skip-data-prep` | Skip PDF/PPTX to text conversion |
| `--force-conversion` | Re-convert even if files already exist |
| `--num-papers N` | Randomly sample N papers (`-1` = all) |
| `--model MODEL` | Override VL model (alias or full HuggingFace ID) |
| `--vllm-api-base URL` | Override vLLM server URL |

---

## Data Preparation

**Module:** `data_prep/convert_to_text.py`

Before any metric runs, the pipeline converts all presentation files and source papers into markdown text. This step can be skipped with `--skip-data-prep` if files already exist.

### How Conversion Works

**Source papers:**
1. If `source.md` already exists (pre-converted), uses it directly.
2. Otherwise, converts `original.pdf` to markdown via `pymupdf4llm`.
3. Strips supplementary material (appendices, supporting information) that appears after the References section. The References section itself is always preserved.

**Presentations (3-tier fallback chain):**

```
Tier 1: python-pptx / pymupdf    (free, fast, no API calls)
   │
   ▼ if all slides come back empty (text is embedded in images)
Tier 2: GPT-4o vision extraction  (requires OPENAI_API_KEY)
   │
   ▼ if GPT-4o is disabled or unavailable
Tier 3: slide_text.txt fallback   (pre-existing text files)
```

- **PPTX files:** Text extracted via `python-pptx`, iterating over all shapes and text frames per slide. Each slide formatted as `## Slide N\n<content>`.
- **PDF files:** Text extracted per page via `pymupdf`, each page treated as a slide.
- **GPT-4o fallback:** When standard extraction yields only empty slides (text baked into images), the pipeline converts slides to PNG and sends each to GPT-4o for OCR-quality text extraction.

### Conversion Output

| File | Path |
|---|---|
| Source paper text | `results/processed_data/{paper}/orig.md` |
| Method slide text | `results/processed_data/{paper}/{method}.md` |

---

## Metrics Overview

### Quick Reference

| # | Metric | Type | LLM Required? | Input | Measures |
|---|---|---|---|---|---|
| 1 | Narrative Flow | Pairwise | Yes (text) | Processed text | Narrative structure preservation |
| 2 | Design Pairwise | Pairwise | Yes (vision) | Slide images | Overall visual design quality |
| 3 | Quiz Eval | Scalar | Yes (text) | Paper + slide text | Information coverage |
| 4 | ROUGE Eval | Scalar | No | Processed text | Text overlap (ROUGE-L F1) |
| 5 | Perplexity Eval | Scalar | No (local LM: LLaMA-3-8B) | Processed text | Language fluency (lower = better) |
| 6 | PPT Score Eval | Scalar | Yes (text + vision) | Slide images + text | Content / Style / Logic (1–5) |
| 7 | General Stats Eval | Scalar | No | Extracted text + PPTX/PDF | Page, char, word, and figure counts |

---

### 1. Narrative Flow (Pairwise)

**Module:** `metrics/narrative_flow.py`
**Prompt:** `prompts/narrative_flow.txt`

Evaluates whether a presentation preserves the source paper's narrative structure — motivation, contributions, logical flow, and section self-sufficiency.

#### Input

| Data | Source | Notes |
|---|---|---|
| Reference text | `processed_data/{paper}/orig.md` | Source paper, truncated to 8 000 chars |
| OURS slide text | `processed_data/{paper}/{OURS_METHOD}.md` | Our method's slides as markdown |
| Baseline slide text | `processed_data/{paper}/{baseline}.md` | One baseline at a time |

#### How It Works

1. For each (paper, baseline) pair, the metric creates a **head-to-head comparison**.
2. **A/B randomization:** OURS and the baseline are randomly assigned to "Option A" and "Option B" (50/50, seeded with `SEED=42`) to prevent position bias.
3. The prompt template is filled with the reference text, Option A text, and Option B text.
4. The VL model is called via `call_text()` and asked to choose a winner.
5. The response is parsed for `"Answer: A"` or `"Answer: B"` using regex, then mapped back to "ours" or "baseline".
6. Wins are aggregated per baseline and across all baselines.

#### What the LLM Evaluates

- **Motivation clarity:** Does the presentation establish the problem and why it matters?
- **Contribution focus:** Are the key contributions clearly highlighted?
- **Logical causality:** Can the viewer follow "because of X, we did Y, which showed Z"?
- **Section self-sufficiency:** Are individual sections clear even if viewed out of order?

#### Output

**File:** `results/narrative_flow.json`

```json
{
  "metadata": { "metric": "narrative_flow", "model": "...", "timestamp": "..." },
  "overall_summary": {
    "total_comparisons": 50,
    "ours_total_wins": 35,
    "baseline_total_wins": 15,
    "ours_overall_win_rate": 0.70
  },
  "per_method_summary": {
    "slidegen": {
      "total_comparisons": 10,
      "ours_wins": 7,
      "baseline_wins": 3,
      "ours_win_rate": 0.70,
      "baseline_win_rate": 0.30
    }
  },
  "per_paper": {
    "paper_name": {
      "slidegen": {
        "assignment": { "A": "ours", "B": "baseline" },
        "winner": "ours",
        "reasoning": "Option A preserves the paper's logical flow...",
        "raw_response": "..."
      }
    }
  }
}
```

**How to read the scores:** `ours_win_rate` of 0.70 means our method was chosen as the better narrative in 70% of comparisons against that baseline.

---

### 2. Design Pairwise (Pairwise)

**Module:** `metrics/design_pairwise.py`
**Prompt:** `prompts/design_pairwise.txt`

Compares the **overall visual design quality** of two presentations using slide images in a single holistic evaluation. The prompt covers layout, readability, text density, aesthetics, and consistency in one pass.

#### Input

| Data | Source | Notes |
|---|---|---|
| OURS slide images | Converted from PPTX/PDF, cached in `results/slide_images/` | PNG files |
| Baseline slide images | Same conversion pipeline | PNG files |
| Prompt template | `prompts/design_pairwise.txt` | Jinja2 with `{{ method_1_count }}`, `{{ method_2_count }}` |

#### How It Works

1. For each (paper, baseline) pair, locate or convert slide images for both methods.
2. **Slide sampling:** Up to 4 slides are selected with even spacing across each deck.
3. **A/B randomization:** OURS and baseline are randomly assigned to Option A or Option B (50/50, seeded with `SEED=42`). Their images are concatenated: `[Option A slides..., Option B slides...]`.
4. **Image resizing:** All images are resized to 480px wide to reduce VLM token usage.
5. **VLM call:** The prompt and all images are sent via `call_vision()`. The model is asked to judge holistically — not slide by slide — considering alignment, overlaps, visual hierarchy, readability, font sizes, crowding, layout/style consistency, whitespace usage, and overall professionalism.
6. **Response parsing:** The model's response is parsed for `"Answer: A"` or `"Answer: B"`, then mapped back to "ours" or "baseline" based on the randomized assignment.
7. **Aggregation:** Win counts and rates per baseline.

#### What the VLM Evaluates (visual quality only)

- Element alignment and overlap-free layouts
- Clear visual hierarchy of titles, body, and figures
- Readable text with appropriate font sizes, no overcrowding
- Consistent layout and style across slides
- Effective whitespace usage
- Professional appearance with no rendering artifacts or clipped content

Content accuracy, writing quality, and narrative structure are explicitly excluded.

#### Output

**File:** `results/design_pairwise.json`

```json
{
  "metadata": { "metric": "design_pairwise", ... },
  "overall_summary": {
    "total_comparisons": 50,
    "ours_total_wins": 30,
    "baseline_total_wins": 20,
    "ours_overall_win_rate": 0.60
  },
  "per_method_summary": {
    "slidegen": {
      "total_comparisons": 10,
      "ours_wins": 6,
      "baseline_wins": 4,
      "ours_win_rate": 0.60,
      "baseline_win_rate": 0.40
    }
  },
  "per_paper": {
    "paper_name": {
      "slidegen": {
        "assignment": { "A": "ours", "B": "baseline" },
        "winner": "ours",
        "reasoning": "Option A has cleaner layouts with better spacing...",
        "raw_response": "..."
      }
    }
  }
}
```

**How to read the scores:** `ours_win_rate` of 0.60 means our method was judged as having better visual design in 60% of comparisons against that baseline.

---

### 3. Quiz Eval (Scalar)

**Module:** `metrics/quiz_eval.py`
**Prompts:** `prompts/quiz_generate_simple.txt`, `quiz_generate_detail.txt`, `quiz_taker_text.txt`

Measures **information coverage** by generating quiz questions from the source paper and testing whether the slide content contains enough information to answer them.

#### Input

| Data | Source | Notes |
|---|---|---|
| Source paper text | `source.md` or `processed_data/{paper}/orig.md` | For quiz generation |
| Slide text (per method) | `processed_data/{paper}/{method}.md` | For quiz answering |
| Cached quizzes | `quiz/{paper}/quiz_simple.json`, `quiz_detail.json` | Generated once, reused |

#### How It Works

**Phase 1 — Quiz Generation (per paper, done once, then cached):**
1. Loads the source paper text and removes the References/Acknowledgements sections.
2. Generates **50 "simple" questions** — high-level understanding: purpose, novelty, core approach. Multiple-choice (A/B/C/D).
3. Generates **50 "detail" questions** — specific factual details: numbers, method names, dataset names found verbatim in the paper. Same format.
4. Both quizzes are cached to `QUIZ_DATA_DIR/{paper}/`. On subsequent runs, generation is skipped.

**Phase 2 — Quiz Taking (per method, per paper):**
1. Loads the method's slide text.
2. Strips correct answers from the quiz before sending to the model.
3. Sends the slide text + stripped questions via `quiz_taker_text.txt`.
4. The model returns a JSON dict mapping question IDs to its chosen answers (A/B/C/D).
5. The score is computed by comparing model answers to ground truth.

#### Output

**File:** `results/quiz_eval.json`

```json
{
  "metadata": { "metric": "quiz_eval", "model": "...", ... },
  "per_method_summary": {
    "ours_rst-4o_newv2": {
      "mean_simple_pct": 0.72,
      "mean_detail_pct": 0.48,
      "papers_evaluated": 10
    }
  },
  "per_paper": {
    "paper_name": {
      "ours_rst-4o_newv2": {
        "simple_score": 36,
        "simple_total": 50,
        "detail_score": 24,
        "detail_total": 50,
        "simple_pct": 0.72,
        "detail_pct": 0.48
      }
    }
  }
}
```

**How to read the scores:**
- `simple_pct` = fraction of high-level questions answered correctly (0.0–1.0). Higher means the slides capture the paper's key messages well.
- `detail_pct` = fraction of factual detail questions answered correctly (0.0–1.0). Higher means the slides preserve fine-grained information.

---

### 4. ROUGE Eval (Scalar)

**Module:** `metrics/rouge_eval.py`

Measures **text overlap** between the slide content and the source paper using ROUGE-L (Longest Common Subsequence).

#### Input

| Data | Source |
|---|---|
| Reference text | `processed_data/{paper}/orig.md` |
| Hypothesis text (per method) | `processed_data/{paper}/{method}.md` |

#### How It Works

1. For each (method, paper) pair, computes **ROUGE-L F1** between the slide text (hypothesis) and the source paper (reference).
2. Uses the HuggingFace `evaluate` library with `rouge_types=["rougeL"]`.
3. **Fallback:** If the `evaluate` library is not installed, computes a manual word-level LCS ratio.
4. No LLM is involved — this is a pure text computation.

**ROUGE-L explained:**
- Finds the Longest Common Subsequence (LCS) of words between the two texts.
- Precision = LCS length / hypothesis length (how much of the slides appears in the paper).
- Recall = LCS length / reference length (how much of the paper appears in the slides).
- F1 = harmonic mean of precision and recall.

#### Output

**File:** `results/rouge_eval.json`

```json
{
  "metadata": { "metric": "rouge_eval", "model": "none (no LLM)", ... },
  "per_method_summary": {
    "ours_rst-4o_newv2": { "mean_rouge_l": 0.2345, "papers_evaluated": 10 }
  },
  "per_paper": {
    "paper_name": {
      "ours_rst-4o_newv2": { "rouge_l": 0.2345 }
    }
  }
}
```

**How to read the scores:** Range 0.0–1.0. Higher means more textual overlap between slides and source paper. Typical values for slide-from-paper tasks are 0.15–0.35.

---

### 5. Perplexity Eval (Scalar)

**Module:** `metrics/perplexity_eval.py`

Measures **language fluency** of slide text using a local causal language model. Lower perplexity means the text reads more naturally.

#### Input

| Data | Source |
|---|---|
| Slide text (per method) | `processed_data/{paper}/{method}.md` |
| Causal LM | `meta-llama/Meta-Llama-3-8B` (loaded locally, no server needed) |

#### How It Works

1. Loads the causal language model (`meta-llama/Meta-Llama-3-8B` by default) **once** and reuses it for all evaluations. Skips loading entirely if all results are already cached.
2. For each method's slide text:
   - Splits the text into segments by double newlines (`\n\n`).
   - For each segment (truncated to 512 tokens):
     - Tokenizes and computes cross-entropy loss with `model(**inputs, labels=input_ids)`.
     - Perplexity = `exp(loss)`.
   - The method's overall perplexity = mean across all segment perplexities.

#### Output

**File:** `results/perplexity_eval.json`

```json
{
  "metadata": { "metric": "perplexity_eval", "model": "meta-llama/Meta-Llama-3-8B", ... },
  "per_method_summary": {
    "ours_rst-4o_newv2": { "mean_ppl": 23.45, "papers_evaluated": 10 }
  },
  "per_paper": {
    "paper_name": {
      "ours_rst-4o_newv2": { "ppl": 23.45 }
    }
  }
}
```

**How to read the scores:** Lower is better. Typical ranges: 10–30 for natural English text, 50–100+ for fragmented or unnatural text.

---

### 6. PPT Score Eval (Scalar)

**Module:** `metrics/ppt_score_eval.py`
**Prompts:** `prompts/ppteval_describe_content.txt`, `ppteval_content.txt`, `ppteval_describe_style.txt`, `ppteval_style.txt`, `ppteval_coherence.txt`

Uses a VLM as a judge to score each presentation on **three dimensions**, each rated 1–5.

#### Input

| Data | Source | Notes |
|---|---|---|
| Slide images (per method) | Converted from PPTX/PDF, cached in `results/slide_images/` | For content and style scoring |
| Slide text (per method) | `processed_data/{paper}/{method}.md` | For logic/coherence scoring |

#### How It Works

**Content Score (per-slide, image-based):**
1. Samples up to 4 slides with even spacing, resizes to 480px.
2. For each slide:
   - **Describe:** Sends the image to the VLM → gets a textual description of information density and content quality.
   - **Score:** Sends the description to the LLM → gets a numeric score (1–5).
3. Content score = mean across sampled slides.

**Style Score (per-slide, image-based):**
1. Same sampling and resizing as content.
2. For each slide:
   - **Describe:** Sends the image to the VLM → gets a description of visual consistency, color scheme, and style.
   - **Score:** Sends the description to the LLM → gets a numeric score (1–5).
3. Style score = mean across sampled slides.

**Logic Score (presentation-level, text-based):**
1. Reads the entire slide text for the method.
2. Sends it to the LLM with the coherence prompt.
3. The model returns `{"score": N, "reason": "..."}` where N is 1–5.

**Scoring scale:**
| Score | Meaning |
|---|---|
| 1 | Poor — significant errors, poorly structured, hard to understand |
| 2 | Below average — lacks clear focus, awkward phrasing, weak organization |
| 3 | Average — clear and complete, but lacks visual aids or overall appeal |
| 4 | Good — clear and well-developed, minor weaknesses |
| 5 | Excellent — well-developed, clear focus, text and images complement effectively |

#### Output

**File:** `results/ppt_score_eval.json`

```json
{
  "metadata": { "metric": "ppt_score_eval", "model": "...", ... },
  "per_method_summary": {
    "ours_rst-4o_newv2": {
      "mean_content": 3.5,
      "mean_style": 4.0,
      "mean_logic": 3.8,
      "papers_evaluated": 10
    }
  },
  "per_paper": {
    "paper_name": {
      "ours_rst-4o_newv2": {
        "content": 3.5,
        "style": 4.0,
        "slides_scored": 4,
        "logic": 3.8,
        "logic_reason": "The presentation follows a clear structure..."
      }
    }
  }
}
```

**How to read the scores:** Each dimension is 1–5. Higher is better. `mean_content`, `mean_style`, and `mean_logic` are averages across all papers.

---

### 7. General Stats Eval (Scalar)

**Module:** `metrics/general_stats_eval.py`

Computes **structural statistics** for each presentation — no LLM needed. Character and word counts are derived from the pipeline's **extracted markdown text** (the same text all other metrics evaluate), while page count and figure count come from the raw PPTX/PDF files.

#### Input

| Data | Source | Used For |
|---|---|---|
| Extracted markdown text | `processed_data/{paper}/{method}.md` | Characters, words (and page count fallback) |
| Presentation files | `generated_samples_final/{method}/{paper}/*.pptx` or `*.pdf` | Page count, figure count |

#### How It Works

**Text statistics (from extracted markdown):**
- **Characters:** Total character count of the extracted markdown text.
- **Words:** Total word count of the extracted markdown text (whitespace-split).

**Structural statistics (from PPTX/PDF):**
- **Pages (PPTX):** Number of slides via `python-pptx`.
- **Pages (PDF):** Number of pages via `pymupdf`.
- **Figures (PPTX):** Count of shapes with `MSO_SHAPE_TYPE.PICTURE` or image placeholders.
- **Figures (PDF):** Count of images per page via `page.get_images(full=True)`.

**Fallback (no PPTX/PDF available):**
- Page count is inferred from `## Slide N` headers in the markdown text.
- Figure count defaults to 0.

#### Output

**File:** `results/general_stats_eval.json`

```json
{
  "metadata": { "metric": "general_stats_eval", "model": "none (no LLM)", ... },
  "per_method_summary": {
    "ours_rst-4o_newv2": {
      "mean_pages": 12.5,
      "mean_characters": 3500.0,
      "mean_words": 580.0,
      "mean_figures": 5.2,
      "papers_evaluated": 10
    }
  },
  "per_paper": {
    "paper_name": {
      "ours_rst-4o_newv2": {
        "pages": 12,
        "characters": 3500,
        "words": 580,
        "figures": 5
      }
    }
  }
}
```

**How to read the scores:** These are raw counts, not quality scores. Character and word counts reflect the extracted text that all metrics evaluate, making them consistent with the rest of the pipeline. Page and figure counts come from the original presentation files.

---

## Supporting Modules

### LLM Client (`llm/client.py`)

Provides two public functions used by all LLM-based metrics:

| Function | Description | Used By |
|---|---|---|
| `call_text(prompt, ...)` | Text-only LLM call | Narrative Flow, Quiz Eval, PPT Score (logic) |
| `call_vision(prompt, image_paths, ...)` | Multimodal (text + images) LLM call | Design Pairwise, PPT Score (content/style), GPT-4o extraction |

Both functions accept an optional `backend=` parameter to override the global `LLM_BACKEND` for that specific call. This is how GPT-4o text extraction always routes through OpenAI even when the pipeline uses vLLM for evaluation.

**Features:**
- Automatic retry with exponential backoff (up to `MAX_RETRIES` = 3).
- JSON response parsing: strips `<think>` blocks, markdown code fences, and extracts `{...}` objects.
- Base64 image encoding for vision calls.
- Server health check for vLLM (`check_server_health()`).

### Result Utils (`utils/result_utils.py`)

| Function | Description |
|---|---|
| `save_incremental(path, data)` | Atomic JSON write (write to `.tmp`, then `os.rename`) |
| `load_existing(path)` | Load previous results for resume support |
| `result_path(metric_name)` | Returns `results/{metric_name}.json` |
| `make_metadata(metric, model)` | Builds standard metadata block (timestamp, model, seed, etc.) |
| `aggregate_pairwise_wins(...)` | Counts wins/losses for pairwise metrics |
| `read_processed_text(paper, method)` | Reads `.md` (or `.txt` fallback) from processed_data |
| `print_pairwise_summary(...)` | Console output for pairwise metric results |
| `print_scalar_summary(...)` | Console output for scalar metric results |

### Image Utils (`utils/image_utils.py`)

| Function | Description |
|---|---|
| `slides_to_images(file, out_dir)` | Convert PPTX/PDF to PNG slides (LibreOffice + pymupdf) |
| `resize_images_tmp(paths, width)` | Non-destructive resize into temp directory |
| `sample_slides(paths, max_n)` | Evenly-spaced slide sampling (e.g., 4 from 20) |
| `find_and_convert_images(method, paper)` | Full lookup: cache → source dir → convert PPTX/PDF |

---

## Output Format

Every metric produces a JSON file with a consistent three-level structure:

```json
{
  "metadata": {
    "metric": "metric_name",
    "timestamp": "2025-01-01T00:00:00",
    "model": "Qwen/Qwen2.5-VL-7B-Instruct",
    "backend": "vllm",
    "ours_method": "ours_rst-4o_newv2",
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

---

## Resumability

The pipeline saves results incrementally using atomic JSON writes. If a run is interrupted:

- **Already-evaluated papers and methods are automatically skipped** on the next run.
- Partial results are saved after each paper completes.
- Use `--force-conversion` to re-process data files from scratch.
- Delete an individual metric JSON file from `results/` to re-run that metric.

---

## Full Configuration Reference

All values are in `constants.py`.

### Paths

| Constant | Default | Description |
|---|---|---|
| `GENERATED_SAMPLES_DIR` | `Dataset/generated_samples_final` | Generated presentations per method |
| `BENCHMARK_DATA_DIR` | `Dataset/final_benchmark_data/science/pdf` | Source paper PDFs |
| `QUIZ_DATA_DIR` | `Dataset/final_benchmark_data/science/quiz` | Cached quiz JSON files |
| `PROCESSED_DATA_DIR` | `results/processed_data` | Converted markdown text output |
| `RESULTS_DIR` | `results` | Metric result JSON output |
| `PROMPTS_DIR` | `prompts` | Prompt template files |
| `IMAGES_CACHE_DIR` | `results/slide_images` | Cached slide PNG images |
| `LIBREOFFICE_PATH` | `~/libreoffice/.../soffice` | LibreOffice binary for image conversion |

### API Keys and Backends

| Constant | Default | Description |
|---|---|---|
| `OPENAI_API_KEY` | `os.environ["OPENAI_API_KEY"]` | Required for GPT-4o vision extraction |
| `OPENAI_API_BASE` | `https://api.openai.com/v1` | OpenAI API base URL |
| `LLM_BACKEND` | `"openai"` | Backend for evaluation metrics |
| `VLLM_API_BASE` | `http://localhost:7001/v1` | vLLM server URL |
| `VLLM_HEALTH_URL` | `http://localhost:7001/health` | vLLM health check endpoint |

### Models

| Constant | Default | Description |
|---|---|---|
| `MODEL` | `"gpt-5"` | Model for all evaluation metrics |
| `PPL_MODEL_PATH` | `"meta-llama/Meta-Llama-3-8B"` | Causal LM for perplexity evaluation |
| `VISION_EXTRACTION_MODEL` | `"gpt-4o-2024-08-06"` | GPT-4o model for slide text extraction |
| `VISION_EXTRACTION_ENABLED` | `True` | Enable/disable GPT-4o text extraction fallback |

### Inference Parameters

| Constant | Default | Description |
|---|---|---|
| `TEMPERATURE` | `0.0` | Deterministic generation |
| `TOP_P` | `1.0` | Nucleus sampling threshold |
| `MAX_TOKENS` | `2048` | Max tokens per LLM response |
| `SEED` | `42` | Random seed for A/B position shuffling |
| `REQUEST_TIMEOUT` | `120` | HTTP timeout in seconds |

### Data Limits

| Constant | Default | Description |
|---|---|---|
| `MAX_REFERENCE_CHARS` | `8000` | Max chars of source paper text fed to LLM |
| `MAX_SLIDES_FOR_VISUAL` | `4` | Max slides sampled for image-based metrics |
| `IMAGE_TARGET_WIDTH` | `480` | Resize width (px) for slide images sent to VLM |

### Rate Limiting

| Constant | Default | Description |
|---|---|---|
| `SLEEP_BETWEEN_CALLS` | `1.0` | Seconds between consecutive API calls |
| `MAX_RETRIES` | `3` | Retry count on transient API errors |

### Pipeline Control

| Constant | Default | Description |
|---|---|---|
| `OURS_METHOD` | `"ours_rst-4o_newv3"` | Primary method being evaluated |
| `BASELINE_METHODS` | 5 baselines | Methods to compare against |
| `ENABLED_METRICS` | All 7 metrics | Which metrics to run by default |
| `SKIP_DATA_PREP` | `False` | Skip PDF/PPTX text conversion step |
| `NUM_PAPERS` | `-1` | Number of papers to evaluate (`-1` = all) |

### Supported Model Aliases

| Alias | Full Model ID | Notes |
|---|---|---|
| `qwen2.5-vl-7b` | `Qwen/Qwen2.5-VL-7B-Instruct` | Lightweight, single GPU |
| `qwen2.5-vl-32b` | `Qwen/Qwen2.5-VL-32B-Instruct` | Balanced performance |
| `qwen2.5-vl-72b` | `Qwen/Qwen2.5-VL-72B-Instruct` | Most capable, needs multi-GPU |
| `gpt-4o` | `gpt-4o-2024-08-06` | Requires OpenAI API key |
| `gpt-5` | `gpt-5` | Requires OpenAI API key |
