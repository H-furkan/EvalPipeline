"""
constants.py — Single source of truth for all configurable values.

Edit this file to change any path, model, threshold, or pipeline behavior.
Everything in pipeline.py and every metric imports from here.
"""

import os

# ── Paths ─────────────────────────────────────────────────────────────────────
# Root directory that holds generated presentation outputs (one sub-folder per method)
GENERATED_SAMPLES_DIR = "/home/furkan/Eval/Evaluation/eval_data/generated_samples_final"

# Root directory for original papers (one sub-folder per paper, each containing original.pdf)
BENCHMARK_DATA_DIR = "/home/furkan/Eval/Evaluation/eval_data/final_benchmark_data/science/pdf"

# Where quiz JSON files are cached (quiz_simple.json / quiz_detail.json per paper)
QUIZ_DATA_DIR = "/home/furkan/Eval/Evaluation/eval_data/final_benchmark_data/science/quiz"

# Where converted text files are written (processed_data/{paper}/{method}.txt)
PROCESSED_DATA_DIR = "/home/furkan/Eval/EvalPipeline/results/processed_data"

# Where all metric result JSON files are saved
RESULTS_DIR = "/home/furkan/Eval/EvalPipeline/results"

# Where prompt templates live (narrative_flow.txt, content_*.txt, etc.)
PROMPTS_DIR = "/home/furkan/Eval/EvalPipeline/prompts"

# Where PPTX-to-image conversions are cached (per method/paper)
IMAGES_CACHE_DIR = "/home/furkan/Eval/EvalPipeline/results/slide_images"

# Path to LibreOffice binary used for PPTX → PNG conversion
LIBREOFFICE_PATH = os.path.expanduser("~/libreoffice/opt/libreoffice25.8/program/soffice")

# ── Methods ───────────────────────────────────────────────────────────────────
# The "our" method being evaluated
OURS_METHOD = "ours_rst-4o"

# Baseline methods to compare against
BASELINE_METHODS = [
    "paper2poster_orig-4o",
    "paper2slides",
    "slidegen",
    "pptagent_template",
    "gt",
]

# ── LLM Backend ───────────────────────────────────────────────────────────────
# "vllm"   → Connect to a local vLLM HTTP server (recommended for Qwen models)
# "openai" → Use OpenAI API directly (requires OPENAI_API_KEY env var or below)
LLM_BACKEND = "vllm"

# vLLM server configuration (used when LLM_BACKEND == "vllm")
VLLM_API_BASE = "http://localhost:7001/v1"
VLLM_HEALTH_URL = "http://localhost:7001/health"

# OpenAI API configuration (used when LLM_BACKEND == "openai")
# Leave empty to read from the OPENAI_API_KEY environment variable
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
OPENAI_API_BASE = "https://api.openai.com/v1"

# ── Models ────────────────────────────────────────────────────────────────────
# Text-only LLM used for narrative flow, content, coherence metrics
TEXT_MODEL = "Qwen/Qwen2.5-32B-Instruct"

# Vision-language model used for design (image-based) and ppt_score metrics
VISION_MODEL = "Qwen/Qwen2.5-VL-32B-Instruct"

# Model for quiz generation and quiz-taking (when backend == "openai")
QUIZ_MODEL = "gpt-4o-2024-08-06"

# Local model path for perplexity evaluation (loaded via transformers)
PPL_MODEL_PATH = "meta-llama/Meta-Llama-3-8B"

# ── Inference Parameters ──────────────────────────────────────────────────────
TEMPERATURE = 0.0          # Deterministic generation
TOP_P = 1.0
MAX_TOKENS = 512           # Max tokens per LLM response
SEED = 42                  # Random seed for A/B position shuffling
REQUEST_TIMEOUT = 120      # HTTP timeout in seconds for API calls

# ── Data Limits ───────────────────────────────────────────────────────────────
# Max characters of the reference (source paper) text fed to the LLM.
# Prevents exceeding context limits. Set 0 to disable truncation.
MAX_REFERENCE_CHARS = 8000

# Max number of slides sampled per paper for design (image-based) evaluation
MAX_SLIDES_FOR_VISUAL = 4

# Resize slide images to this width (px) before sending to VLM (reduces tokens)
IMAGE_TARGET_WIDTH = 480

# ── Rate Limiting ─────────────────────────────────────────────────────────────
# Seconds to sleep between consecutive API calls (set 0 to disable)
SLEEP_BETWEEN_CALLS = 1.0

# Max retries on transient API errors (with exponential back-off)
MAX_RETRIES = 3

# ── Pipeline Control ──────────────────────────────────────────────────────────
# List of metrics to run. Comment out any entry to skip that metric.
ENABLED_METRICS = [
    "narrative_flow",     # Pairwise: does presentation preserve paper's narrative structure?
    "content_pairwise",   # Pairwise: 4 content attributes (text quality, structure, org, conciseness)
    "design_pairwise",    # Pairwise: 5 design attributes (layout, readability, …) — image-based
    "coherence_pairwise", # Pairwise: 3 coherence attributes (logical flow, consistency, completeness)
    "quiz_eval",          # Generate Q&A from paper, answer from slide text → information coverage
    "rouge_eval",         # ROUGE-L: text overlap between slides and source paper
    "perplexity_eval",    # Perplexity (PPL): language fluency via Llama-3-8B
    "ppt_score_eval",     # VLM-as-judge: per-slide content / style / logic scores (1-5)
]

# Skip data preparation step (PDF/PPTX → text conversion).
# Set True if processed_data/ already exists from a previous run.
SKIP_DATA_PREP = False

# Number of papers to evaluate. -1 evaluates all discovered papers.
# Positive integer randomly samples that many papers (same set for all metrics).
NUM_PAPERS = -1
