"""
constants.py — Single source of truth for all configurable values.

Edit this file to change any path, model, threshold, or pipeline behavior.
Everything in pipeline.py and every metric imports from here.
"""

import os

# ── Paths ─────────────────────────────────────────────────────────────────────
_BASE = os.path.dirname(os.path.abspath(__file__))
_DATASET = os.path.join(_BASE, "Dataset")

# Root directory that holds generated presentation outputs (one sub-folder per method)
GENERATED_SAMPLES_DIR = os.path.join(_DATASET, "generated_samples_final")

# Root directory for original papers (one sub-folder per paper, each containing original.pdf)
BENCHMARK_DATA_DIR = os.path.join(_DATASET, "final_benchmark_data/science/pdf")

# Where quiz JSON files are cached (quiz_simple.json / quiz_detail.json per paper)
QUIZ_DATA_DIR = os.path.join(_DATASET, "final_benchmark_data/science/quiz")

# Where converted text files are written (processed_data/{paper}/{method}.md)
PROCESSED_DATA_DIR = os.path.join(_BASE, "results/processed_data")

# Where all metric result JSON files are saved
RESULTS_DIR = os.path.join(_BASE, "results")

# Where prompt templates live (narrative_flow.txt, content_*.txt, etc.)
PROMPTS_DIR = os.path.join(_BASE, "prompts")

# Where slide-to-image conversions are cached (per method/paper)
IMAGES_CACHE_DIR = os.path.join(_BASE, "results/slide_images")

# Path to LibreOffice binary used for PPTX/PDF → PNG conversion
LIBREOFFICE_PATH = os.path.expanduser("~/libreoffice/opt/libreoffice25.8/program/soffice")

# ── Methods ───────────────────────────────────────────────────────────────────
# The "our" method being evaluated
OURS_METHOD = "ours_rst-4o_newv3"

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
LLM_BACKEND = "openai"

# vLLM server configuration (used when LLM_BACKEND == "vllm")
VLLM_API_BASE = "http://localhost:7001/v1"
VLLM_HEALTH_URL = "http://localhost:7001/health"

# ── OpenAI API ───────────────────────────────────────────────────────────────
# Required for: GPT-4o vision text extraction during data prep,
#               and when LLM_BACKEND == "openai" for evaluation metrics.
# Set via environment variable (recommended) or hard-code below.
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
OPENAI_API_BASE = "https://api.openai.com/v1"

# ── Model ─────────────────────────────────────────────────────────────────────
# Single vision-language model used for ALL evaluation tasks (text and image).
MODEL = "gpt-5"

# Local model path for perplexity evaluation (loaded via transformers, no server needed)
PPL_MODEL_PATH = "meta-llama/Meta-Llama-3-8B"

# ── GPT-4o Vision Text Extraction ────────────────────────────────────────────
# Some slides contain text embedded in images that python-pptx / pymupdf cannot
# extract. When standard extraction yields empty slides, the pipeline converts
# them to images and sends each slide to GPT-4o vision to extract the text.
# This always uses the OpenAI API regardless of the LLM_BACKEND setting.
VISION_EXTRACTION_ENABLED = True
VISION_EXTRACTION_MODEL = "gpt-4o-2024-08-06"

# ── Supported Models ─────────────────────────────────────────────────────────
# Pre-defined model configurations for easy switching.
# Each entry maps a short alias to the full model identifier.
# Use --model CLI arg (or edit MODEL above) to switch.
#
# To serve a model with vLLM:
#   python -m vllm.entrypoints.openai.api_server \
#       --model <MODEL_ID> --port 7001 --dtype float16 \
#       --trust-remote-code --tensor-parallel-size <N_GPUS>

SUPPORTED_MODELS = {
    "qwen2.5-vl-7b":  "Qwen/Qwen2.5-VL-7B-Instruct",
    "qwen2.5-vl-32b": "Qwen/Qwen2.5-VL-32B-Instruct",
    "qwen2.5-vl-72b": "Qwen/Qwen2.5-VL-72B-Instruct",
    "gpt-4o":         "gpt-4o-2024-08-06",
    "gpt-5":          "gpt-5",
}

# ── Inference Parameters ──────────────────────────────────────────────────────
TEMPERATURE = 0.0          # Deterministic generation
TOP_P = 1.0
MAX_TOKENS = 2048          # Max tokens per LLM response (needs room for reasoning)
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
    "design_pairwise",    # Pairwise: overall visual design quality — image-based
    "quiz_eval",          # Generate Q&A from paper, answer from slide text → information coverage
    "rouge_eval",         # ROUGE-L: text overlap between slides and source paper
    "perplexity_eval",    # Perplexity (PPL): language fluency via causal LM
    "ppt_score_eval",     # VLM-as-judge: per-slide content / style / logic scores (1-5)
    "general_stats_eval", # General stats: page count, character count, figure count per presentation
]

# Skip data preparation step (PDF/PPTX → text conversion).
# Set True if processed_data/ already exists from a previous run.
SKIP_DATA_PREP = False

# Number of papers to evaluate. -1 evaluates all discovered papers.
# Positive integer randomly samples that many papers (same set for all metrics).
NUM_PAPERS = -1
