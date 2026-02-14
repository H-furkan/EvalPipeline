"""
llm/client.py — Unified LLM client.

Provides call_text() and call_vision() that route to either a local vLLM server
or the OpenAI API depending on constants.LLM_BACKEND.

Both functions default to the same VL model (constants.MODEL) which handles
both text-only and multimodal (text + images) prompts.

All other modules import only from here — no direct HTTP or OpenAI SDK calls elsewhere.
"""

import base64
import json
import time
from pathlib import Path
from typing import Optional, Union

import requests

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
import constants as C


# ── Helpers ───────────────────────────────────────────────────────────────────

def _encode_image(image_path: str) -> str:
    """Base64-encode a local image for the OpenAI vision API."""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def _image_ext(path: str) -> str:
    ext = Path(path).suffix.lower().lstrip(".")
    return "jpeg" if ext in ("jpg", "jpeg") else ext


def _build_openai_vision_content(prompt: str, image_paths: list[str]) -> list:
    """Build the content list for an OpenAI vision message."""
    content = [{"type": "text", "text": prompt}]
    for img_path in image_paths:
        b64 = _encode_image(img_path)
        media = f"image/{_image_ext(img_path)}"
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:{media};base64,{b64}"},
        })
    return content


def _parse_json_response(text: str) -> Optional[dict]:
    """Try to extract a JSON object from a model response string."""
    import re

    text = text.strip()

    # Strip <think>...</think> blocks (reasoning models)
    text = re.sub(r"<think>[\s\S]*?</think>", "", text).strip()

    # Strip markdown code fences anywhere in the text
    fence_match = re.search(r"```(?:json)?\s*\n([\s\S]*?)```", text)
    if fence_match:
        text = fence_match.group(1).strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Try to find the first {...} block (handles extra text around JSON)
        match = re.search(r"\{[\s\S]*\}", text)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
    return None


# ── vLLM backend ──────────────────────────────────────────────────────────────

def _call_vllm(
    prompt: str,
    model: str,
    image_paths: Optional[list[str]] = None,
    max_tokens: int = C.MAX_TOKENS,
    temperature: float = C.TEMPERATURE,
    top_p: float = C.TOP_P,
    return_json: bool = False,
) -> Optional[Union[str, dict]]:
    """POST to a running vLLM server's /chat/completions endpoint."""
    url = f"{C.VLLM_API_BASE}/chat/completions"

    if image_paths:
        # Multimodal message for vision models
        content = [{"type": "text", "text": prompt}]
        for img_path in image_paths:
            b64 = _encode_image(img_path)
            media = f"image/{_image_ext(img_path)}"
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:{media};base64,{b64}"},
            })
        messages = [{"role": "user", "content": content}]
    else:
        messages = [{"role": "user", "content": prompt}]

    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "n": 1,
    }
    if return_json:
        payload["response_format"] = {"type": "json_object"}

    for attempt in range(1, C.MAX_RETRIES + 1):
        try:
            resp = requests.post(url, json=payload, timeout=C.REQUEST_TIMEOUT)
            resp.raise_for_status()
            text = resp.json()["choices"][0]["message"]["content"].strip()
            if return_json:
                parsed = _parse_json_response(text)
                if parsed is None:
                    print(f"    [llm] Warning: could not parse JSON response: {text[:200]}")
                return parsed
            return text
        except requests.exceptions.RequestException as e:
            print(f"    [llm] vLLM API error (attempt {attempt}/{C.MAX_RETRIES}): {e}")
            if attempt < C.MAX_RETRIES:
                time.sleep(2 ** attempt)
    return None


# ── OpenAI backend ────────────────────────────────────────────────────────────

def _call_openai(
    prompt: str,
    model: str,
    image_paths: Optional[list[str]] = None,
    max_tokens: int = C.MAX_TOKENS,
    temperature: float = C.TEMPERATURE,
    top_p: float = C.TOP_P,
    return_json: bool = False,
) -> Optional[Union[str, dict]]:
    """Call OpenAI chat completions (with optional vision)."""
    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError("openai package is not installed. Run: pip install openai")

    api_key = C.OPENAI_API_KEY or None
    client = OpenAI(api_key=api_key, base_url=C.OPENAI_API_BASE)

    if image_paths:
        content = _build_openai_vision_content(prompt, image_paths)
        messages = [{"role": "user", "content": content}]
    else:
        messages = [{"role": "user", "content": prompt}]

    kwargs: dict = dict(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
    )
    if return_json:
        kwargs["response_format"] = {"type": "json_object"}

    for attempt in range(1, C.MAX_RETRIES + 1):
        try:
            resp = client.chat.completions.create(**kwargs)
            text = resp.choices[0].message.content.strip()
            if return_json:
                parsed = _parse_json_response(text)
                if parsed is None:
                    print(f"    [llm] Warning: could not parse JSON response: {text[:200]}")
                return parsed
            return text
        except Exception as e:
            print(f"    [llm] OpenAI API error (attempt {attempt}/{C.MAX_RETRIES}): {e}")
            if attempt < C.MAX_RETRIES:
                time.sleep(2 ** attempt)
    return None


# ── Public API ────────────────────────────────────────────────────────────────

def call_text(
    prompt: str,
    model: Optional[str] = None,
    max_tokens: int = C.MAX_TOKENS,
    temperature: float = C.TEMPERATURE,
    top_p: float = C.TOP_P,
    return_json: bool = False,
) -> Optional[Union[str, dict]]:
    """
    Send a text-only prompt to the configured LLM backend.

    Args:
        prompt:      The full prompt string.
        model:       Model identifier. Defaults to constants.MODEL (VL model).
        max_tokens:  Max tokens to generate.
        temperature: Sampling temperature.
        top_p:       Nucleus sampling threshold.
        return_json: If True, attempt to parse the response as JSON and return a dict.

    Returns:
        str | dict | None
    """
    model = model or C.MODEL
    if C.LLM_BACKEND == "openai":
        return _call_openai(prompt, model, max_tokens=max_tokens,
                            temperature=temperature, top_p=top_p,
                            return_json=return_json)
    return _call_vllm(prompt, model, max_tokens=max_tokens,
                      temperature=temperature, top_p=top_p,
                      return_json=return_json)


def call_vision(
    prompt: str,
    image_paths: list[str],
    model: Optional[str] = None,
    max_tokens: int = C.MAX_TOKENS,
    temperature: float = C.TEMPERATURE,
    top_p: float = C.TOP_P,
    return_json: bool = False,
) -> Optional[Union[str, dict]]:
    """
    Send a multimodal (text + images) prompt to the configured LLM backend.

    Args:
        prompt:      The text prompt.
        image_paths: List of local image file paths to attach.
        model:       Model identifier. Defaults to constants.MODEL (VL model).
        max_tokens:  Max tokens to generate.
        temperature: Sampling temperature.
        top_p:       Nucleus sampling threshold.
        return_json: If True, attempt to parse the response as JSON and return a dict.

    Returns:
        str | dict | None
    """
    model = model or C.MODEL
    if C.LLM_BACKEND == "openai":
        return _call_openai(prompt, model, image_paths=image_paths,
                            max_tokens=max_tokens, temperature=temperature,
                            top_p=top_p, return_json=return_json)
    return _call_vllm(prompt, model, image_paths=image_paths,
                      max_tokens=max_tokens, temperature=temperature,
                      top_p=top_p, return_json=return_json)


def check_server_health() -> bool:
    """
    Check whether the vLLM server is reachable (only meaningful when backend == 'vllm').
    Returns True if healthy, False otherwise.
    """
    if C.LLM_BACKEND != "vllm":
        return True
    try:
        resp = requests.get(C.VLLM_HEALTH_URL, timeout=5)
        return resp.status_code == 200
    except requests.exceptions.RequestException:
        return False
