"""
lam_intent_generator.py — Multimodal LAM intent parser for CSCA-SemCom.

Public API
----------
parse_intent_from_text(sentence, model)    -> (delay_s: float, quality: float)
parse_intent_from_image(image_path, model) -> (delay_s: float, quality: float)
parse_intent_from_audio(audio_path, ...)   -> (delay_s: float, quality: float)
normalize_intent(delay_s, quality)         -> (delay_s_clipped, quality_clipped)

Hardening
---------
- Pydantic v1/v2 compat: uses schema() on v1, model_json_schema() on v2
- Ollama version guard: checks format= support (requires >= 0.5)
- Falls back to regex parse if grammar-constrained output still misbehaves
- Audio: Whisper unloaded from VRAM before Ollama call
- All entry points return a safe neutral intent on any unhandled exception
"""

from __future__ import annotations

import json
import os
import re
from typing import Tuple

# ---------------------------------------------------------------------------
# Training-distribution bounds (medium difficulty, sim_channel.py L2103-2104)
# ---------------------------------------------------------------------------
DELAY_MIN = 0.50
DELAY_MAX = 2.50
QUAL_MIN  = 0.10
QUAL_MAX  = 0.40

_NEUTRAL = ((DELAY_MIN + DELAY_MAX) / 2, (QUAL_MIN + QUAL_MAX) / 2)  # (1.50, 0.25)

# ---------------------------------------------------------------------------
# Pydantic v1/v2 compatibility
# ---------------------------------------------------------------------------
try:
    from pydantic import BaseModel, Field
    import pydantic as _pydantic

    _PYDANTIC_V2 = int(_pydantic.VERSION.split(".")[0]) >= 2

    class _Intent(BaseModel):
        delay_seconds: float = Field(
            description="Seconds within which the task must complete"
        )
        quality_score: float = Field(
            description="Desired fidelity 0.0-1.0 (1.0 = perfect)"
        )

    def _intent_schema() -> dict:
        if _PYDANTIC_V2:
            return _Intent.model_json_schema()
        return _Intent.schema()

    def _parse_intent_json(text: str) -> _Intent:
        if _PYDANTIC_V2:
            return _Intent.model_validate_json(text)
        return _Intent.parse_raw(text)

except ImportError as _e:
    raise ImportError("pydantic is required: pip install pydantic>=1.10") from _e

# ---------------------------------------------------------------------------
# Ollama version guard
# ---------------------------------------------------------------------------
def _check_ollama_version():
    try:
        import ollama as _ol
        ver = getattr(_ol, "__version__", "0.0.0")
        major, minor = int(ver.split(".")[0]), int(ver.split(".")[1])
        if (major, minor) < (0, 5):
            import warnings
            warnings.warn(
                f"ollama=={ver} detected. format= structured output requires >= 0.5. "
                "Upgrade with: pip install -U ollama",
                stacklevel=3,
            )
    except Exception:
        pass

# ---------------------------------------------------------------------------
# Regex fallback
# ---------------------------------------------------------------------------
def _regex_fallback(text: str) -> Tuple[float, float]:
    nums = re.findall(r"[-+]?\d*\.?\d+", text)
    if len(nums) >= 2:
        return float(nums[0]), float(nums[1])
    return _NEUTRAL

# ---------------------------------------------------------------------------
# Prompt template
# ---------------------------------------------------------------------------
_TEXT_PROMPT = (
    "This sentence may contain the user's intent regarding delay and quality. "
    "Please convert it into structured values: "
    "delay_seconds is the time (in seconds) within which the user expects the task "
    "to be completed; quality_score is the desired communication quality expressed "
    "as data similarity (0.0 = no fidelity, 1.0 = perfect fidelity).\n\n"
    'Sentence: "{sentence}"'
)

_IMAGE_PROMPT = (
    "Look at the image. Based on its content, infer what communication intent "
    "a user transmitting this image likely has: "
    "delay_seconds (how urgently they need it delivered, in seconds) and "
    "quality_score (how much fidelity they need, 0.0-1.0). "
    "If the image is a high-detail document or medical scan, quality should be high "
    "(near 1.0) and delay moderate. If it is a casual photo, quality can be lower."
)

# ---------------------------------------------------------------------------
# Core Ollama call
# ---------------------------------------------------------------------------
def _ollama_structured(messages: list, model: str) -> Tuple[float, float]:
    import ollama
    _check_ollama_version()

    resp = ollama.chat(
        model=model,
        messages=messages,
        format=_intent_schema(),
        options={"temperature": 0},
    )
    raw = resp["message"]["content"]

    try:
        intent = _parse_intent_json(raw)
        return intent.delay_seconds, intent.quality_score
    except Exception:
        return _regex_fallback(raw)

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def parse_intent_from_text(
    sentence: str,
    model: str = "qwen2.5vl:3b",
) -> Tuple[float, float]:
    try:
        return _ollama_structured(
            messages=[{"role": "user",
                       "content": _TEXT_PROMPT.format(sentence=sentence)}],
            model=model,
        )
    except Exception as exc:
        import warnings
        warnings.warn(f"parse_intent_from_text failed ({exc}); returning neutral intent.")
        return _NEUTRAL


def parse_intent_from_image(
    image_path: str,
    model: str = "qwen2.5vl:3b",
) -> Tuple[float, float]:
    if not os.path.exists(image_path):
        import warnings
        warnings.warn(f"Image not found: {image_path}; returning neutral intent.")
        return _NEUTRAL
    try:
        return _ollama_structured(
            messages=[{
                "role": "user",
                "content": _IMAGE_PROMPT,
                "images": [image_path],
            }],
            model=model,
        )
    except Exception as exc:
        import warnings
        warnings.warn(f"parse_intent_from_image failed ({exc}); returning neutral intent.")
        return _NEUTRAL


def parse_intent_from_audio(
    audio_path: str,
    whisper_dir: str | None = None,
    whisper_model: str = "base",
    lam_model: str = "qwen2.5vl:3b",
) -> Tuple[float, float]:
    if not os.path.exists(audio_path):
        import warnings
        warnings.warn(f"Audio file not found: {audio_path}; returning neutral intent.")
        return _NEUTRAL

    try:
        import whisper
    except ImportError:
        import warnings
        warnings.warn("openai-whisper not installed; returning neutral.")
        return _NEUTRAL

    try:
        load_kwargs = {}
        if whisper_dir:
            load_kwargs["download_root"] = whisper_dir
        wm = whisper.load_model(whisper_model, **load_kwargs)
        result = wm.transcribe(audio_path)
        transcript = result["text"].strip()
        del wm

        if not transcript:
            return _NEUTRAL

        return parse_intent_from_text(transcript, model=lam_model)

    except Exception as exc:
        import warnings
        warnings.warn(f"parse_intent_from_audio failed ({exc}); returning neutral intent.")
        return _NEUTRAL


def normalize_intent(
    delay_s: float,
    quality: float,
) -> Tuple[float, float]:
    return (
        float(max(DELAY_MIN, min(DELAY_MAX, delay_s))),
        float(max(QUAL_MIN,  min(QUAL_MAX,  quality))),
    )

# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    sentence = sys.argv[1] if len(sys.argv) > 1 else "send it accurately within 1 second"
    print(f"Input : {sentence!r}")
    raw_d, raw_q = parse_intent_from_text(sentence)
    norm_d, norm_q = normalize_intent(raw_d, raw_q)
    print(f"Raw   : delay={raw_d:.3f} s  quality={raw_q:.3f}")
    print(f"Normed: delay={norm_d:.3f} s  quality={norm_q:.3f}")
