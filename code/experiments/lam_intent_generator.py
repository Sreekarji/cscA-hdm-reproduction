"""
lam_intent_generator.py — Multimodal LAM intent parser for CSCA-SemCom.

Public API
----------
parse_intent_from_text(sentence, model)    -> (delay_s: float, quality: float)
parse_intent_from_image(image_path, model) -> (delay_s: float, quality: float)
parse_intent_from_audio(audio_path, ...)   -> (delay_s: float, quality: float)
normalize_intent(delay_s, quality)         -> (delay_s_clipped, quality_clipped)
lkb_retrieve(query, k)                     -> top-k LKB exemplars
cohere_rerank(query, docs, top_n)          -> reranked exemplars

Text parsing runs a self-adaptive RAG loop over the local knowledge base
(USE_ADAPTIVE_RAG); see the RAG section below.

Hardening
---------
- Pydantic v1/v2 compat: uses schema() on v1, model_json_schema() on v2
- Ollama version guard: checks format= support (requires >= 0.5)
- Falls back to regex parse if grammar-constrained output still misbehaves
- Audio: Whisper unloaded from VRAM before Ollama call
- RAG stages degrade to no-retrieval rather than failing
- All entry points return a safe neutral intent on any unhandled exception
"""

from __future__ import annotations

import json
import os
import re
import sys
from typing import Tuple

# code/ on the path so `from config import MINIML_PATH` resolves when this
# module is imported standalone (the experiment scripts insert it themselves).
_CODE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _CODE_DIR not in sys.path:
    sys.path.insert(0, _CODE_DIR)

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
# Self-adaptive RAG over the Local Knowledge Base (paper Sec IV-B)
#
# Retrieve -> rerank -> ISREL reflection -> generate -> ISSUP reflection.
# ISREL ("is relevant") gates whether the retrieved exemplars are on-topic; if
# not we reformulate the query and retry, up to _MAX_REFORMULATE times. ISSUP
# ("is supported") checks the produced intent against the exemplars it was
# conditioned on; if the exemplars do not support it, we fall back to the plain
# no-retrieval parse rather than shipping a hallucinated intent.
#
# Every stage degrades gracefully: no corpus -> no retrieval; no COHERE_API_KEY
# -> embedding-order rerank; no encoder -> lexical overlap scoring.
# ---------------------------------------------------------------------------
USE_ADAPTIVE_RAG = True

_MAX_REFORMULATE  = 2
_RETRIEVE_K       = 8
_RERANK_TOP_N     = 3
_ISREL_MIN_SCORE  = 0.30    # below this the retrieved set is judged off-topic
_ISSUP_MAX_DELTA  = 0.60    # |delay - nearest exemplar delay|, seconds

_lkb_cache = None
_lkb_embs  = None
_st_model  = None


def _lkb_path() -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(os.path.dirname(here))
    return os.path.join(root, "data", "lkb", "intent_corpus.json")


def _load_lkb() -> list:
    """Load the intent corpus once. Returns [] if the file is absent."""
    global _lkb_cache
    if _lkb_cache is not None:
        return _lkb_cache
    try:
        with open(_lkb_path(), "r", encoding="utf-8") as f:
            _lkb_cache = json.load(f).get("examples", [])
    except Exception:
        _lkb_cache = []
    return _lkb_cache


def _get_st_model():
    """MiniLM on CPU for LKB retrieval. None if sentence_transformers is absent."""
    global _st_model
    if _st_model is not None:
        return _st_model
    try:
        from sentence_transformers import SentenceTransformer
        from config import MINIML_PATH
        _st_model = SentenceTransformer(str(MINIML_PATH), device="cpu")
    except Exception:
        _st_model = None
    return _st_model


def _lexical_score(query: str, doc: str) -> float:
    q, d = set(query.lower().split()), set(doc.lower().split())
    return len(q & d) / max(len(q | d), 1)


def lkb_retrieve(query: str, k: int = _RETRIEVE_K) -> list:
    """Top-k LKB examples by MiniLM cosine, falling back to lexical overlap.

    Returns list of dicts, each the corpus entry plus a "score" field.
    """
    corpus = _load_lkb()
    if not corpus:
        return []
    sentences = [e["sentence"] for e in corpus]

    model = _get_st_model()
    if model is not None:
        global _lkb_embs
        try:
            import numpy as _np
            if _lkb_embs is None or len(_lkb_embs) != len(sentences):
                _lkb_embs = model.encode(sentences, convert_to_numpy=True,
                                         show_progress_bar=False)
                _lkb_embs = _lkb_embs / (
                    _np.linalg.norm(_lkb_embs, axis=1, keepdims=True) + 1e-8)
            q = model.encode([query], convert_to_numpy=True,
                             show_progress_bar=False)[0]
            q = q / (_np.linalg.norm(q) + 1e-8)
            scores = _lkb_embs @ q
        except Exception:
            scores = [_lexical_score(query, s) for s in sentences]
    else:
        scores = [_lexical_score(query, s) for s in sentences]

    ranked = sorted(zip(corpus, scores), key=lambda p: -float(p[1]))[:k]
    return [{**e, "score": float(s)} for e, s in ranked]


def cohere_rerank(query: str, docs: list, top_n: int = _RERANK_TOP_N) -> list:
    """Cross-encoder rerank via Cohere. Keyless / offline -> identity truncation.

    docs are the dicts returned by lkb_retrieve; the return preserves that shape
    with "score" overwritten by the rerank relevance when Cohere is used.
    """
    if not docs:
        return []
    api_key = os.environ.get("COHERE_API_KEY")
    if not api_key:
        return docs[:top_n]
    try:
        import cohere
        client = cohere.Client(api_key)
        resp = client.rerank(
            query=query,
            documents=[d["sentence"] for d in docs],
            top_n=min(top_n, len(docs)),
            model="rerank-english-v3.0",
        )
        out = []
        for r in resp.results:
            d = dict(docs[r.index])
            d["score"] = float(r.relevance_score)
            out.append(d)
        return out
    except Exception:
        return docs[:top_n]


def _check_isrel(query: str, docs: list) -> bool:
    """ISREL reflection token: are the retrieved exemplars relevant to query?"""
    if not docs:
        return False
    return max(float(d.get("score", 0.0)) for d in docs) >= _ISREL_MIN_SCORE


def _reformulate(query: str, attempt: int) -> str:
    """Widen the query when ISREL rejects the retrieved set."""
    if attempt == 1:
        return f"communication intent, delivery deadline and fidelity: {query}"
    return f"how urgent and how accurate must this transmission be: {query}"


def _check_issup(delay_s: float, quality: float, docs: list) -> bool:
    """ISSUP reflection token: is the generated intent supported by the exemplars?

    Supported means the parsed delay sits near at least one retrieved exemplar's
    delay. A value far outside the exemplar neighbourhood means the LAM ignored
    the retrieved evidence, so the caller should drop back to no-retrieval.
    """
    if not docs:
        return False
    if not (DELAY_MIN - 1e-6 <= delay_s <= DELAY_MAX + 1e-6):
        return False
    return any(abs(delay_s - float(d["delay_s"])) <= _ISSUP_MAX_DELTA
               for d in docs)


def _format_exemplars(docs: list) -> str:
    lines = [
        f'- "{d["sentence"]}" -> delay_seconds={d["delay_s"]}, '
        f'quality_score={d["quality"]}'
        for d in docs
    ]
    return (
        "Here are similar requests from the local knowledge base and the "
        "structured intents they map to:\n" + "\n".join(lines) + "\n\n"
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def _parse_intent_plain(sentence: str, model: str) -> Tuple[float, float]:
    """Single LAM call with no retrieval — the pre-RAG behaviour."""
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


def parse_intent_from_text(
    sentence: str,
    model: str = "qwen2.5vl:3b",
    use_rag: bool = None,
) -> Tuple[float, float]:
    if use_rag is None:
        use_rag = USE_ADAPTIVE_RAG
    if not use_rag:
        return _parse_intent_plain(sentence, model)

    query = sentence
    docs  = []
    for attempt in range(_MAX_REFORMULATE + 1):
        candidates = cohere_rerank(query, lkb_retrieve(query))
        if _check_isrel(query, candidates):
            docs = candidates
            break
        if attempt == _MAX_REFORMULATE:
            break
        query = _reformulate(sentence, attempt + 1)

    if not docs:
        return _parse_intent_plain(sentence, model)

    try:
        d, q = _ollama_structured(
            messages=[{
                "role": "user",
                "content": _format_exemplars(docs)
                           + _TEXT_PROMPT.format(sentence=sentence),
            }],
            model=model,
        )
    except Exception as exc:
        import warnings
        warnings.warn(f"RAG parse failed ({exc}); falling back to no-retrieval.")
        return _parse_intent_plain(sentence, model)

    if not _check_issup(d, q, docs):
        # ISSUP rejected: the exemplars do not back this intent.
        return _parse_intent_plain(sentence, model)
    return d, q


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
    sentence = sys.argv[1] if len(sys.argv) > 1 else "send it accurately within 1 second"
    print(f"Input : {sentence!r}")

    docs = cohere_rerank(sentence, lkb_retrieve(sentence))
    print(f"LKB   : {len(_load_lkb())} exemplars, {len(docs)} retained  "
          f"ISREL={_check_isrel(sentence, docs)}")
    for d in docs:
        print(f"        [{d['score']:.3f}] {d['sentence']!r} "
              f"-> ({d['delay_s']}, {d['quality']})")

    raw_d, raw_q = parse_intent_from_text(sentence)
    norm_d, norm_q = normalize_intent(raw_d, raw_q)
    print(f"Raw   : delay={raw_d:.3f} s  quality={raw_q:.3f}")
    print(f"Normed: delay={norm_d:.3f} s  quality={norm_q:.3f}")
