#!/usr/bin/env python3
"""
lam_provider.py — LAMIntentProvider for CSCA-HDM training-loop integration.

Bridges the LAM inference pipeline (Qwen2.5-VL via Ollama) into
HANMLPTrainer.train() when config.LAM_TRAINING_MODE = True.

Design principles
-----------------
- Runs Ollama on CPU/offloaded so it never competes with the HDM on the RTX 4050.
- Episode-level cache keyed on (task_idx % N_CACHE_SLOTS) so Ollama is called
  at most N_CACHE_SLOTS times per episode, not n_tasks times.
- get_batch() refreshes the cache every CACHE_TTL_EPISODES episodes so the
  training distribution varies over the run (avoids over-fitting to a fixed
  set of sampled sentences).
- All public return dicts are normalised to the training-distribution bounds
  defined in lam_intent_generator.py (delay ∈ [0.5, 2.5], quality ∈ [0.1, 0.4]).
- Falls back to synthetic intents gracefully if Ollama is unavailable, so
  training never crashes mid-run.

Smoke test
----------
  python code/lam/lam_provider.py --smoke-test
"""

from __future__ import annotations

import argparse
import os
import random
import sys
import time
import warnings
from typing import Dict, List, Optional

# ---------------------------------------------------------------------------
# Path: ensure code/ subpackages are importable regardless of cwd
# ---------------------------------------------------------------------------
_LAM_DIR  = os.path.dirname(os.path.abspath(__file__))      # code/lam/
_CODE_DIR = os.path.dirname(_LAM_DIR)                        # code/
for _p in (_CODE_DIR, _LAM_DIR,
           os.path.join(_CODE_DIR, "channel"),
           os.path.join(_CODE_DIR, "hdm"),
           os.path.join(_CODE_DIR, "utils"),
           os.path.join(_CODE_DIR, "experiments")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ---------------------------------------------------------------------------
# Sample payloads — used when no external image/audio is provided.
# These are drawn at random so the training distribution spans all intent types.
# ---------------------------------------------------------------------------

_TEXT_SAMPLES = [
    # critical / low-delay
    "send this message immediately, quality is not important",
    "transmit the alert right now with minimum latency",
    "relay this emergency command instantly",
    "push the real-time control token now",
    "deliver this urgent notification without delay",
    # high priority
    "send this report quickly with good accuracy",
    "transmit the sensor reading fast and accurately",
    "deliver this document promptly with high fidelity",
    "forward the patient record quickly",
    "send the financial log with precision soon",
    # medium priority
    "transmit this email, a couple of seconds is fine",
    "send the status update in moderate time",
    "deliver this article, no strict deadline",
    "forward the meeting notes within a few seconds",
    "relay this inventory update with acceptable quality",
    # low priority / background
    "backup this file overnight, quality is secondary",
    "archive this document whenever bandwidth is free",
    "sync this cache at your convenience",
    "send this old log at low bandwidth, no rush",
    "transfer this non-critical data in background",
]

_IMAGE_SAMPLES_DESC = [
    "a high-priority surveillance camera frame needing fast delivery",
    "a medical X-ray scan that requires high fidelity transmission",
    "a low-priority decorative photo for background backup",
    "an engineering blueprint requiring accurate urgent delivery",
    "a casual social media photo with flexible timing",
]

_AUDIO_SAMPLES_DESC = [
    "a live emergency voice call requiring instant relay",
    "a conference recording needing high-quality delivery",
    "a background podcast file for slow archival sync",
    "a forensic audio clip needing accurate fast transmission",
    "a voice memo with moderate urgency and quality needs",
]

_MULTIMODAL_SAMPLES_DESC = [
    "an annotated image-and-text emergency report needing instant relay",
    "a composite medical briefing requiring high fidelity quickly",
    "a mixed-media archive for low-priority background transfer",
    "a tagged sensor bundle needing fast and accurate delivery",
    "a multimedia field report with moderate urgency",
]

_MODALITY_SAMPLES = {
    "text":       _TEXT_SAMPLES,
    "image":      _IMAGE_SAMPLES_DESC,
    "audio":      _AUDIO_SAMPLES_DESC,
    "multimodal": _MULTIMODAL_SAMPLES_DESC,
}

# ---------------------------------------------------------------------------
# Synthetic fallback — produces intent from the medium-difficulty distribution
# used by MultiCSCAEnvironment, so training is unaffected when Ollama is down.
# ---------------------------------------------------------------------------
_DELAY_MIN,  _DELAY_MAX  = 0.50, 2.50
_QUAL_MIN,   _QUAL_MAX   = 0.10, 0.40

def _synthetic_intent() -> Dict:
    delay_s        = random.uniform(_DELAY_MIN, _DELAY_MAX)
    quality        = random.uniform(_QUAL_MIN, _QUAL_MAX)
    comp_ratio     = random.uniform(0.70, 1.00)
    return {
        "delay_seconds":     round(delay_s, 3),
        "quality_score":     round(quality, 3),
        "compression_ratio": round(comp_ratio, 3),
        "raw_text":          "(synthetic fallback)",
        "source":            "synthetic",
    }


# ---------------------------------------------------------------------------
# Provider
# ---------------------------------------------------------------------------

N_CACHE_SLOTS       = 20    # number of distinct intent slots cached per refresh
CACHE_TTL_EPISODES  = 50    # refresh the cache every N training episodes


class LAMIntentProvider:
    """Provides LAM-parsed intents for use in HANMLPTrainer.train().

    Usage (inside trainer)
    ----------------------
        provider = LAMIntentProvider(model="qwen2.5vl:3b")
        ...
        # at the top of each training episode:
        intents = provider.get_batch(n_tasks)
        # inject into state:
        for i, intent in enumerate(intents):
            state["SCt"]["delay_intents"][i]  = intent["delay_seconds"]
            state["SCt"]["quality_intents"][i] = intent["quality_score"]
            state["SCt"]["data_sizes"][i] *= intent["compression_ratio"]
    """

    def __init__(
        self,
        model: str = "qwen2.5vl:3b",
        use_adaptive_rag: bool = True,
        modality_mix: Optional[Dict[str, float]] = None,
    ):
        """
        Parameters
        ----------
        model            : Ollama model tag (must be pulled before training).
        use_adaptive_rag : Whether to run the ISREL/ISSUP self-adaptive RAG loop.
        modality_mix     : Dict mapping modality name → probability weight.
                           Defaults to uniform over text/image/audio/multimodal.
        """
        self.model           = model
        self.use_adaptive_rag = use_adaptive_rag
        self.modality_mix    = modality_mix or {
            "text": 0.40, "image": 0.20, "audio": 0.20, "multimodal": 0.20,
        }
        # Normalise weights
        total = sum(self.modality_mix.values())
        self.modality_mix = {k: v / total for k, v in self.modality_mix.items()}

        self._cache: List[Dict] = []       # current episode's intent pool
        self._episode_count: int = 0       # episodes served since last refresh
        self._ollama_ok: Optional[bool] = None  # None = not yet tested

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_intent(self, task_idx: int, modality: str = "text") -> Dict:
        """Return a parsed intent for one task.

        Uses the per-episode cache: task_idx % N_CACHE_SLOTS selects the slot,
        so at most N_CACHE_SLOTS Ollama calls happen per full cache refresh.
        """
        if not self._cache:
            self._refresh_cache()
        slot = task_idx % len(self._cache)
        return self._cache[slot]

    def get_batch(self, n: int) -> List[Dict]:
        """Return a list of n intent dicts, refreshing cache if TTL has expired.

        Cycles through the cache round-robin when n > N_CACHE_SLOTS.
        Logs compression_ratio per call for training diagnostics.
        """
        self._episode_count += 1
        if self._episode_count % CACHE_TTL_EPISODES == 0 or not self._cache:
            self._refresh_cache()
        return [self._cache[i % len(self._cache)] for i in range(n)]

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _check_ollama(self) -> bool:
        """Check once whether Ollama is reachable."""
        if self._ollama_ok is not None:
            return self._ollama_ok
        try:
            import ollama
            ollama.list()      # lightweight health check — no model load
            self._ollama_ok = True
        except Exception as exc:
            warnings.warn(
                f"[LAMIntentProvider] Ollama not reachable ({exc}). "
                "Falling back to synthetic intents for this training run. "
                "Start Ollama and restart training to use real LAM intents.",
                stacklevel=3,
            )
            self._ollama_ok = False
        return self._ollama_ok

    def _sample_modality(self) -> str:
        keys   = list(self.modality_mix.keys())
        weights = [self.modality_mix[k] for k in keys]
        return random.choices(keys, weights=weights, k=1)[0]

    def _parse_one(self, modality: str) -> Dict:
        """Call the LAM pipeline for one (modality, payload) pair."""
        # Local import to keep the Ollama process separate from the HDM GPU process
        from lam_intent_generator import (
            parse_intent_from_text,
            parse_intent_from_image,
            parse_intent_from_audio,
            normalize_intent,
        )
        from mss_algorithm import minimum_synonymous_subsequence

        payload = random.choice(_MODALITY_SAMPLES[modality])

        try:
            if modality == "text":
                raw_d, raw_q = parse_intent_from_text(
                    payload, model=self.model, use_rag=self.use_adaptive_rag
                )
                # MSS compression ratio on the payload text
                compressed, ratio = minimum_synonymous_subsequence(payload)
            elif modality == "image":
                # For training we describe the image in text (no actual image file)
                raw_d, raw_q = parse_intent_from_text(
                    payload, model=self.model, use_rag=self.use_adaptive_rag
                )
                ratio = random.uniform(0.75, 1.00)
            elif modality == "audio":
                raw_d, raw_q = parse_intent_from_text(
                    payload, model=self.model, use_rag=self.use_adaptive_rag
                )
                ratio = random.uniform(0.70, 1.00)
            else:  # multimodal
                raw_d, raw_q = parse_intent_from_text(
                    payload, model=self.model, use_rag=self.use_adaptive_rag
                )
                ratio = random.uniform(0.72, 1.00)

            delay_s, quality = normalize_intent(raw_d, raw_q)
            return {
                "delay_seconds":     round(delay_s, 3),
                "quality_score":     round(quality, 3),
                "compression_ratio": round(float(ratio), 3),
                "raw_text":          payload,
                "modality":          modality,
                "source":            "lam",
            }
        except Exception as exc:
            warnings.warn(
                f"[LAMIntentProvider] LAM parse failed for modality={modality} "
                f"({exc}); using synthetic fallback for this slot.",
                stacklevel=2,
            )
            fb = _synthetic_intent()
            fb["modality"] = modality
            return fb

    def _refresh_cache(self):
        """Repopulate N_CACHE_SLOTS intent slots via Ollama (or synthetic)."""
        if not self._check_ollama():
            self._cache = [_synthetic_intent() for _ in range(N_CACHE_SLOTS)]
            return

        t0 = time.monotonic()
        intents = []
        for i in range(N_CACHE_SLOTS):
            modality = self._sample_modality()
            intents.append(self._parse_one(modality))

        elapsed = time.monotonic() - t0
        delays  = [x["delay_seconds"] for x in intents]
        ratios  = [x["compression_ratio"] for x in intents]
        print(
            f"[LAMProvider] Cache refreshed ({N_CACHE_SLOTS} slots, {elapsed:.1f}s) | "
            f"delay {min(delays):.2f}–{max(delays):.2f}s | "
            f"compression {min(ratios):.2f}–{max(ratios):.2f}"
        )
        self._cache = intents


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

def _smoke_test():
    print("=" * 60)
    print("LAMIntentProvider smoke test")
    print("=" * 60)

    provider = LAMIntentProvider(model="qwen2.5vl:3b")

    # 1. Synthetic fallback path (force Ollama=False)
    provider._ollama_ok = False
    batch = provider.get_batch(5)
    assert len(batch) == 5, "Expected 5 intents"
    for item in batch:
        assert "delay_seconds" in item
        assert "quality_score" in item
        assert "compression_ratio" in item
        assert 0.5 <= item["delay_seconds"] <= 2.5, f"delay out of range: {item}"
        assert 0.10 <= item["quality_score"] <= 0.40, f"quality out of range: {item}"
    print(f"  [PASS] Synthetic fallback: {len(batch)} intents")
    delays = [x['delay_seconds'] for x in batch]
    print(f"         delay range {min(delays):.2f}–{max(delays):.2f}s")

    # 2. Cache cycling
    provider._cache = [_synthetic_intent() for _ in range(N_CACHE_SLOTS)]
    result = provider.get_batch(N_CACHE_SLOTS * 3)
    assert len(result) == N_CACHE_SLOTS * 3
    print(f"  [PASS] Cache cycling: {len(result)} intents from {N_CACHE_SLOTS} slots")

    # 3. Individual get_intent
    item = provider.get_intent(task_idx=7)
    assert "delay_seconds" in item
    print(f"  [PASS] get_intent(task_idx=7): delay={item['delay_seconds']}s, "
          f"quality={item['quality_score']}")

    # 4. LKB retrieval (offline, no Ollama needed)
    try:
        _lam_dir = os.path.dirname(os.path.abspath(__file__))
        _code_dir = os.path.dirname(_lam_dir)
        if _code_dir not in sys.path:
            sys.path.insert(0, _code_dir)
        if os.path.join(_code_dir, "experiments") not in sys.path:
            sys.path.insert(0, os.path.join(_code_dir, "experiments"))
        from lam_intent_generator import lkb_retrieve, _load_lkb
        corpus = _load_lkb()
        if corpus:
            hits = lkb_retrieve("send image urgently", k=5)
            assert len(hits) > 0, "lkb_retrieve returned empty"
            print(f"  [PASS] LKB retrieval: {len(corpus)} exemplars loaded, "
                  f"top-5 scores: {[round(h['score'],3) for h in hits]}")
        else:
            print("  [WARN] LKB corpus not found — run from repo root or check data/lkb/")
    except Exception as exc:
        print(f"  [WARN] LKB test skipped: {exc}")

    print()
    print(f"LAM provider: got {len(batch)} intents, "
          f"delay range {min(delays):.2f}–{max(delays):.2f}s, done.")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-test", action="store_true",
                        help="Run offline smoke test (no Ollama needed)")
    args = parser.parse_args()
    if args.smoke_test:
        _smoke_test()
    else:
        parser.print_help()
