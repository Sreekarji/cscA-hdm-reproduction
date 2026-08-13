"""
Tests for Gap 1: LKB corpus and lkb_retrieve / cohere_rerank.
All offline — no Ollama, no Cohere API key needed.
"""
import json
import os
import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _corpus_path():
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(os.path.dirname(here))
    return os.path.join(root, "data", "lkb", "intent_corpus.json")


# ---------------------------------------------------------------------------
# Corpus schema tests
# ---------------------------------------------------------------------------

class TestLKBCorpus:
    @pytest.fixture(scope="class")
    def corpus(self):
        path = _corpus_path()
        assert os.path.exists(path), f"LKB corpus not found at {path}"
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data

    def test_top_level_keys(self, corpus):
        assert "version"     in corpus
        assert "n_examples"  in corpus
        assert "examples"    in corpus

    def test_example_count(self, corpus):
        assert len(corpus["examples"]) == 200, (
            f"Expected 200 examples, got {len(corpus['examples'])}"
        )

    def test_n_examples_matches(self, corpus):
        assert corpus["n_examples"] == len(corpus["examples"])

    def test_required_fields(self, corpus):
        required = {"id", "sentence", "delay_s", "quality", "modality", "priority"}
        for i, ex in enumerate(corpus["examples"]):
            missing = required - set(ex.keys())
            assert not missing, f"Example {i} missing fields: {missing}"

    def test_delay_range(self, corpus):
        for ex in corpus["examples"]:
            assert 0.1 <= ex["delay_s"] <= 2.0, (
                f"{ex['id']}: delay_s={ex['delay_s']} out of [0.1, 2.0]"
            )

    def test_quality_range(self, corpus):
        for ex in corpus["examples"]:
            assert 0.0 <= ex["quality"] <= 1.0, (
                f"{ex['id']}: quality={ex['quality']} out of [0.0, 1.0]"
            )

    def test_modality_values(self, corpus):
        valid = {"text", "image", "audio", "multimodal"}
        for ex in corpus["examples"]:
            assert ex["modality"] in valid, (
                f"{ex['id']}: unknown modality {ex['modality']!r}"
            )

    def test_priority_values(self, corpus):
        valid = {"low", "medium", "high", "critical"}
        for ex in corpus["examples"]:
            assert ex["priority"] in valid, (
                f"{ex['id']}: unknown priority {ex['priority']!r}"
            )

    def test_modality_distribution(self, corpus):
        from collections import Counter
        counts = Counter(ex["modality"] for ex in corpus["examples"])
        for mod in ("text", "image", "audio", "multimodal"):
            assert counts[mod] == 50, (
                f"Expected 50 {mod} examples, got {counts[mod]}"
            )

    def test_unique_sentences(self, corpus):
        sentences = [ex["sentence"] for ex in corpus["examples"]]
        assert len(set(sentences)) == len(sentences), "Duplicate sentences found"

    def test_unique_ids(self, corpus):
        ids = [ex["id"] for ex in corpus["examples"]]
        assert len(set(ids)) == len(ids), "Duplicate IDs found"


# ---------------------------------------------------------------------------
# lkb_retrieve tests (no Ollama, no Cohere)
# ---------------------------------------------------------------------------

class TestLKBRetrieve:
    def test_retrieve_returns_list(self):
        # Force reload by clearing cache
        import lam_intent_generator as lig
        lig._lkb_cache = None
        results = lig.lkb_retrieve("send image urgently", k=5)
        assert isinstance(results, list)

    def test_retrieve_non_empty(self):
        import lam_intent_generator as lig
        lig._lkb_cache = None
        results = lig.lkb_retrieve("send image urgently", k=5)
        assert len(results) > 0, "lkb_retrieve returned empty — check corpus path"

    def test_retrieve_count(self):
        import lam_intent_generator as lig
        lig._lkb_cache = None
        results = lig.lkb_retrieve("transmit voice call instantly", k=3)
        assert len(results) <= 3

    def test_retrieve_has_score(self):
        import lam_intent_generator as lig
        lig._lkb_cache = None
        results = lig.lkb_retrieve("backup file overnight", k=5)
        for r in results:
            assert "score" in r, f"Result missing 'score': {r}"
            assert isinstance(r["score"], float)

    def test_retrieve_has_delay_quality(self):
        import lam_intent_generator as lig
        lig._lkb_cache = None
        results = lig.lkb_retrieve("emergency alert", k=5)
        for r in results:
            assert "delay_s"  in r, f"Result missing 'delay_s': {r}"
            assert "quality"  in r, f"Result missing 'quality': {r}"
            assert "sentence" in r, f"Result missing 'sentence': {r}"

    def test_retrieve_sorted_by_score(self):
        import lam_intent_generator as lig
        lig._lkb_cache = None
        results = lig.lkb_retrieve("urgent transmission needed", k=8)
        scores = [r["score"] for r in results]
        assert scores == sorted(scores, reverse=True), "Results not sorted by score"


# ---------------------------------------------------------------------------
# cohere_rerank local fallback test (no API key)
# ---------------------------------------------------------------------------

class TestLocalReranker:
    def test_rerank_no_key(self, monkeypatch):
        """Without COHERE_API_KEY the local MiniLM reranker should run."""
        monkeypatch.delenv("COHERE_API_KEY", raising=False)
        import lam_intent_generator as lig
        lig._lkb_cache = None
        docs = lig.lkb_retrieve("send data immediately", k=8)
        if not docs:
            pytest.skip("LKB empty — skipping reranker test")
        reranked = lig.cohere_rerank("send data immediately", docs, top_n=3)
        assert isinstance(reranked, list)
        assert len(reranked) <= 3
        for r in reranked:
            assert "score" in r

    def test_rerank_empty_docs(self):
        import lam_intent_generator as lig
        result = lig.cohere_rerank("any query", [], top_n=3)
        assert result == []
