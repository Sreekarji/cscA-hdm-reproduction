"""
Tests for Gap 2: LAMIntentProvider (offline — no Ollama needed).
Forces synthetic fallback path via _ollama_ok = False.
"""
import pytest
import sys
import os


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def provider():
    from lam_provider import LAMIntentProvider
    p = LAMIntentProvider(model="qwen2.5vl:3b")
    p._ollama_ok = False   # force synthetic fallback — no Ollama needed
    return p


# ---------------------------------------------------------------------------
# get_intent
# ---------------------------------------------------------------------------

class TestGetIntent:
    def test_returns_dict(self, provider):
        item = provider.get_intent(task_idx=0)
        assert isinstance(item, dict)

    def test_required_keys(self, provider):
        item = provider.get_intent(task_idx=3)
        for key in ("delay_seconds", "quality_score", "compression_ratio", "raw_text"):
            assert key in item, f"Missing key: {key}"

    def test_delay_in_range(self, provider):
        for i in range(10):
            item = provider.get_intent(task_idx=i)
            assert 0.5 <= item["delay_seconds"] <= 2.5, (
                f"delay_seconds={item['delay_seconds']} out of [0.5, 2.5]"
            )

    def test_quality_in_range(self, provider):
        for i in range(10):
            item = provider.get_intent(task_idx=i)
            assert 0.10 <= item["quality_score"] <= 0.40, (
                f"quality_score={item['quality_score']} out of [0.1, 0.4]"
            )

    def test_compression_ratio_in_range(self, provider):
        for i in range(10):
            item = provider.get_intent(task_idx=i)
            assert 0.0 < item["compression_ratio"] <= 1.0, (
                f"compression_ratio={item['compression_ratio']} out of (0, 1]"
            )

    def test_cache_slot_cycling(self, provider):
        """task_idx % N_CACHE_SLOTS should always return a cached item."""
        from lam_provider import N_CACHE_SLOTS
        _ = provider.get_batch(1)   # ensure cache populated
        item_a = provider.get_intent(task_idx=0)
        item_b = provider.get_intent(task_idx=N_CACHE_SLOTS)
        # Both index slot 0 — same cached entry
        assert item_a["delay_seconds"] == item_b["delay_seconds"]


# ---------------------------------------------------------------------------
# get_batch
# ---------------------------------------------------------------------------

class TestGetBatch:
    def test_returns_list(self, provider):
        batch = provider.get_batch(5)
        assert isinstance(batch, list)

    def test_length(self, provider):
        for n in (1, 5, 20, 23):
            batch = provider.get_batch(n)
            assert len(batch) == n, f"Expected {n} items, got {len(batch)}"

    def test_all_items_valid(self, provider):
        batch = provider.get_batch(20)
        for item in batch:
            assert 0.5  <= item["delay_seconds"]     <= 2.5
            assert 0.10 <= item["quality_score"]     <= 0.40
            assert 0.0  <  item["compression_ratio"] <= 1.0

    def test_cache_ttl_refresh(self, provider):
        """After CACHE_TTL_EPISODES calls get_batch triggers a refresh."""
        from lam_provider import CACHE_TTL_EPISODES
        provider._episode_count = 0
        provider._cache = []
        # First call should populate cache
        b1 = provider.get_batch(5)
        assert len(provider._cache) > 0

        # Simulate TTL expiry
        provider._episode_count = CACHE_TTL_EPISODES - 1
        b2 = provider.get_batch(5)   # this call increments to TTL, triggers refresh
        assert len(b2) == 5


# ---------------------------------------------------------------------------
# Inject-into-state interface test
# ---------------------------------------------------------------------------

class TestStateInjection:
    def test_inject_overwrites_correct_keys(self, provider):
        """Simulate what _inject_lam_intents does to a state dict."""
        import sys, os
        # Build a minimal state dict like MultiCSCAEnvironment.generate_state()
        n_tasks = 5
        state = {
            "Rt": {},
            "SCt": {
                "delay_intents":    [1.5] * n_tasks,
                "quality_intents":  [0.25] * n_tasks,
                "data_sizes":       [3e5] * n_tasks,
                "message_features": [[0.5, 0.5, 0.5, 0.5]] * n_tasks,
                "semantic_types":   [0] * n_tasks,
            }
        }

        batch = provider.get_batch(n_tasks)
        for i, intent in enumerate(batch):
            state["SCt"]["delay_intents"][i]  = intent["delay_seconds"]
            state["SCt"]["quality_intents"][i] = intent["quality_score"]
            state["SCt"]["data_sizes"][i]     *= intent["compression_ratio"]

        for i in range(n_tasks):
            assert 0.5 <= state["SCt"]["delay_intents"][i] <= 2.5
            assert 0.10 <= state["SCt"]["quality_intents"][i] <= 0.40
            # data_size should have shrunk (compression_ratio <= 1)
            assert state["SCt"]["data_sizes"][i] <= 3e5 + 1e-6


# ---------------------------------------------------------------------------
# Smoke-test runner (matches the CLI --smoke-test path)
# ---------------------------------------------------------------------------

class TestSmokeTest:
    def test_smoke_test_passes(self):
        """Run the built-in smoke test — must not raise."""
        from lam_provider import _smoke_test
        _smoke_test()   # prints output; will raise AssertionError on failure
