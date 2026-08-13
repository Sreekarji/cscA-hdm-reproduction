"""
Tests for Gap 3: DDPMActor.forward_with_logprob and USE_ENTROPY_REG flag.
All CPU-only, no Ollama needed.
"""
import math
import pytest
import torch


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def actor():
    from ddpm_policy import DDPMActor
    return DDPMActor(
        graph_emb_dim=256, task_emb_dim=256,
        action_dim=80,          # n_tasks=20, n_mcs=3, no relay: 20 + 20*3 = 80
        n_tasks=20, n_relays=5, n_mcs=3,
        n_denoising_steps=6,
        use_relay=False,
    )


@pytest.fixture(scope="module")
def dummy_inputs():
    ge = torch.randn(1, 256)
    me = torch.randn(20, 256)
    return ge, me


# ---------------------------------------------------------------------------
# forward_with_logprob: existence and signature
# ---------------------------------------------------------------------------

class TestForwardWithLogprob:
    def test_method_exists(self, actor):
        assert hasattr(actor, "forward_with_logprob"), (
            "DDPMActor is missing forward_with_logprob method"
        )
        assert callable(actor.forward_with_logprob)

    def test_return_is_tuple_of_two(self, actor, dummy_inputs):
        ge, me = dummy_inputs
        result = actor.forward_with_logprob(ge, me)
        assert isinstance(result, tuple) and len(result) == 2, (
            f"Expected (action, log_pi) tuple, got {type(result)}"
        )

    def test_action_shape(self, actor, dummy_inputs):
        ge, me = dummy_inputs
        action, _ = actor.forward_with_logprob(ge, me)
        assert action.shape == (1, 80), f"action shape {action.shape} != (1, 80)"

    def test_log_pi_is_scalar(self, actor, dummy_inputs):
        ge, me = dummy_inputs
        _, log_pi = actor.forward_with_logprob(ge, me)
        assert log_pi.numel() == 1, f"log_pi has {log_pi.numel()} elements (expected 1)"

    def test_log_pi_is_finite(self, actor, dummy_inputs):
        ge, me = dummy_inputs
        _, log_pi = actor.forward_with_logprob(ge, me)
        assert torch.isfinite(log_pi), f"log_pi is not finite: {log_pi.item()}"

    def test_log_pi_is_negative(self, actor, dummy_inputs):
        """Gaussian log-prob is negative for non-degenerate distributions."""
        ge, me = dummy_inputs
        _, log_pi = actor.forward_with_logprob(ge, me)
        assert log_pi.item() < 0, (
            f"log_pi={log_pi.item():.4f} should be negative for Gaussian"
        )

    def test_bw_sums_to_one(self, actor, dummy_inputs):
        ge, me = dummy_inputs
        action, _ = actor.forward_with_logprob(ge, me)
        bw_sum = action[0, :20].sum().item()
        assert abs(bw_sum - 1.0) < 0.01, f"BW doesn't sum to 1: {bw_sum:.4f}"

    def test_mcs_in_zero_one(self, actor, dummy_inputs):
        ge, me = dummy_inputs
        action, _ = actor.forward_with_logprob(ge, me)
        mcs = action[0, 20:]
        assert mcs.min().item() >= 0.0 and mcs.max().item() <= 1.0, (
            f"MCS out of [0,1]: min={mcs.min():.4f} max={mcs.max():.4f}"
        )

    def test_gradient_flows_through_both(self, actor, dummy_inputs):
        """Gradients must flow through action AND log_pi back to denoiser."""
        ge, me = dummy_inputs
        # Fresh actor to avoid accumulated grads
        from ddpm_policy import DDPMActor
        a = DDPMActor(action_dim=80, n_tasks=20, n_relays=5, n_mcs=3,
                      n_denoising_steps=6, use_relay=False)
        a.train()
        ge2 = torch.randn(1, 256)
        me2 = torch.randn(20, 256)
        action, log_pi = a.forward_with_logprob(ge2, me2)
        loss = -action.sum() - 0.01 * log_pi
        loss.backward()
        gnorm = sum(
            p.grad.norm().item()
            for p in a.denoiser.parameters()
            if p.grad is not None
        )
        assert gnorm > 0, "No gradient flowed to denoiser from forward_with_logprob"

    def test_requires_message_embs(self, actor):
        ge = torch.randn(1, 256)
        with pytest.raises((ValueError, TypeError)):
            actor.forward_with_logprob(ge, None)

    def test_consistent_action_layout_with_forward(self, actor, dummy_inputs):
        """forward() and forward_with_logprob() must produce the same action layout."""
        ge, me = dummy_inputs
        torch.manual_seed(0)
        action_fwd = actor.forward(ge, message_embs=me, deterministic=True)
        torch.manual_seed(0)
        action_fwlp, _ = actor.forward_with_logprob(ge, me)
        # Shapes must match; values will differ (different RNG state) — check shape only
        assert action_fwd.shape == action_fwlp.shape


# ---------------------------------------------------------------------------
# USE_ENTROPY_REG flag wiring
# ---------------------------------------------------------------------------

class TestEntropyRegFlag:
    def test_flag_exists_in_config(self):
        import config
        assert hasattr(config, "USE_ENTROPY_REG"), "USE_ENTROPY_REG missing from config"
        assert hasattr(config, "ENTROPY_COEFF"),   "ENTROPY_COEFF missing from config"

    def test_flag_default_false(self):
        import config
        assert config.USE_ENTROPY_REG is False, (
            "USE_ENTROPY_REG must default to False to preserve reproducibility"
        )

    def test_entropy_coeff_positive(self):
        import config
        assert config.ENTROPY_COEFF > 0, "ENTROPY_COEFF must be positive"

    def test_lam_training_mode_default_false(self):
        import config
        assert config.LAM_TRAINING_MODE is False, (
            "LAM_TRAINING_MODE must default to False"
        )


# ---------------------------------------------------------------------------
# Noise schedule sanity
# ---------------------------------------------------------------------------

class TestNoiseSchedule:
    def test_alpha_bar_N_small(self, actor):
        """alpha_bar_N < 0.1 ensures the chain reaches near-pure noise."""
        alpha_bar_N = float(actor.alphas_cumprod[-1])
        assert alpha_bar_N < 0.1, (
            f"alpha_bar_N={alpha_bar_N:.4f} >= 0.1; reverse chain may not converge"
        )

    def test_beta_tilde_positive(self, actor):
        assert (actor.beta_tilde > 0).all(), "beta_tilde must be positive"

    def test_betas_positive(self, actor):
        assert (actor.betas > 0).all() and (actor.betas < 1).all()
