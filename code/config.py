"""
Central path and flag configuration for CSCA project.
All absolute paths and experiment-level flags go here.
Import this module instead of hardcoding paths or flags anywhere else.
"""
from pathlib import Path

# ---------------------------------------------------------------------------
# Project roots
# ---------------------------------------------------------------------------
MP2_ROOT     = Path(__file__).resolve().parent.parent
CODE_ROOT    = MP2_ROOT / "code"

# ---------------------------------------------------------------------------
# Model / data paths
# ---------------------------------------------------------------------------
MINIML_PATH     = MP2_ROOT / "all-MiniLM-L6-v2"
DEEPSC_PATH     = MP2_ROOT / "models" / "deepsc" / "text" / "best_model.pth"
QWEN_PATH       = MP2_ROOT / "models" / "Qwen.Qwen2-VL-7B.Q4_K_M.gguf"
DATA_PATH       = MP2_ROOT / "data"
CHECKPOINT_PATH = MP2_ROOT / "results" / "checkpoints"
CHECKPOINT_PATH.mkdir(parents=True, exist_ok=True)
DEEPSC_REPO     = MP2_ROOT / "repos" / "DeepSC"

# ---------------------------------------------------------------------------
# Training flags
# ---------------------------------------------------------------------------
# LAM_TRAINING_MODE — When True, HANMLPTrainer.train() calls the LAM pipeline
# (Ollama/Qwen2.5-VL) per episode to supply real parsed intents instead of
# the synthetic uniform draws from MultiCSCAEnvironment.generate_state().
# Requires Ollama running with qwen2.5vl:3b pulled.
# Default False: existing 5-CSCA audit-verified results are fully reproducible
# without Ollama.
LAM_TRAINING_MODE: bool = False

# USE_ENTROPY_REG — When True, the DDPMActor actor loss uses Eq. 33:
#   actor_loss = -Q.mean() - ENTROPY_COEFF * log_pi.mean()
# via DDPMActor.forward_with_logprob(). Default False: plain -Q.mean() loss,
# preserving committed ablation_han.csv and isr_vs_tpc.csv numbers.
USE_ENTROPY_REG: bool = False
ENTROPY_COEFF:   float = 0.01
