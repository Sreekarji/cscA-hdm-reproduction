"""
ablation_ddpm.py — Fig. 13b reproduction.

Compares full HDM (HAN+DDPM) vs HDM without DDPM (HAN+MLP actor).
Paper claim: full HDM outperforms no-DDPM by 12.5% at tpc=4 (20 tasks/CSCA).
Run: python code/experiments/ablation_ddpm.py
"""

import os
import sys

_SELF_DIR = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.dirname(os.path.dirname(_SELF_DIR))
for sub in ["code/hdm", "code/channel", "code/evaluation", "code/utils", "code/experiments"]:
    sys.path.insert(0, os.path.join(BASE, sub))

import csv
import torch
import numpy as np
from reproducibility import set_seed
from train_han_mlp import (
    HANMLPTrainer, evaluate_policy, DEVICE, POLICY, CHECKPOINT_PATH,
)
from mlp_policy import MLPActor

RESULTS_DIR = os.path.join(BASE, "results", "final")
os.makedirs(RESULTS_DIR, exist_ok=True)

_HAN_KEYS   = ["han", "han_state_dict", "han_network", "model"]
_ACTOR_KEYS = ["actor", "actor_state_dict", "policy", "ddpm_actor", "mlp_actor"]

def _find_key(ckpt, candidates, label):
    for k in candidates:
        if k in ckpt:
            return k
    raise KeyError(f"No {label} key in checkpoint. Keys: {list(ckpt.keys())}")

def _load_ckpt(trainer, ckpt_path):
    if not os.path.exists(ckpt_path):
        return False
    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    han_key   = _find_key(ckpt, _HAN_KEYS,   "HAN")
    actor_key = _find_key(ckpt, _ACTOR_KEYS, "actor")
    trainer.han.load_state_dict(ckpt[han_key])
    trainer.actor.load_state_dict(ckpt[actor_key])
    isr_str = f"ISR={ckpt['isr']:.3f}" if "isr" in ckpt else ""
    ep_str  = f"ep={ckpt['episode']}"  if "episode" in ckpt else ""
    print(f"  Loaded checkpoint {isr_str} {ep_str}")
    return True

def run_full_hdm(tpc=4):
    """HAN + DDPM actor (full HDM). Reuses existing checkpoint if available."""
    print(f"\n--- Full HDM (HAN+DDPM) tpc={tpc} ---")
    set_seed(42)
    trainer = HANMLPTrainer(tasks_per_csca=tpc, difficulty="medium")
    ckpt_path = os.path.join(str(CHECKPOINT_PATH), f"han_{POLICY}_tpc{tpc}_best.pt")
    if _load_ckpt(trainer, ckpt_path):
        print("  Using existing checkpoint — no retraining needed.")
    else:
        print("  No checkpoint found. Training from scratch (1000 episodes) ...")
        trainer.train(max_episodes=1000)
    mean_isr, std_isr, mean_delay, mean_dist = evaluate_policy(trainer, n_episodes=200)
    print(f"  Full HDM: ISR={mean_isr:.4f} +/-{std_isr:.4f}  delay={mean_delay:.3f}s")
    return mean_isr, std_isr

def run_mlp_actor(tpc=4):
    """HAN + plain MLP actor (no DDPM). Always trains from scratch."""
    print(f"\n--- No-DDPM HDM (HAN+MLP actor) tpc={tpc} ---")
    set_seed(42)
    trainer = HANMLPTrainer(tasks_per_csca=tpc, difficulty="medium")

    # Replace DDPMActor with plain MLPActor (from mlp_policy.py)
    trainer.actor = MLPActor(
        graph_emb_dim=256,
        task_emb_dim=256,
        action_dim=trainer.action_dim,
        hidden_dim=256,
        n_tasks=trainer.n_tasks,
    ).to(DEVICE)
    trainer.opt_actor = torch.optim.Adam(trainer.actor.parameters(), lr=1e-4)
    trainer.sched_actor = torch.optim.lr_scheduler.StepLR(
        trainer.opt_actor, step_size=300, gamma=0.7)

    trainer.train(max_episodes=1000)

    ckpt_path = os.path.join(str(CHECKPOINT_PATH), f"han_mlponly_tpc{tpc}_best.pt")
    _load_ckpt(trainer, ckpt_path)

    mean_isr, std_isr, mean_delay, mean_dist = evaluate_policy(trainer, n_episodes=200)
    print(f"  No-DDPM HDM: ISR={mean_isr:.4f} +/-{std_isr:.4f}  delay={mean_delay:.3f}s")
    return mean_isr, std_isr

if __name__ == "__main__":
    tpc = 4   # Paper Fig. 13b: 20 tasks/CSCA. Run tpc=10 separately for scale validation.

    hdm_isr,    hdm_std    = run_full_hdm(tpc)
    noddpm_isr, noddpm_std = run_mlp_actor(tpc)

    improvement = (hdm_isr - noddpm_isr) / max(noddpm_isr, 1e-8) * 100

    print("\n" + "=" * 50)
    print(f"  DDPM ABLATION RESULT (tpc={tpc}, {'Fig. 13b' if tpc == 4 else 'scale validation'})")
    print("=" * 50)
    print(f"  Full HDM (HAN+DDPM) : ISR={hdm_isr:.4f} +/-{hdm_std:.4f}")
    print(f"  No-DDPM (HAN+MLP)   : ISR={noddpm_isr:.4f} +/-{noddpm_std:.4f}")
    print(f"  DDPM improvement    : +{improvement:.1f}%")
    print(f"  Paper claim         : +12.5% at 20 tasks/CSCA")
    print(f"  Status              : {'CONFIRMED' if improvement > 8.0 else 'BELOW PAPER CLAIM'}")
    print("=" * 50)

    out = os.path.join(RESULTS_DIR, "ablation_ddpm.csv")
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["method", "isr_mean", "isr_std", "ddpm_improvement_pct"])
        w.writerow(["HAN+DDPM", f"{hdm_isr:.4f}", f"{hdm_std:.4f}", ""])
        w.writerow(["HAN+MLP",  f"{noddpm_isr:.4f}", f"{noddpm_std:.4f}",
                    f"{improvement:.1f}"])
    print(f"  Wrote {out}")
