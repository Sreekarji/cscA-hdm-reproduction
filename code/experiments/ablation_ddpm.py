"""
ablation_ddpm.py — Fig. 13b reproduction + tpc=10 scale validation.

Compares full HDM (HAN+DDPM) vs HDM without DDPM (HAN+MLP actor).
Paper claim: full HDM outperforms no-DDPM by ~12.5% at tpc=4.
Audit fix: now runs at BOTH tpc=4 AND tpc=10 and writes a tpc column to CSV.

Run: python code/experiments/ablation_ddpm.py
Output: results/final/ablation_ddpm.csv  (columns: tpc, method, isr_mean, isr_std,
                                           delay_mean, ddpm_improvement_pct)
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


def run_full_hdm(tpc):
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
    print(f"  Full HDM: ISR={mean_isr:.4f} ±{std_isr:.4f}  delay={mean_delay:.3f}s")
    return mean_isr, std_isr, mean_delay


def run_mlp_actor(tpc):
    """HAN + plain MLP actor (no DDPM). Always trains from scratch."""
    print(f"\n--- No-DDPM HDM (HAN+MLP actor) tpc={tpc} ---")
    set_seed(42)
    trainer = HANMLPTrainer(tasks_per_csca=tpc, difficulty="medium")

    # Replace DDPMActor with plain MLPActor
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
    print(f"  No-DDPM HDM: ISR={mean_isr:.4f} ±{std_isr:.4f}  delay={mean_delay:.3f}s")
    return mean_isr, std_isr, mean_delay


def run_one_tpc(tpc):
    """Run full ablation for a single tpc value. Returns row dicts for CSV."""
    hdm_isr,    hdm_std,    hdm_delay    = run_full_hdm(tpc)
    noddpm_isr, noddpm_std, noddpm_delay = run_mlp_actor(tpc)

    improvement = (hdm_isr - noddpm_isr) / max(noddpm_isr, 1e-8) * 100

    label = "Fig. 13b" if tpc == 4 else "scale validation"
    print("\n" + "=" * 55)
    print(f"  DDPM ABLATION (tpc={tpc}, {label})")
    print("=" * 55)
    print(f"  Full HDM (HAN+DDPM) : ISR={hdm_isr:.4f} ±{hdm_std:.4f}  delay={hdm_delay:.3f}s")
    print(f"  No-DDPM (HAN+MLP)   : ISR={noddpm_isr:.4f} ±{noddpm_std:.4f}  delay={noddpm_delay:.3f}s")
    print(f"  DDPM improvement    : +{improvement:.1f}%")
    if tpc == 4:
        print(f"  Paper claim         : +12.5% at tpc=4")
        status = "CONFIRMED" if improvement > 8.0 else "BELOW PAPER CLAIM"
        print(f"  Status              : {status}")
    print("=" * 55)

    return [
        {"tpc": tpc, "method": "HAN+DDPM", "isr_mean": f"{hdm_isr:.4f}",
         "isr_std": f"{hdm_std:.4f}", "delay_mean": f"{hdm_delay:.3f}",
         "ddpm_improvement_pct": ""},
        {"tpc": tpc, "method": "HAN+MLP",  "isr_mean": f"{noddpm_isr:.4f}",
         "isr_std": f"{noddpm_std:.4f}", "delay_mean": f"{noddpm_delay:.3f}",
         "ddpm_improvement_pct": f"{improvement:.1f}"},
    ]


if __name__ == "__main__":
    all_rows = []
    for tpc in (4, 10):
        rows = run_one_tpc(tpc)
        all_rows.extend(rows)

    out = os.path.join(RESULTS_DIR, "ablation_ddpm.csv")
    fieldnames = ["tpc", "method", "isr_mean", "isr_std", "delay_mean",
                  "ddpm_improvement_pct"]
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(all_rows)
    print(f"\nWrote {out}")
