"""
4B — HAN Architecture Ablation
Compares HAN+DDPM (HDM) vs MLPEncoder+DDPM at tpc=4.
Run: python code/experiments/ablation_han.py
Output: D:/MP2/results/final/ablation_han.csv
"""
import os
import sys
import csv
import torch
import numpy as np
from datetime import datetime

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for sub in ["code/hdm", "code/channel", "code/evaluation", "code/utils", "code/experiments"]:
    sys.path.insert(0, os.path.join(BASE, sub))

from reproducibility import set_seed
from train_han_mlp import (
    HANMLPTrainer, evaluate_policy,
    DEVICE, POLICY, CHECKPOINT_PATH, LOG_PATH,
)
from mlp_encoder import MLPEncoder

TPC = 4
N_EVAL = 200
RESULTS_DIR = os.path.join(BASE, "results", "final")
os.makedirs(RESULTS_DIR, exist_ok=True)


def log(msg):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line)
    with open(LOG_PATH, "a") as f:
        f.write(line + "\n")


class MLPEncoderTrainer(HANMLPTrainer):
    """
    HANMLPTrainer with HANNetwork swapped for flat MLPEncoder.
    Training loop, optimizer schedule, eval, checkpointing all inherited unchanged.
    """

    def __init__(self, tasks_per_csca=4, difficulty="medium"):
        super().__init__(tasks_per_csca=tasks_per_csca, difficulty=difficulty)

        self.han = MLPEncoder(
            task_input_dim=6,
            csca_input_dim=3,
            hidden_dim=256,
            output_dim=256,
        ).to(self.device)

        self.opt_han = torch.optim.Adam(self.han.parameters(), lr=1e-4)
        self.sched_han = torch.optim.lr_scheduler.StepLR(
            self.opt_han, step_size=300, gamma=0.7
        )

        self._ckpt_suffix = "_mlpenc"
        self.best_isr = 0.0
        self.history = []
        self.episode = 0
        self._ema_reward = 0.0
        self.replay = []


def main():
    log("=" * 60)
    log(f"HAN Architecture Ablation | tpc={TPC} | policy={POLICY}")
    log("HAN+DDPM (HDM) vs MLPEncoder+DDPM (flat, no graph attention)")
    log("=" * 60)

    # Step 1: Train MLPEncoder+DDPM
    log("\n[Step 1] Training MLPEncoder+DDPM (1000 episodes, seed 42)...")
    set_seed(42)
    mlp_trainer = MLPEncoderTrainer(tasks_per_csca=TPC, difficulty="medium")
    mlp_trainer.train(max_episodes=1000)

    ckpt_mlp = os.path.join(CHECKPOINT_PATH, f"han_{POLICY}_tpc{TPC}_mlpenc_best.pt")
    if os.path.exists(ckpt_mlp):
        ckpt = torch.load(ckpt_mlp, map_location=DEVICE)
        mlp_trainer.han.load_state_dict(ckpt["han"])
        mlp_trainer.actor.load_state_dict(ckpt["actor"])
        log(f"  Loaded best MLPEncoder ckpt (ISR={ckpt['isr']:.3f}, ep {ckpt['episode']})")

    mlp_mean, mlp_std, mlp_delay, mlp_dist = evaluate_policy(mlp_trainer, N_EVAL)
    log(f"MLPEncoder+DDPM: ISR={mlp_mean:.4f} +/- {mlp_std:.4f}  delay={mlp_delay:.3f}s  dist={mlp_dist:.4f}")

    # Step 2: Load existing HDM checkpoint and evaluate
    log(f"\n[Step 2] Loading existing HAN+DDPM (HDM) ckpt for tpc={TPC}...")
    ckpt_hdm = os.path.join(CHECKPOINT_PATH, f"han_{POLICY}_tpc{TPC}_best.pt")

    set_seed(42)
    hdm_trainer = HANMLPTrainer(tasks_per_csca=TPC, difficulty="medium")

    if os.path.exists(ckpt_hdm):
        ckpt = torch.load(ckpt_hdm, map_location=DEVICE)
        hdm_trainer.han.load_state_dict(ckpt["han"])
        hdm_trainer.actor.load_state_dict(ckpt["actor"])
        log(f"  Loaded HDM ckpt (ISR={ckpt['isr']:.3f}, ep {ckpt['episode']})")
    else:
        log("  WARNING: HDM ckpt not found — training from scratch (adds ~90 min)")
        hdm_trainer.train(max_episodes=1000)

    hdm_mean, hdm_std, hdm_delay, hdm_dist = evaluate_policy(hdm_trainer, N_EVAL)
    log(f"HAN+DDPM (HDM):  ISR={hdm_mean:.4f} +/- {hdm_std:.4f}  delay={hdm_delay:.3f}s  dist={hdm_dist:.4f}")

    # Step 3: Compare and write
    pct = (hdm_mean - mlp_mean) / max(mlp_mean, 1e-6) * 100

    log("\n" + "=" * 60)
    log(f"ABLATION RESULT — HAN vs flat MLP encoder (tpc={TPC})")
    log("=" * 60)
    log(f"  HAN+DDPM (HDM):           ISR={hdm_mean:.4f} +/- {hdm_std:.4f}")
    log(f"  MLPEncoder+DDPM (ablat):  ISR={mlp_mean:.4f} +/- {mlp_std:.4f}")
    log(f"  HAN advantage: {pct:+.1f}%")

    if pct > 5:
        verdict = "HAN graph attention adds meaningful benefit. Architecture justified."
    elif pct > 0:
        verdict = "HAN marginally better than flat MLP. Advantage within noise band."
    else:
        verdict = "Flat MLP matches or exceeds HAN. Report honestly — still a valid finding."
    log(f"  VERDICT: {verdict}")

    # CSV
    abl_path = os.path.join(RESULTS_DIR, "ablation_han.csv")
    with open(abl_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["encoder", "isr_mean", "isr_std", "delay_mean", "dist_mean", "n_eval", "tpc"])
        w.writerow(["HAN (graph attention)", f"{hdm_mean:.4f}", f"{hdm_std:.4f}",
                    f"{hdm_delay:.4f}", f"{hdm_dist:.4f}", N_EVAL, TPC])
        w.writerow(["MLPEncoder (flat, no attention)", f"{mlp_mean:.4f}", f"{mlp_std:.4f}",
                    f"{mlp_delay:.4f}", f"{mlp_dist:.4f}", N_EVAL, TPC])
        w.writerow(["HAN_advantage_%", f"{pct:.2f}", "", "", "", "", ""])
    log(f"Wrote {abl_path}")

    # Append to SUMMARY.txt
    summary = os.path.join(RESULTS_DIR, "SUMMARY.txt")
    with open(summary, "a") as f:
        f.write("\n" + "=" * 60 + "\n")
        f.write("HAN Architecture Ablation (Fig 13a equivalent)\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"tpc={TPC}, {N_EVAL} eval episodes, policy={POLICY}, seed=42\n")
        f.write("=" * 60 + "\n")
        f.write(f"  HAN+DDPM (HDM):              ISR={hdm_mean:.4f} +/- {hdm_std:.4f}  delay={hdm_delay:.3f}s\n")
        f.write(f"  MLPEncoder+DDPM (ablation):  ISR={mlp_mean:.4f} +/- {mlp_std:.4f}  delay={mlp_delay:.3f}s\n")
        f.write(f"  HAN advantage: {pct:+.1f}%\n")
        f.write(f"  Verdict: {verdict}\n")
    log(f"Appended to {summary}")


if __name__ == "__main__":
    main()
