"""
2A — log π Actor Loss (Eq. 33) validation at tpc=4.
Adds forward_with_logprob() to DDPMActor inline (no permanent file modification).
actor_loss = -(LAMBDA_PI * log_pi + Q.mean()), LAMBDA_PI=0.01.

Run: python code/experiments/ablation_logpi.py
Output: D:/MP2/results/final/ablation_logpi.csv
"""
import os
import sys
import csv
import math
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from datetime import datetime

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for sub in ["code/hdm", "code/channel", "code/evaluation", "code/utils", "code/experiments"]:
    sys.path.insert(0, os.path.join(BASE, sub))

from reproducibility import set_seed
from sim_channel import MultiCSCAEnvironment
from cscqi import compute_cscqi, compute_isr
from ddpm_policy import DDPMActor
from han_network import HANNetwork
from mlp_policy import MLPCritic
from train_han_mlp import (
    DEVICE, POLICY, CHECKPOINT_PATH, LOG_PATH,
    sample_eval_state, intents_from_state, parse_action,
    evaluate_policy, _act,
)

TPC = 4
N_EVAL = 200
LAMBDA_PI = 0.01
RESULTS_DIR = os.path.join(BASE, "results", "final")
os.makedirs(RESULTS_DIR, exist_ok=True)


def log(msg):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line)
    with open(LOG_PATH, "a") as f:
        f.write(line + "\n")


_LOG_2PI = math.log(2 * math.pi)


def _forward_with_logprob(self, graph_emb, message_embs):
    """
    Run reverse diffusion, return (action [1, action_dim], log_pi scalar tensor).
    log_pi = sum of Gaussian log-probs over all stochastic steps (n=N..2).
    Step n=1 is deterministic (no noise added) so contributes 0 to log_pi.
    Gradients flow through both action and log_pi into the denoiser.
    """
    device = graph_emb.device
    if graph_emb.dim() == 1:
        graph_emb = graph_emb.unsqueeze(0)

    a_n = torch.randn(self.n_tasks, self.task_dim, device=device)
    log_pi = torch.zeros(1, device=device)

    for n in range(self.N, 0, -1):
        beta_n  = self.betas[n - 1]
        alpha_n = self.alphas[n - 1]
        abar_n  = self.alphas_cumprod[n - 1]

        eps_pred = self.denoiser(a_n, n, graph_emb, message_embs)
        mean = (1.0 / torch.sqrt(alpha_n)) * (
            a_n - (beta_n / torch.sqrt(1.0 - abar_n)) * eps_pred
        )

        if n > 1:
            var   = self.beta_tilde[n - 1]
            noise = torch.randn_like(a_n)
            a_n   = mean + torch.sqrt(var) * noise
            log_pi = log_pi + (
                -0.5 * (noise.pow(2) + _LOG_2PI + torch.log(var))
            ).sum()
        else:
            a_n = mean

    bw  = torch.softmax(a_n[:, 0] / self.bw_temperature.abs().clamp_min(0.1),
                        dim=0).unsqueeze(0)
    mcs = torch.sigmoid(a_n[:, 1:]).reshape(1, -1)
    action = torch.cat([bw, mcs], dim=-1)
    return action, log_pi


DDPMActor.forward_with_logprob = _forward_with_logprob


from train_han_mlp import HANMLPTrainer


class LogPiTrainer(HANMLPTrainer):
    """HANMLPTrainer with Eq. 33 log-pi actor loss for ablation."""

    def __init__(self, tasks_per_csca=4, difficulty="medium"):
        super().__init__(tasks_per_csca=tasks_per_csca, difficulty=difficulty)
        self._ckpt_suffix = "_logpi"

    def train_step(self):
        self.han.train()
        self.actor.train()
        self.critic.train()
        self.episode += 1

        state = sample_eval_state(self.env)
        intent_vectors = intents_from_state(state)

        graph_emb, _, msg_embs = self.han.encode_state(
            state, intent_vectors=intent_vectors
        )
        action = self.actor(graph_emb, message_embs=msg_embs)
        sigma = max(0.02, 0.10 * (1.0 - self.episode / 1000))
        bw_logits = torch.log(action[:, :self.n_tasks].clamp_min(1e-8))
        bw_logits = bw_logits + torch.randn_like(bw_logits) * sigma * 3.0
        bw = torch.softmax(bw_logits, dim=-1)
        mcs = (action[:, self.n_tasks:]
               + torch.randn_like(action[:, self.n_tasks:]) * sigma).clamp(0.0, 1.0)
        action = torch.cat([bw, mcs], dim=-1)

        result = self.env.step(
            parse_action(action, self.n_tasks, self.n_relays, self.n_mcs), state
        )
        tasks = result["tasks"]
        cscqi_vals = [
            compute_cscqi(
                t["tau_S"], t["vartheta_S"],
                t["tau_S_int"], t["vartheta_S_int"],
                w_tau=0.5, w_vartheta=0.5,
            )
            for t in tasks
        ]
        reward = float(np.mean(cscqi_vals))
        isr    = compute_isr(tasks)

        if not np.isfinite(reward):
            return 0.0, 0.0, 0.0, 0.0

        self.replay.append((graph_emb.detach(), action.detach(), reward))
        if len(self.replay) > self.replay_cap:
            self.replay.pop(0)

        if len(self.replay) >= self.critic_batch:
            batch = random.sample(self.replay, self.critic_batch)
            g_mb = torch.cat([b[0] for b in batch], dim=0)
            a_mb = torch.cat([b[1] for b in batch], dim=0)
            r_mb = torch.tensor([[b[2]] for b in batch], dtype=torch.float, device=self.device)
        else:
            g_mb = graph_emb.detach()
            a_mb = action.detach()
            r_mb = torch.tensor([[reward]], dtype=torch.float, device=self.device)

        value_pred  = self.critic(g_mb, a_mb)
        critic_loss = nn.MSELoss()(value_pred, r_mb)
        self.opt_critic.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
        self.opt_critic.step()

        # Actor + HAN update — Eq. 33
        graph_emb2, _, msg_embs2 = self.han.encode_state(
            state, intent_vectors=intent_vectors
        )
        action2, log_pi = self.actor.forward_with_logprob(graph_emb2, msg_embs2)
        q_value = self.critic(graph_emb2, action2)

        actor_loss = -(LAMBDA_PI * log_pi + q_value.mean())

        self.opt_han.zero_grad()
        self.opt_actor.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.han.parameters(),   self.max_grad_norm)
        nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
        self.opt_han.step()
        self.opt_actor.step()
        self.opt_critic.zero_grad()

        self.sched_han.step()
        self.sched_actor.step()

        c_loss_val = critic_loss.item()
        a_loss_val = actor_loss.item()
        if not (np.isfinite(c_loss_val) and np.isfinite(a_loss_val)):
            self.opt_han.zero_grad()
            self.opt_actor.zero_grad()
            self.opt_critic.zero_grad()
            return 0.0, 0.0, 0.0, 0.0

        return reward, c_loss_val, a_loss_val, isr


BASELINE_ISR_TPC4 = 0.3822


def main():
    log("=" * 60)
    log(f"2A — log pi Actor Loss (Eq. 33) | tpc={TPC} | LAMBDA_PI={LAMBDA_PI}")
    log(f"Baseline to beat: ISR={BASELINE_ISR_TPC4}")
    log("=" * 60)

    set_seed(42)
    trainer = LogPiTrainer(tasks_per_csca=TPC, difficulty="medium")
    trainer.train(max_episodes=1000)

    ckpt_logpi = os.path.join(CHECKPOINT_PATH, f"han_{POLICY}_tpc{TPC}_logpi_best.pt")
    if os.path.exists(ckpt_logpi):
        ckpt = torch.load(ckpt_logpi, map_location=DEVICE)
        trainer.han.load_state_dict(ckpt["han"])
        trainer.actor.load_state_dict(ckpt["actor"])
        log(f"  Loaded best logpi ckpt (ISR={ckpt['isr']:.3f}, ep {ckpt['episode']})")

    new_mean, new_std, new_delay, new_dist = evaluate_policy(trainer, N_EVAL)

    delta = new_mean - BASELINE_ISR_TPC4
    pct   = delta / max(BASELINE_ISR_TPC4, 1e-6) * 100

    log("\n" + "=" * 60)
    log(f"RESULT — log pi loss (tpc={TPC}, LAMBDA_PI={LAMBDA_PI})")
    log("=" * 60)
    log(f"  Baseline HDM (-Q.mean()):        ISR={BASELINE_ISR_TPC4:.4f}")
    log(f"  log pi HDM (Eq. 33):             ISR={new_mean:.4f} +/- {new_std:.4f}  delay={new_delay:.3f}s")
    log(f"  Delta: {delta:+.4f}  ({pct:+.1f}%)")

    if delta > 0.01:
        recommendation = "PROCEED: run full tpc=[1,2,4,10] with log pi loss."
    elif delta > -0.01:
        recommendation = "MARGINAL: within noise band. Either loss is defensible."
    else:
        recommendation = "REVERT: -Q.mean() is stronger. Defend as DPG-stable variant."
    log(f"  RECOMMENDATION: {recommendation}")

    out_path = os.path.join(RESULTS_DIR, "ablation_logpi.csv")
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["loss_variant", "isr_mean", "isr_std", "delay_mean", "dist_mean",
                    "delta_vs_baseline", "pct_vs_baseline", "n_eval", "tpc", "lambda_pi"])
        w.writerow(["baseline_Q_only", f"{BASELINE_ISR_TPC4:.4f}", "0.0964",
                    "", "", "0.0", "0.0", N_EVAL, TPC, "N/A"])
        w.writerow([f"logpi_lambda{LAMBDA_PI}", f"{new_mean:.4f}", f"{new_std:.4f}",
                    f"{new_delay:.4f}", f"{new_dist:.4f}",
                    f"{delta:+.4f}", f"{pct:+.1f}", N_EVAL, TPC, LAMBDA_PI])
        w.writerow(["recommendation", recommendation, "", "", "", "", "", "", "", ""])

    log(f"Wrote {out_path}")

    summary = os.path.join(RESULTS_DIR, "SUMMARY.txt")
    with open(summary, "a") as f:
        f.write("\n" + "=" * 60 + "\n")
        f.write(f"log pi Actor Loss Ablation (Eq. 33) | tpc={TPC} | LAMBDA_PI={LAMBDA_PI}\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 60 + "\n")
        f.write(f"  Baseline (-Q.mean()):  ISR={BASELINE_ISR_TPC4:.4f}\n")
        f.write(f"  Eq. 33 (log pi + Q):  ISR={new_mean:.4f} +/- {new_std:.4f}  delay={new_delay:.3f}s\n")
        f.write(f"  Delta: {delta:+.4f} ({pct:+.1f}%)\n")
        f.write(f"  Recommendation: {recommendation}\n")
    log(f"Appended to {summary}")


if __name__ == "__main__":
    main()
