"""
final_results.py (LAM-augmented) — FIX 20 + LAM integration.

USE_LAM=False (default) -> zero divergence from original. Published results.
USE_LAM=True            -> LAM intent injection at eval time. No retraining.

  set USE_LAM=1 && python code/experiments/final_results_lam.py
  set LAM_TEXT=send it fast, 1 second && set USE_LAM=1 && python code/experiments/final_results_lam.py
"""

import os
import sys

_SELF_DIR = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.dirname(os.path.dirname(_SELF_DIR))
for sub in ["code/hdm", "code/channel", "code/evaluation", "code/utils", "code/experiments"]:
    sys.path.insert(0, os.path.join(BASE, sub))
sys.path.insert(0, _SELF_DIR)

import csv
import torch
import numpy as np
from datetime import datetime

from reproducibility import set_seed
from train_han_mlp import (
    HANMLPTrainer, evaluate_policy, evaluate_static, evaluate_baseline_actor,
    sample_eval_state, intents_from_state, parse_action, DEVICE, POLICY, CHECKPOINT_PATH,
    train_ac_baseline, train_ppo_baseline, train_sac_baseline,
    MultiCSCAEnvironment, compute_isr, compute_cscqi, _task_metrics,
)
from ddpm_policy import DDPMActor

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS_DIR = os.path.join(BASE, "results", "final")
os.makedirs(RESULTS_DIR, exist_ok=True)

TGC_LIST = [1, 2, 4, 10]
N_EVAL   = 200

USE_LAM   = bool(int(os.environ.get("USE_LAM", "0")))
LAM_TEXT  = os.environ.get("LAM_TEXT", "send the data accurately within 2 seconds")
LAM_MODEL = os.environ.get("LAM_MODEL", "qwen2.5vl:3b")

_DELAY_MIN, _DELAY_MAX = 0.50, 2.50
_QUAL_MIN,  _QUAL_MAX  = 0.10, 0.40

def _lam_intents():
    if not hasattr(_lam_intents, "_cache"):
        from lam_intent_generator import parse_intent_from_text, normalize_intent
        raw_d, raw_q = parse_intent_from_text(LAM_TEXT, model=LAM_MODEL)
        d, q = normalize_intent(raw_d, raw_q)
        print(f"[LAM] '{LAM_TEXT}'")
        print(f"      raw    delay={raw_d:.3f}s  quality={raw_q:.3f}")
        print(f"      normed delay={d:.3f}s  quality={q:.3f}")
        _lam_intents._cache = (d, q)
    return _lam_intents._cache

def _inject_lam(state, delay_s, quality, n_tasks):
    state["SCt"]["delay_intents"]   = [delay_s]  * n_tasks
    state["SCt"]["quality_intents"] = [quality]  * n_tasks
    for i in range(n_tasks):
        ds_norm = min(state["SCt"]["data_sizes"][i] / 6e5, 1.0)
        di      = delay_s / 10.0
        qi      = quality
        urgency = (1.0 - di) * 0.5 + qi * 0.5
        state["SCt"]["message_features"][i] = [ds_norm, di, qi, urgency]
    return state

def _sample_state(env, n_tasks):
    state = sample_eval_state(env)
    if USE_LAM:
        d, q = _lam_intents()
        state = _inject_lam(state, d, q, n_tasks)
    return state

def _call_actor(actor, graph_emb, msg_embs):
    cls = type(actor).__name__
    if "DDPM" in cls or "Diffusion" in cls:
        return actor(graph_emb, message_embs=msg_embs, deterministic=True)
    return actor(graph_emb, message_embs=msg_embs)

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
    print(f"  Loaded best checkpoint {isr_str} {ep_str}")
    return True

def evaluate_policy_lam(trainer, n_episodes=200):
    if not USE_LAM:
        return evaluate_policy(trainer, n_episodes)
    trainer.han.eval(); trainer.actor.eval()
    isrs, delays, dists = [], [], []
    with torch.no_grad():
        for _ in range(n_episodes):
            state = _sample_state(trainer.env, trainer.n_tasks)
            intent_vectors = intents_from_state(state)
            graph_emb, _, msg_embs = trainer.han.encode_state(
                state, intent_vectors=intent_vectors)
            action = _call_actor(trainer.actor, graph_emb, msg_embs)
            result = trainer.env.step(
                parse_action(action, trainer.n_tasks, trainer.n_relays, trainer.n_mcs), state)
            isr, delay, dist = _task_metrics(result["tasks"])
            isrs.append(isr); delays.append(delay); dists.append(dist)
    trainer.han.train(); trainer.actor.train()
    return (float(np.mean(isrs)), float(np.std(isrs)),
            float(np.mean(delays)), float(np.mean(dists)))

def evaluate_baseline_actor_lam(actor, han, env, n_tasks, n_relays, n_mcs, n_episodes=200):
    if not USE_LAM:
        return evaluate_baseline_actor(actor, han, env, n_tasks, n_relays, n_mcs, n_episodes)
    try:
        from train_han_mlp import RawStateActor, flatten_state
        _has_raw = True
    except ImportError:
        _has_raw = False
    actor.eval()
    device = next(actor.parameters()).device
    isrs, delays, dists = [], [], []
    with torch.no_grad():
        for _ in range(n_episodes):
            state = _sample_state(env, n_tasks)
            if _has_raw and isinstance(actor, RawStateActor):
                sv = flatten_state(state, n_tasks).to(device)
                action = actor(sv)
            else:
                intent_vectors = intents_from_state(state)
                graph_emb, _, msg_embs = han.encode_state(
                    state, intent_vectors=intent_vectors)
                action = actor(graph_emb, msg_embs)
            result = env.step(parse_action(action, n_tasks, n_relays, n_mcs), state)
            isr, delay, dist = _task_metrics(result["tasks"])
            isrs.append(isr); delays.append(delay); dists.append(dist)
    return (float(np.mean(isrs)), float(np.std(isrs)),
            float(np.mean(delays)), float(np.mean(dists)))

def evaluate_static_lam(env, n_tasks, n_relays, n_mcs, n_episodes=200):
    if not USE_LAM:
        return evaluate_static(env, n_tasks, n_relays, n_mcs, n_episodes)
    _orig = env.generate_state
    d, q  = _lam_intents()
    def _patched():
        state = _orig()
        return _inject_lam(state, d, q, n_tasks)
    env.generate_state = _patched
    try:
        result = evaluate_static(env, n_tasks, n_relays, n_mcs, n_episodes)
    finally:
        env.generate_state = _orig
    return result

def timestamp():
    return datetime.now().strftime("%H:%M:%S")

def run_one_tpc(tpc):
    print(f"\n{'='*60}")
    print(f"[{timestamp()}] tpc={tpc} ({tpc*5} tasks)" + (" [USE_LAM=True]" if USE_LAM else ""))
    print(f"{'='*60}")

    set_seed(42)
    trainer = HANMLPTrainer(tasks_per_csca=tpc, difficulty="medium")
    trainer.train(max_episodes=1000)

    ckpt_path = os.path.join(str(CHECKPOINT_PATH), f"han_{POLICY}_tpc{tpc}_best.pt")
    _load_ckpt(trainer, ckpt_path)
    eval_env = trainer.env

    hdm_mean, hdm_std, hdm_delay, hdm_dist = evaluate_policy_lam(trainer, N_EVAL)

    set_seed(42)
    print(f"[{timestamp()}] Training AC baseline...")
    ac_actor = train_ac_baseline(
        trainer.han, eval_env, trainer.n_tasks, trainer.n_relays, trainer.n_mcs,
        trainer.action_dim, n_episodes=1000)
    set_seed(42)
    print(f"[{timestamp()}] Training PPO baseline...")
    ppo_actor = train_ppo_baseline(
        trainer.han, eval_env, trainer.n_tasks, trainer.n_relays, trainer.n_mcs,
        trainer.action_dim, n_episodes=1000)
    set_seed(42)
    print(f"[{timestamp()}] Training SAC baseline...")
    sac_actor = train_sac_baseline(
        trainer.han, eval_env, trainer.n_tasks, trainer.n_relays, trainer.n_mcs,
        trainer.action_dim, n_episodes=1000)

    ac_mean,  ac_std,  ac_delay,  ac_dist  = evaluate_baseline_actor_lam(
        ac_actor,  trainer.han, eval_env, trainer.n_tasks, trainer.n_relays, trainer.n_mcs, N_EVAL)
    ppo_mean, ppo_std, ppo_delay, ppo_dist = evaluate_baseline_actor_lam(
        ppo_actor, trainer.han, eval_env, trainer.n_tasks, trainer.n_relays, trainer.n_mcs, N_EVAL)
    sac_mean, sac_std, sac_delay, sac_dist = evaluate_baseline_actor_lam(
        sac_actor, trainer.han, eval_env, trainer.n_tasks, trainer.n_relays, trainer.n_mcs, N_EVAL)
    static_mean, static_std, static_delay, static_dist = evaluate_static_lam(
        eval_env, trainer.n_tasks, trainer.n_relays, trainer.n_mcs, N_EVAL)

    results = {
        "HDM":    (hdm_mean,    hdm_std,    hdm_delay,    hdm_dist),
        "AC":     (ac_mean,     ac_std,     ac_delay,     ac_dist),
        "PPO":    (ppo_mean,    ppo_std,    ppo_delay,    ppo_dist),
        "SAC":    (sac_mean,    sac_std,    sac_delay,    sac_dist),
        "Static": (static_mean, static_std, static_delay, static_dist),
    }

    print(f"\n  tpc={tpc} results:")
    for name, (m, s, d, v) in results.items():
        print(f"    {name:<10} ISR={m:.4f} +/-{s:.4f}  delay={d:.3f}s  dist={v:.4f}")

    return results, trainer.history

def run_ablation_N():
    print(f"\n{'='*60}\nN-step ablation (tpc=4, N in [5, 6, 7])\n{'='*60}")
    ablation_results = {}
    for N in [5, 6, 7]:
        print(f"\n[{timestamp()}] Training N={N}...")
        set_seed(42)
        trainer = HANMLPTrainer(tasks_per_csca=4, difficulty="medium")
        trainer._ckpt_suffix = f"_N{N}"
        trainer.actor = DDPMActor(
            graph_emb_dim=256, task_emb_dim=256,
            action_dim=trainer.action_dim, hidden_dim=256,
            n_tasks=trainer.n_tasks, n_relays=trainer.n_relays, n_mcs=trainer.n_mcs,
            n_denoising_steps=N,
        ).to(DEVICE)
        trainer.opt_actor = torch.optim.Adam(trainer.actor.parameters(), lr=1e-4)
        trainer.sched_actor = torch.optim.lr_scheduler.StepLR(
            trainer.opt_actor, step_size=300, gamma=0.7)
        trainer.train(max_episodes=1000)
        ckpt_path = os.path.join(str(CHECKPOINT_PATH), f"han_{POLICY}_tpc4_N{N}_best.pt")
        _load_ckpt(trainer, ckpt_path)
        mean_isr, std_isr, _, _ = evaluate_policy_lam(trainer, 200)
        ablation_results[N] = (mean_isr, std_isr)
        print(f"  N={N}: ISR={mean_isr:.4f} +/- {std_isr:.4f}")

    ablation_path = os.path.join(RESULTS_DIR, "ablation_N.csv")
    with open(ablation_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["N_steps", "isr_mean", "isr_std"])
        for N, (m, s) in ablation_results.items():
            w.writerow([N, f"{m:.4f}", f"{s:.4f}"])
    print(f"Wrote {ablation_path}")
    return ablation_results

def main():
    print(f"{'='*60}")
    print(f"CSCA Final Experiments ({POLICY.upper()} policy)")
    if USE_LAM:
        print(f"LAM MODE  intent: '{LAM_TEXT}'  model: {LAM_MODEL}")
    else:
        print("USE_LAM=False — synthetic intents (original results)")
    print(f"tpc values: {TGC_LIST}  |  eval episodes: {N_EVAL}  |  {timestamp()}")
    print(f"{'='*60}")

    all_results = {}
    all_history = {}
    for tpc in TGC_LIST:
        results, history = run_one_tpc(tpc)
        all_results[tpc] = results
        all_history[tpc] = history

    suffix = "_lam" if USE_LAM else ""

    csv_path = os.path.join(RESULTS_DIR, f"isr_vs_tpc{suffix}.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["tpc", "policy", "isr_mean", "isr_std", "delay_mean", "dist_mean", "n_eval"])
        for tpc in TGC_LIST:
            for policy in ["HDM", "AC", "PPO", "SAC", "Static"]:
                m, s, d, v = all_results[tpc][policy]
                w.writerow([tpc, policy, f"{m:.4f}", f"{s:.4f}", f"{d:.4f}", f"{v:.4f}", N_EVAL])
    print(f"\nWrote {csv_path}")

    for tpc in TGC_LIST:
        conv_path = os.path.join(RESULTS_DIR, f"convergence_tpc{tpc}{suffix}.csv")
        with open(conv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["episode", "train_cscqi", "train_isr", "critic_loss", "actor_loss"])
            for ep, cscqi, isr, closs, aloss in all_history[tpc]:
                w.writerow([ep, f"{cscqi:.4f}", f"{isr:.3f}", f"{closs:.4f}", f"{aloss:.4f}"])
        print(f"Wrote {conv_path}")

    fig, ax = plt.subplots(figsize=(8, 5))
    styles = {
        "HDM":    {"color": "blue",   "marker": "o", "ls": "-",  "lw": 2},
        "AC":     {"color": "red",    "marker": "s", "ls": "--"},
        "PPO":    {"color": "green",  "marker": "^", "ls": "--"},
        "SAC":    {"color": "orange", "marker": "D", "ls": "--"},
        "Static": {"color": "black",  "marker": "x", "ls": ":",  "lw": 2},
    }
    for policy in ["HDM", "AC", "PPO", "SAC", "Static"]:
        means = [all_results[tpc][policy][0] for tpc in TGC_LIST]
        stds  = [all_results[tpc][policy][1] for tpc in TGC_LIST]
        ax.errorbar(TGC_LIST, means, yerr=stds, label=policy, capsize=3, **styles[policy])
    ax.set_xlabel("Tasks per CSCA (tpc)")
    ax.set_ylabel("ISR (Intent Satisfaction Rate)")
    ax.set_title("ISR vs Congestion Level" + (" (LAM intents)" if USE_LAM else ""))
    ax.legend(); ax.grid(True, alpha=0.3); ax.set_xticks(TGC_LIST)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, f"isr_vs_tpc{suffix}.png"), dpi=150)
    plt.close()
    print(f"Wrote isr_vs_tpc{suffix}.png")

    if 4 in all_history:
        hist4     = all_history[4]
        eps       = [h[0] for h in hist4]
        train_isr = [h[2] for h in hist4]
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.scatter(eps, train_isr, alpha=0.3, s=10, color="gray", label="Train ISR (per-step)")
        if len(train_isr) > 5:
            window   = min(10, len(train_isr) // 3)
            smoothed = np.convolve(train_isr, np.ones(window) / window, mode="valid")
            ax.plot(eps[window-1:], smoothed, color="blue", lw=2, label="Train ISR (smoothed)")
        ax.set_xlabel("Episode"); ax.set_ylabel("ISR")
        ax.set_title("Convergence (tpc=4)"); ax.legend(); ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, f"convergence_tpc4{suffix}.png"), dpi=150)
        plt.close()
        print(f"Wrote convergence_tpc4{suffix}.png")

if __name__ == "__main__":
    main()
