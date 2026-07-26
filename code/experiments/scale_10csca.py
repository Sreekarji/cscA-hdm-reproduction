"""
scale_10csca.py — 10-CSCA scaling experiment (paper Figs. 10/11).

Paper: "To validate the superiority of HDM in larger-scale wireless
communication networks, we increase the number of CSCA and relays to 10."

Patches HANMLPTrainer post-construction to use n_cscas=10, n_relays=10.
Run: python code/experiments/scale_10csca.py
~1-2 hours total.
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
from datetime import datetime
from reproducibility import set_seed
from train_han_mlp import (
    HANMLPTrainer, evaluate_policy, evaluate_static, evaluate_baseline_actor,
    DEVICE, POLICY, CHECKPOINT_PATH,
    train_ac_baseline, train_ppo_baseline, train_sac_baseline,
)

RESULTS_DIR = os.path.join(BASE, "results", "final")
os.makedirs(RESULTS_DIR, exist_ok=True)

N_CSCAS  = 10
N_RELAYS = 10
TPC_LIST = [1, 2, 4, 10]
N_EVAL   = 200

_HAN_KEYS   = ["han", "han_state_dict", "han_network", "model"]
_ACTOR_KEYS = ["actor", "actor_state_dict", "policy", "ddpm_actor", "mlp_actor"]

def timestamp():
    return datetime.now().strftime("%H:%M:%S")

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

def make_trainer(tpc):
    """
    Construct HANMLPTrainer then patch it to use N_CSCAS/N_RELAYS.
    HANMLPTrainer hardcodes n_cscas=5, n_relays=5 in __init__.
    We rebuild env, HAN, actor, and critic after patching.
    """
    from sim_channel import MultiCSCAEnvironment
    from han_network import HANNetwork
    from ddpm_policy import DDPMActor
    from mlp_policy import MLPCritic
    import torch.optim as optim

    # Build with default n_cscas=5 first (to get all the internal setup)
    trainer = HANMLPTrainer(tasks_per_csca=tpc, difficulty="medium")

    # Patch scale
    trainer.n_cscas  = N_CSCAS
    trainer.n_relays = N_RELAYS
    trainer.n_tasks  = tpc * N_CSCAS
    n_base_stations  = N_CSCAS  # paper: n_base_stations = n_cscas

    # Recompute action_dim: BW + relay logits + MCS logits
    trainer.action_dim = (trainer.n_tasks
                          + trainer.n_tasks * N_RELAYS
                          + trainer.n_tasks * trainer.n_mcs)

    # Rebuild environment with new scale
    trainer.env = MultiCSCAEnvironment(
        n_cscas=N_CSCAS,
        n_relays=N_RELAYS,
        tasks_per_csca=tpc,
        difficulty="medium",
    )

    # Rebuild HAN with correct node counts
    trainer.han = HANNetwork(
        hidden_channels=256, num_heads=8, num_layers=3, dropout=0.1,
        n_cscas=N_CSCAS, n_relays=N_RELAYS,
        n_messages=trainer.n_tasks, n_base_stations=n_base_stations,
    ).to(DEVICE)

    # Rebuild DDPM actor with new action_dim
    trainer.actor = DDPMActor(
        graph_emb_dim=256, task_emb_dim=256,
        action_dim=trainer.action_dim, hidden_dim=256,
        n_tasks=trainer.n_tasks, n_relays=N_RELAYS, n_mcs=trainer.n_mcs,
        n_denoising_steps=6,
    ).to(DEVICE)

    # Rebuild critic and critic_target with correct action_dim
    from train_han_mlp import MLPCritic
    trainer.critic = MLPCritic(
        state_dim=256, action_dim=trainer.action_dim, hidden_dim=512,
    ).to(DEVICE)
    import copy
    trainer.critic_target = copy.deepcopy(trainer.critic).to(DEVICE)
    for _p in trainer.critic_target.parameters():
        _p.requires_grad = False

    # Rebuild optimizers
    actor_lr = 5e-4 if trainer.n_tasks >= 40 else 1e-4
    trainer.opt_han    = optim.Adam(trainer.han.parameters(),    lr=1e-4)
    trainer.opt_actor  = optim.Adam(trainer.actor.parameters(),  lr=actor_lr)
    trainer.opt_critic = optim.Adam(trainer.critic.parameters(), lr=3e-4)
    trainer.sched_actor = optim.lr_scheduler.StepLR(
        trainer.opt_actor, step_size=300, gamma=0.7)

    return trainer

def run_one_tpc(tpc):
    print(f"\n{'='*60}")
    print(f"[{timestamp()}] n_cscas={N_CSCAS}  tpc={tpc}  ({tpc*N_CSCAS} tasks)")
    print(f"{'='*60}")

    set_seed(42)
    trainer = make_trainer(tpc)
    trainer.train(max_episodes=1000)

    ckpt_path = os.path.join(
        str(CHECKPOINT_PATH),
        f"han_{POLICY}_10csca_tpc{tpc}_best.pt"
    )
    _load_ckpt(trainer, ckpt_path)
    eval_env = trainer.env

    hdm_mean, hdm_std, hdm_delay, hdm_dist = evaluate_policy(trainer, N_EVAL)

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

    ac_mean,  ac_std,  ac_delay,  ac_dist  = evaluate_baseline_actor(
        ac_actor,  trainer.han, eval_env, trainer.n_tasks,
        trainer.n_relays, trainer.n_mcs, N_EVAL)
    ppo_mean, ppo_std, ppo_delay, ppo_dist = evaluate_baseline_actor(
        ppo_actor, trainer.han, eval_env, trainer.n_tasks,
        trainer.n_relays, trainer.n_mcs, N_EVAL)
    sac_mean, sac_std, sac_delay, sac_dist = evaluate_baseline_actor(
        sac_actor, trainer.han, eval_env, trainer.n_tasks,
        trainer.n_relays, trainer.n_mcs, N_EVAL)
    static_mean, static_std, static_delay, static_dist = evaluate_static(
        eval_env, trainer.n_tasks, trainer.n_relays, trainer.n_mcs, N_EVAL)

    results = {
        "HDM":    (hdm_mean,    hdm_std,    hdm_delay,    hdm_dist),
        "AC":     (ac_mean,     ac_std,     ac_delay,     ac_dist),
        "PPO":    (ppo_mean,    ppo_std,    ppo_delay,    ppo_dist),
        "SAC":    (sac_mean,    sac_std,    sac_delay,    sac_dist),
        "Static": (static_mean, static_std, static_delay, static_dist),
    }

    print(f"\n  n_cscas={N_CSCAS} tpc={tpc} results:")
    for name, (m, s, d, v) in results.items():
        print(f"    {name:<10} ISR={m:.4f} +/-{s:.4f}  delay={d:.3f}s  dist={v:.4f}")

    return results

if __name__ == "__main__":
    print(f"{'='*60}")
    print(f"10-CSCA Scaling Experiment (paper Figs. 10/11)")
    print(f"n_cscas={N_CSCAS}  n_relays={N_RELAYS}")
    print(f"tpc values: {TPC_LIST}  |  eval episodes: {N_EVAL}")
    print(f"Started: {timestamp()}")
    print(f"{'='*60}")

    all_results = {}
    for tpc in TPC_LIST:
        all_results[tpc] = run_one_tpc(tpc)

    csv_path = os.path.join(RESULTS_DIR, "isr_vs_tpc_10csca.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["n_cscas", "tpc", "policy", "isr_mean", "isr_std",
                    "delay_mean", "dist_mean", "n_eval"])
        for tpc in TPC_LIST:
            for policy in ["HDM", "AC", "PPO", "SAC", "Static"]:
                m, s, d, v = all_results[tpc][policy]
                w.writerow([N_CSCAS, tpc, policy,
                            f"{m:.4f}", f"{s:.4f}",
                            f"{d:.4f}", f"{v:.4f}", N_EVAL])
    print(f"\nWrote {csv_path}")

    print(f"\n{'='*60}")
    print("  10-CSCA ISR SUMMARY")
    print(f"{'='*60}")
    print(f"  {'tpc':>4}  {'HDM':>8}  {'AC':>8}  {'PPO':>8}  "
          f"{'SAC':>8}  {'Static':>8}  {'vs AC':>8}  {'vs PPO':>8}")
    print(f"  {'-'*72}")
    for tpc in TPC_LIST:
        hdm = all_results[tpc]["HDM"][0]
        ac  = all_results[tpc]["AC"][0]
        ppo = all_results[tpc]["PPO"][0]
        sac = all_results[tpc]["SAC"][0]
        sta = all_results[tpc]["Static"][0]
        vs_ac  = (hdm - ac)  / max(ac,  1e-8) * 100
        vs_ppo = (hdm - ppo) / max(ppo, 1e-8) * 100
        print(f"  {tpc:>4}  {hdm:>8.4f}  {ac:>8.4f}  {ppo:>8.4f}  "
              f"{sac:>8.4f}  {sta:>8.4f}  {vs_ac:>7.1f}%  {vs_ppo:>7.1f}%")
    print(f"{'='*60}")
    print(f"Done: {timestamp()}")
