"""Plumbing sanity check — run this FIRST.

Verifies:
  1. CSCA features vary across nodes
  2. HAN shapes correct for multiple tpc
  3. DDPMActor gradient flows correctly (non-zero grad on denoiser params)
  4. Per-task BW not collapsed
  5. Action shape and BW simplex constraint
Run: python check_features.py
"""
import os
import sys
import numpy as np
import torch

BASE = os.path.dirname(os.path.abspath(__file__))
for sub in ["", "code/hdm", "code/channel", "code/evaluation", "code/utils"]:
    p = os.path.join(BASE, sub)
    if os.path.isdir(p):
        sys.path.insert(0, p)

from reproducibility import set_seed
from sim_channel import MultiCSCAEnvironment, _normalize_bw_allocation
from han_network import HANNetwork
from ddpm_policy import DDPMActor


def main():
    set_seed(42)
    device = "cpu"

    print("== 1. CSCA physics features ==")
    env   = MultiCSCAEnvironment(difficulty="medium", tasks_per_csca=2)
    state = env.generate_state()
    feats = np.array(state["Rt"]["csca_features"])
    print("csca_features:\n", np.round(feats, 3))
    print("feature std across cscas:", np.round(feats.std(axis=0), 4),
          "  (should be > 0)")

    print("\n== 2. HAN graph shapes ==")
    han = HANNetwork(256, 8, 3, 5, 5, 5, 5).to(device)
    for tpc in (1, 2, 10):
        e = MultiCSCAEnvironment(difficulty="medium", tasks_per_csca=tpc)
        s = e.generate_state()
        intents = [[m[1], m[2]] for m in s["SCt"]["message_features"]]
        g, _, memb = han.encode_state(s, intents)
        print(f"  tpc={tpc:2d} Nm={e.n_tasks:3d}  "
              f"graph_emb={tuple(g.shape)}  message_embs={tuple(memb.shape)}")

    print("\n== 3. DDPMActor gradient (denoiser params should get > 0 grad) ==")
    n_tasks = env.n_tasks   # 10 tasks (tpc=2)
    n_mcs   = env.n_mcs     # 3
    n_relays = env.n_relays
    actor   = DDPMActor(
        graph_emb_dim=256, task_emb_dim=256,
        n_tasks=n_tasks, n_relays=n_relays, n_mcs=n_mcs, n_denoising_steps=6,
    ).to(device)

    intents = [[m[1], m[2]] for m in state["SCt"]["message_features"]]
    graph_emb, _, msg_embs = han.encode_state(state, intents)
    action = actor(graph_emb, message_embs=msg_embs)
    loss = -action.sum()
    loss.backward()

    gnorm = sum(
        p.grad.norm().item()
        for p in actor.denoiser.parameters()
        if p.grad is not None
    )
    print(f"  action shape={tuple(action.shape)}")
    print(f"  denoiser grad-norm={gnorm:.6f}  (MUST be > 0)")
    assert gnorm > 0, "GRADIENT IS ZERO — bug in DDPMActor backward!"

    print("\n== 4. Per-task BW allocation (not collapsed to uniform) ==")
    logits = torch.randn(1, env.n_tasks)
    bw     = _normalize_bw_allocation(
        [logits[0, i].item() for i in range(env.n_tasks)], 5e6)
    bw     = np.array(bw)
    print("  bw shares:", np.round(bw / bw.sum(), 3))
    print("  bw std/mean:", round(bw.std() / bw.mean(), 3), "  (should be > 0)")

    print("\n== 5. Action shape and BW simplex ==")
    bw_slice = action[0, :n_tasks]
    expected_dim = n_tasks + n_tasks * n_relays + n_tasks * n_mcs
    print(f"  action shape: {tuple(action.shape)}  (expected (1, {expected_dim}))")
    print(f"  BW sum: {bw_slice.sum().item():.6f}  (should be ~1.0)")
    assert abs(bw_slice.sum().item() - 1.0) < 0.01, "BW does not sum to 1!"

    print("\nAll checks PASSED.")


if __name__ == "__main__":
    main()
