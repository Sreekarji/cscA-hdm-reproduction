#!/usr/bin/env python3
"""
temp_scale_sweep.py — Read-only diagnostic. NO training, NO checkpoint writes.

Question: does the DDPM actor's temp_scale anneal (2.0 -> 0.5 over training)
cause tpc=2 eval ISR to peak early (ep ~50) and then degrade?

temp_scale sets how peaked the bandwidth softmax is in the DETERMINISTIC eval
policy (ddpm_policy.py:145-146), independent of the reverse-diffusion noise.
  high temp_scale (~2.0, early training)  -> near-uniform bandwidth
  low  temp_scale (0.5, late training)    -> sharply peaked bandwidth
At low load (tpc=2) uniform allocation is near-optimal (Static ~0.83), so if
the anneal is the mechanism, ISR should FALL as temp_scale drops.

Method: load one existing checkpoint, override temp_scale to each fixed value,
and run the SAME eval episodes at each (set_seed before every row, so all rows
see an identical state sequence and only temp_scale varies).

Interpretation:
  ISR rises with temp_scale  -> the 0.5 endpoint hurts; floor it higher, retrain.
  ISR ~flat                  -> anneal is NOT the mechanism; look elsewhere.

Run:
  python code/experiments/temp_scale_sweep.py --tpc 2
  python code/experiments/temp_scale_sweep.py --tpc 4 --temps 2.0 1.0 0.5
"""

import os
import sys
import argparse

_SELF_DIR = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.dirname(os.path.dirname(_SELF_DIR))
for _sub in ["code/hdm", "code/channel", "code/evaluation", "code/utils", "code/experiments"]:
    sys.path.insert(0, os.path.join(BASE, _sub))
sys.path.insert(0, _SELF_DIR)

import torch  # noqa: F401  (imported for parity with training env init)
from reproducibility import set_seed
from train_han_mlp import evaluate_policy
from lam_intent_demo import load_checkpoint


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--tpc", type=int, default=2)
    p.add_argument("--episodes", type=int, default=200)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--temps", type=float, nargs="+",
                   default=[2.0, 1.5, 1.0, 0.5])
    return p.parse_args()


def main():
    args = get_args()
    # Seed BEFORE constructing the env: CSCA/BS/relay positions are frozen at
    # construction (sim_channel.py:365-380). Training did set_seed(42) then built
    # the env, so we must too -- otherwise the checkpoint is evaluated on
    # off-distribution geometry (the 0.828-vs-0.557 gap).
    set_seed(42)
    trainer = load_checkpoint(args.tpc)

    if not hasattr(trainer.actor, "temp_scale"):
        print("actor has no temp_scale buffer; this sweep applies only to the "
              "DDPM policy (POLICY='ddpm').")
        return

    orig = float(trainer.actor.temp_scale.item())
    print(f"\ntemp_scale sweep  tpc={args.tpc}  episodes={args.episodes}  "
          f"seed={args.seed}")
    print(f"checkpoint's own temp_scale = {orig:.3f}  (this is what the reported "
          f"number used; overridden per row below)\n")
    print(f"  {'temp_scale':>10}  {'ISR':>7}  {'std':>6}  {'delay':>7}  {'dist':>7}")
    print("  " + "-" * 46)

    rows = []
    for ts in args.temps:
        set_seed(args.seed)                       # identical eval states each row
        trainer.actor.temp_scale.fill_(ts)
        isr, isr_std, delay, dist = evaluate_policy(
            trainer, n_episodes=args.episodes)
        rows.append((ts, isr, isr_std, delay, dist))
        print(f"  {ts:>10.2f}  {isr:>7.4f}  {isr_std:>6.3f}  "
              f"{delay:>7.3f}  {dist:>7.4f}")

    print("  " + "-" * 46)
    trainer.actor.temp_scale.fill_(orig)          # restore, leave nothing mutated

    best = max(rows, key=lambda r: r[1])
    worst = min(rows, key=lambda r: r[1])
    spread = best[1] - worst[1]
    print(f"\n  best   temp_scale={best[0]:.2f}  ISR={best[1]:.4f}")
    print(f"  worst  temp_scale={worst[0]:.2f}  ISR={worst[1]:.4f}")
    print(f"  spread = {spread:+.4f}")
    if spread < 0.02:
        print("  => ISR ~flat across temp_scale: the anneal is NOT the "
              "mechanism. Look elsewhere.")
    elif best[0] > worst[0]:
        print("  => ISR rises with temp_scale: the 0.5 anneal endpoint HURTS. "
              "Fix = floor temp_scale higher, then one retrain.")
    else:
        print("  => ISR rises as temp_scale drops: sharper allocation is "
              "better here; the anneal is not the problem.")


if __name__ == "__main__":
    main()
