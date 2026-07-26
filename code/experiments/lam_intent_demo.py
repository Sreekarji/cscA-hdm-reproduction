#!/usr/bin/env python3
"""
lam_intent_demo.py — End-to-end LAM intent cognition demo.

Pipeline:
  user text -> Qwen2.5-VL-3B (Ollama) -> [delay_s, quality] -> HAN+DDPM -> allocation

Does NOT retrain. Loads existing checkpoint, feeds it LAM-parsed intents.

Run:
  python code/experiments/lam_intent_demo.py
  python code/experiments/lam_intent_demo.py --text "stream now, 0.5 seconds"
  python code/experiments/lam_intent_demo.py --tpc 2
"""

import os
import sys
import argparse

_SELF_DIR = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.dirname(os.path.dirname(_SELF_DIR))
for _sub in ["code/hdm", "code/channel", "code/evaluation", "code/utils", "code/experiments"]:
    sys.path.insert(0, os.path.join(BASE, _sub))
sys.path.insert(0, _SELF_DIR)

import torch
import numpy as np

from train_han_mlp import (
    HANMLPTrainer, sample_eval_state, intents_from_state,
    parse_action, _task_metrics, DEVICE, POLICY, CHECKPOINT_PATH,
)
from lam_intent_generator import parse_intent_from_text, normalize_intent

def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--text", type=str, default="send the data accurately within 2 seconds")
    p.add_argument("--tpc", type=int, default=4)
    p.add_argument("--model", type=str, default="qwen2.5vl:3b")
    return p.parse_args()

_HAN_KEYS   = ["han", "han_state_dict", "han_network", "model"]
_ACTOR_KEYS = ["actor", "actor_state_dict", "policy", "ddpm_actor", "mlp_actor"]

def _find_key(ckpt, candidates, label):
    for k in candidates:
        if k in ckpt:
            return k
    raise KeyError(f"No {label} key in checkpoint. Keys: {list(ckpt.keys())}")

def load_checkpoint(tpc):
    trainer = HANMLPTrainer(tasks_per_csca=tpc, difficulty="medium")
    ckpt_path = os.path.join(str(CHECKPOINT_PATH), f"han_{POLICY}_tpc{tpc}_best.pt")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f"\nCheckpoint not found: {ckpt_path}\n"
            f"Run final_results.py (or train_han_mlp.py) first."
        )
    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    han_key   = _find_key(ckpt, _HAN_KEYS,   "HAN")
    actor_key = _find_key(ckpt, _ACTOR_KEYS, "actor")
    trainer.han.load_state_dict(ckpt[han_key])
    trainer.actor.load_state_dict(ckpt[actor_key])
    trainer.han.eval()
    trainer.actor.eval()
    isr_str = f"ISR={ckpt['isr']:.3f}" if "isr" in ckpt else ""
    ep_str  = f"ep={ckpt['episode']}"  if "episode" in ckpt else ""
    print(f"[demo] Loaded checkpoint  tpc={tpc}  {isr_str}  {ep_str}")
    return trainer

def inject_lam_intents(state, delay_s, quality, n_tasks):
    state["SCt"]["delay_intents"]   = [delay_s]  * n_tasks
    state["SCt"]["quality_intents"] = [quality]  * n_tasks
    for i in range(n_tasks):
        ds_norm = min(state["SCt"]["data_sizes"][i] / 6e5, 1.0)
        di      = delay_s / 10.0
        qi      = quality
        urgency = (1.0 - di) * 0.5 + qi * 0.5
        state["SCt"]["message_features"][i] = [ds_norm, di, qi, urgency]
    return state

def _call_actor(actor, graph_emb, msg_embs):
    cls = type(actor).__name__
    if "DDPM" in cls or "Diffusion" in cls:
        return actor(graph_emb, message_embs=msg_embs, deterministic=True)
    return actor(graph_emb, message_embs=msg_embs)

def run_demo(trainer, delay_s, quality):
    n_tasks  = trainer.n_tasks
    n_relays = trainer.n_relays
    n_mcs    = trainer.n_mcs

    state = sample_eval_state(trainer.env)
    state = inject_lam_intents(state, delay_s, quality, n_tasks)

    d_arr   = np.array(state["SCt"]["delay_intents"])
    d_range = d_arr.max() - d_arr.min() + 1e-8
    urgency = 1.0 - (d_arr - d_arr.min()) / d_range
    q_arr   = np.array(state["SCt"]["quality_intents"])
    intent_vectors = np.stack([urgency, q_arr], axis=1).tolist()

    with torch.no_grad():
        graph_emb, _, msg_embs = trainer.han.encode_state(
            state, intent_vectors=intent_vectors)
        action = _call_actor(trainer.actor, graph_emb, msg_embs)
        parsed = parse_action(action, n_tasks, n_relays, n_mcs)
        result = trainer.env.step(parsed, state)

    isr, delay_out, dist_out = _task_metrics(result["tasks"])
    bw_vec  = parsed["bandwidth"][0].cpu().numpy()
    relay_v = parsed["relay"][0].cpu().numpy()
    mcs_v   = parsed["mcs"][0].cpu().numpy()
    mcs_labels = ["Low (QPSK)", "Mid (16QAM)", "High (64QAM)"]

    print("\n" + "=" * 62)
    print("  ALLOCATION RESULT")
    print("=" * 62)
    print(f"  Input intent   : delay <= {delay_s:.2f} s  |  quality >= {quality:.2f}")
    print(f"  ISR            : {isr:.4f}")
    print(f"  Mean delay     : {delay_out:.3f} s")
    print(f"  Mean distortion: {dist_out:.4f}")
    print("-" * 62)
    print(f"  {'Task':>4}  {'BW share':>10}  {'Top relay':>10}  {'MCS':>14}")
    print("-" * 62)
    for i in range(n_tasks):
        top_relay  = int(np.argmax(relay_v[i]))
        mcs_bucket = int(np.argmax(mcs_v[i]))
        print(f"  {i:>4}  {bw_vec[i]:>10.4f}  {top_relay:>10}  "
              f"{mcs_labels[min(mcs_bucket, 2)]:>14}")
    print("=" * 62)

def main():
    args = get_args()
    print(f"\n[demo] Parsing intent with LAM ({args.model}) ...")
    raw_d, raw_q = parse_intent_from_text(args.text, model=args.model)
    delay_s, quality = normalize_intent(raw_d, raw_q)
    print(f"[demo] Text           : {args.text!r}")
    print(f"[demo] Raw LAM output : delay={raw_d:.3f} s  quality={raw_q:.3f}")
    print(f"[demo] Normalised     : delay={delay_s:.3f} s  quality={quality:.3f}")
    trainer = load_checkpoint(args.tpc)
    run_demo(trainer, delay_s, quality)

if __name__ == "__main__":
    main()
