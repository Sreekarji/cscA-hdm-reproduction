#!/usr/bin/env python3
"""
e2e_demo.py — Full CSCA pipeline on real per-task multimodal inputs.

  text / image / audio  ->  LAM left brain (self-adaptive RAG intent cognition)
                        ->  MSS semantic compression (paper Algorithm 1)
                        ->  HDM right brain (HAN + DDPM allocation)
                        ->  channel simulation -> ISR / delay / distortion

Unlike lam_intent_demo.py, which parses ONE sentence and broadcasts that intent
to every task, this script accepts a DIFFERENT input per task. That is the
configuration the paper describes: heterogeneous tasks with heterogeneous
intents contending for the same bandwidth, which is the only setting where the
HAN's per-message attention has anything to attend to.

Does NOT train. Loads an existing checkpoint.

Run:
  python code/experiments/e2e_demo.py --text "stream now, half a second" \
                                      --text "medical scan, lossless" \
                                      --image data/raw/landmarks/x.jpg \
                                      --audio data/raw/audio/wav/y.wav
  python code/experiments/e2e_demo.py --tpc 2 --no-rag
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
    intents_from_state, parse_action, _task_metrics, sample_eval_state,
)
from lam_intent_demo import load_checkpoint, _call_actor
from lam_intent_generator import (
    parse_intent_from_text, parse_intent_from_image, parse_intent_from_audio,
    normalize_intent,
)
from mss_algorithm import minimum_synonymous_subsequence, DEFAULT_EPSILON
from sim_channel import normalise_intents

WHISPER_DIR = os.path.join(BASE, "models", "whisper")


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--text",  action="append", default=[],
                   help="repeatable; one per task")
    p.add_argument("--image", action="append", default=[])
    p.add_argument("--audio", action="append", default=[])
    p.add_argument("--tpc", type=int, default=4)
    p.add_argument("--model", type=str, default="qwen2.5vl:3b")
    p.add_argument("--mss-epsilon", type=float, default=DEFAULT_EPSILON)
    p.add_argument("--no-rag", action="store_true",
                   help="bypass the self-adaptive RAG loop (ablation)")
    return p.parse_args()


def collect_inputs(args, n_tasks):
    """Interleave the provided inputs, then cycle to fill all n_tasks slots."""
    items = ([("text", t)  for t in args.text]
             + [("image", p) for p in args.image]
             + [("audio", p) for p in args.audio])
    if not items:
        items = [
            ("text", "stream the live feed right now, I cannot wait"),
            ("text", "send the data accurately within two seconds"),
            ("text", "this is a medical image, accuracy is critical"),
            ("text", "casual photo, whatever is fastest"),
        ]
        print("[e2e] No inputs given; using built-in demo sentences.")
    return [items[i % len(items)] for i in range(n_tasks)]


def parse_one(kind, payload, model, use_rag):
    if kind == "text":
        return parse_intent_from_text(payload, model=model, use_rag=use_rag)
    if kind == "image":
        return parse_intent_from_image(payload, model=model)
    return parse_intent_from_audio(payload, whisper_dir=WHISPER_DIR,
                                   lam_model=model)


def compress_one(kind, payload, epsilon):
    """MSS compression for text tasks. Non-text payloads pass through.

    Returns (ratio, shown) where ratio scales the task's payload size and
    shown is what gets printed in the per-task table.
    """
    if kind != "text":
        return 1.0, os.path.basename(payload)
    mss, ratio = minimum_synonymous_subsequence(payload, epsilon=epsilon)
    return ratio, mss


def run(trainer, inputs, args):
    n_tasks  = trainer.n_tasks
    n_relays = trainer.n_relays
    n_mcs    = trainer.n_mcs
    use_rag  = not args.no_rag

    state = sample_eval_state(trainer.env)

    print(f"\n[e2e] Parsing {n_tasks} intents  (RAG={'on' if use_rag else 'off'}) ...")
    parsed_intents, ratios, shown = [], [], []
    for i, (kind, payload) in enumerate(inputs):
        raw_d, raw_q = parse_one(kind, payload, args.model, use_rag)
        d, q = normalize_intent(raw_d, raw_q)
        r, s = compress_one(kind, payload, args.mss_epsilon)
        parsed_intents.append((d, q))
        ratios.append(r)
        shown.append((kind, s))
        print(f"  task {i:>2} [{kind:>5}] delay={d:.2f}s quality={q:.2f} "
              f"mss_ratio={r:.2f}")

    # Write the per-task intents and MSS-shrunk payload sizes into the state so
    # both the HAN and the channel see the same numbers.
    for i in range(n_tasks):
        d, q = parsed_intents[i]
        state["SCt"]["delay_intents"][i]   = d
        state["SCt"]["quality_intents"][i] = q
        state["SCt"]["data_sizes"][i]      = state["SCt"]["data_sizes"][i] * ratios[i]
        ds_norm = min(state["SCt"]["data_sizes"][i] / 6e5, 1.0)
        di, qi  = normalise_intents(d, q)
        urgency = (1.0 - di) * 0.5 + qi * 0.5
        state["SCt"]["message_features"][i] = [ds_norm, di, qi, urgency]

    intent_vectors = intents_from_state(state)

    with torch.no_grad():
        graph_emb, _, msg_embs = trainer.han.encode_state(
            state, intent_vectors=intent_vectors)
        action = _call_actor(trainer.actor, graph_emb, msg_embs)
        parsed = parse_action(action, n_tasks, n_relays, n_mcs)
        result = trainer.env.step(parsed, state)

    isr, delay_out, dist_out = _task_metrics(result["tasks"])
    bw_vec = parsed["bandwidth"][0].cpu().numpy()
    mcs_v  = parsed["mcs"][0].cpu().numpy()
    mcs_labels = ["Low (QPSK)", "Mid (16QAM)", "High (64QAM)"]

    print("\n" + "=" * 78)
    print("  END-TO-END RESULT")
    print("=" * 78)
    print(f"  ISR            : {isr:.4f}")
    print(f"  Mean delay     : {delay_out:.3f} s")
    print(f"  Mean distortion: {dist_out:.4f}")
    print(f"  Mean MSS ratio : {float(np.mean(ratios)):.3f}")
    print("-" * 78)
    print(f"  {'Task':>4} {'Mode':>6} {'Delay int':>10} {'Qual int':>9} "
          f"{'BW':>8} {'MCS':>13}")
    print("-" * 78)
    for i in range(n_tasks):
        d, q = parsed_intents[i]
        print(f"  {i:>4} {shown[i][0]:>6} {d:>10.2f} {q:>9.2f} "
              f"{bw_vec[i]:>8.4f} {mcs_labels[min(int(np.argmax(mcs_v[i])), 2)]:>13}")
    print("=" * 78)
    for i in range(n_tasks):
        if shown[i][0] == "text":
            print(f"  task {i:>2} MSS: {shown[i][1]!r}")


def main():
    args = get_args()
    trainer = load_checkpoint(args.tpc)
    inputs = collect_inputs(args, trainer.n_tasks)
    run(trainer, inputs, args)


if __name__ == "__main__":
    main()
