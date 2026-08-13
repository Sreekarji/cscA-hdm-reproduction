# CSCA-SemCom Reproduction

Reproduction of: Y. Sun, Y. Liu, S. Guo, X. Qiu, J. Chen, J. Hao, D. Niyato,
"Edge Large AI Model Agent-Empowered Cognitive Multimodal Semantic
Communication," *IEEE Transactions on Mobile Computing*, Vol. 25, No. 1,
Jan. 2026. DOI: [10.1109/TMC.2025.3590723](https://doi.org/10.1109/TMC.2025.3590723)

**Student:** Sreekar Balagoni, 3rd-year B.E. ECE, Vasavi College of Engineering,
Hyderabad (2024–2028)                                                                            
**Supervisor:** Dr. Sandeep Joshi, Associate Professor, Department of EEE,
BITS Pilani

---

## What this is

The paper proposes CSCA — a cognitive semantic communication agent combining a
Large AI Model (left brain) with a wireless communication planning model (right
brain). The planning model, HDM, uses a Heterogeneous Attention Network (HAN)
to encode system state and a Denoising Diffusion Probabilistic Model (DDPM) to
generate per-task bandwidth, relay, and MCS policies.

This repository re-implements HDM from scratch and evaluates it against SAC,
PPO, and Actor-Critic baselines. The LAM left brain is implemented with
Qwen2.5-VL-3B via Ollama plus a self-adaptive RAG loop over a local knowledge
base, and runs at **inference time only** — RL training uses intents sampled
from the paper's specified distributions (paper Table II), which is what the
paper does.

The central qualitative finding reproduces: intent-aware per-task policy
generation strongly outperforms uniform allocation under resource competition.
Three quantitative claims do not reproduce — delay ordering, HAN ablation
magnitude, and 10-CSCA scaling. All deviations are documented with structural
explanations in [`AUDIT_NOTES.md`](AUDIT_NOTES.md).

> **Results note.** The ISR numbers in the table below are from the committed
> `results/final/isr_vs_tpc.csv` (seed 42, 200 eval episodes, 5 CSCA, medium
> difficulty). They reflect all fixes from `audit_fixes.md` except the dynamic
> topology fix (§4), which raised the HAN ablation from +15.9 % → +18.9 % but
> requires a full retrain to propagate to the main ISR table. Rerun
> `final_results.py` after any checkpoint retrain to refresh these numbers.

---

## Results

### Intent Satisfaction Rate — 5 CSCA (Fig. 9a equivalent)

| tpc | Tasks | HDM   | AC    | PPO   | SAC   | Static | HDM vs Static |
|-----|-------|-------|-------|-------|-------|--------|---------------|
| 1   | 5     | 0.928 | 0.822 | 0.881 | 0.894 | 0.840  | +10.5%        |
| 2   | 10    | 0.839 | 0.805 | 0.805 | 0.655 | 0.812  | +3.4%         |
| 4   | 20    | 0.817 | 0.700 | 0.716 | 0.458 | 0.743  | +10.0%        |
| 10  | 50    | 0.696 | 0.498 | 0.508 | 0.262 | 0.624  | +11.4%        |

HDM outperforms all baselines at tpc ≥ 2. At tpc=1 all policies clear
deadlines easily and uniform QPSK (Static) is near-optimal — learned-policy
advantage emerges only under resource competition.

### Ablation Studies

| Study | This repo | Paper claim | Note |
|---|---|---|---|
| HAN vs flat MLP encoder (tpc=4) | **+18.9%** ISR (dynamic topology) | ~+18.9% | Matches paper after Bug-D fix; old CSV shows +15.87% (static topology) |
| DDPM vs MLP actor (tpc=4) | +14.1% ISR | +12.5% | tpc=4 result; tpc=10 run added in ablation_ddpm.py fix |
| N-step denoising (tpc=4) | N=7 marginally best; N=6≈N=5≈N=7 | N=6 best | All within noise |

### Multimodal Semantic Accuracy (Fig. 6 equivalent, SNR=10 dB)

| Modality | Similarity | Accuracy (sim > 0.80) |
|---|---|---|
| Text (SST) | 0.622 | 23.0% |
| Audio (VoxCeleb + Whisper) | 0.700 | 36.0% |
| Image (Landmarks + Qwen2.5-VL) | 0.961 | 88.0% |

### 10-CSCA Scaling (Paper Figs. 10/11)

| tpc | HDM   | AC    | PPO   | SAC   | Static |
|-----|-------|-------|-------|-------|--------|
| 1   | 0.318 | 0.534 | 0.472 | 0.415 | 0.604  |
| 4   | 0.228 | 0.331 | 0.355 | 0.143 | 0.345  |
| 10  | 0.159 | 0.187 | 0.181 | 0.084 | 0.205  |

HDM underperforms baselines at this scale. The DDPM policy does not generalise
to the larger action space within 1000 training episodes on a single GPU. Full
explanation in AUDIT_NOTES.md Section 5.

---

## Setup

```bash
git clone https://github.com/Sreekarji/cscA-hdm-reproduction.git
cd csca-hdm-reproduction
pip install -r requirements.txt

# For LAM demo only
# Install Ollama: https://ollama.com/download
ollama pull qwen2.5vl:3b
```

Hardware used: RTX 4050 6 GB, i7-13620H, 16 GB RAM, Windows 11.
All results: Python 3.11, PyTorch 2.6, seed 42.

---

## Repository Structure

```
code/
├── hdm/
│   ├── train_han_mlp.py      # Main trainer (HAN + DDPM); USE_RELAY_ACTION flag
│   ├── ddpm_policy.py        # DDPM diffusion policy
│   ├── han_network.py        # Heterogeneous Attention Network
│   ├── csc_graph_builder.py  # CSC graph construction
│   ├── mss_algorithm.py      # Minimum Synonymous Subsequence (paper Alg. 1)
│   ├── mlp_policy.py         # MLP actor/critic (baselines)
│   └── mlp_encoder.py        # Flat encoder (HAN ablation)
├── channel/
│   ├── sim_channel.py        # Wireless channel env; normalise_intents helper
│   ├── relay_selection.py    # Relay selection logic
│   └── mcs_table.py          # 3GPP MCS tables + MIM (Eq. 5)
├── evaluation/
│   ├── cscqi.py              # CSCQI metric (paper Eq. 17)
│   └── shaped_reward.py      # Shaped reward function
└── experiments/
    ├── final_results.py      # Main experiment suite
    ├── final_results_lam.py  # LAM-augmented variant
    ├── multimodal_eval.py    # Fig. 6 equivalent
    ├── ablation_han.py       # HAN vs MLP encoder
    ├── ablation_ddpm.py      # DDPM vs MLP actor
    ├── ablation_logpi.py     # Eq. 33 vs -Q.mean()
    ├── scale_10csca.py       # 10-CSCA scaling
    ├── calibrate_env.py      # Environment calibration
    ├── lam_intent_generator.py # LAM intent parser + self-adaptive RAG
    ├── lam_intent_demo.py    # One intent, broadcast to all tasks
    └── e2e_demo.py           # Per-task text/image/audio, MSS compression

data/lkb/intent_corpus.json   # Local Knowledge Base (RAG exemplars)
results/final/                # All CSVs, PNGs, SUMMARY.txt
AUDIT_NOTES.md                # Complete deviation documentation
```

### Configuration flags

| Flag | File | Default | Effect |
|---|---|---|---|
| `USE_RELAY_ACTION` | `code/hdm/train_han_mlp.py` | `False` | Include relay block in action vector. Off: `action_dim = n_tasks*(1+n_mcs)` (80 at tpc=4). On: `+n_tasks*n_relays` (180). Relay outputs are wired through `parse_action()` but ignored by `sim_channel.step()` which uses the heuristic selector. |
| `USE_ADAPTIVE_RAG` | `code/lam/lam_intent_generator.py` | `True` | Run self-adaptive RAG (retrieve → rerank → ISREL → generate → ISSUP) before intent parsing. Off: single unconditioned LAM call. |
| `LAM_TRAINING_MODE` | `code/config.py` | `False` | When True, `HANMLPTrainer.train()` calls `LAMIntentProvider.get_batch()` each episode and injects real Qwen2.5-VL intents into the state. Requires Ollama. False: synthetic intents (existing results reproducible without Ollama). |
| `USE_ENTROPY_REG` | `code/config.py` | `False` | When True, actor loss = `-Q.mean() - ENTROPY_COEFF * log_pi.mean()` (paper Eq. 33) via `DDPMActor.forward_with_logprob()`. False: plain `-Q.mean()`. |
| `ENTROPY_COEFF` | `code/config.py` | `0.01` | Coefficient λ in the Eq. 33 entropy term. |
| `COMPRESSION_METHOD` | `code/experiments/multimodal_eval.py` | `"eta"` | `"eta"` = fixed 73% truncation (reproducible Fig. 6b); `"mss"` = MSS Algorithm 1. |

---

## Running Experiments

```bash
# Run all offline tests first (no Ollama, no GPU needed, ~30s)
python -m pytest code/tests/ -q

# Main results — 5 CSCA, seed 42, ~30 min
python code/experiments/final_results.py

# Ablations
python code/experiments/ablation_han.py       # HAN vs MLP, tpc=4, ~5 min
python code/experiments/ablation_ddpm.py      # DDPM vs MLP actor, tpc=4+10, ~15 min

# 10-CSCA scaling — ~1-2 hours
python code/experiments/scale_10csca.py

# Multimodal evaluation — ~10 min
python code/experiments/multimodal_eval.py

# LAM demos — require Ollama + qwen2.5vl:3b
python code/experiments/lam_intent_demo.py --text "send it accurately within 2 seconds"
python code/experiments/e2e_demo.py --text "stream now, half a second" \
                                    --text "medical scan, lossless"

# LAM provider smoke test (offline, no Ollama needed)
python code/lam/lam_provider.py --smoke-test

# LAM training mode (Ollama required)
# Edit code/config.py: set LAM_TRAINING_MODE = True
python code/experiments/final_results.py
```

## LAM Integration (lam-integration branch)

The `lam-integration` branch closes the three audit gaps identified in the
Aug 2026 audit:

### Gap 1 — LKB corpus
`data/lkb/intent_corpus.json` — 200 synthetic exemplars (50 per modality:
text, image, audio, multimodal) with `delay_s`, `quality`, `priority` fields.
`lkb_retrieve()` now returns non-empty results; `cohere_rerank()` falls back
to a local MiniLM reranker (instead of silent identity truncation) when
`COHERE_API_KEY` is absent.

### Gap 2 — LAM in the training loop
`code/lam/lam_provider.py` — `LAMIntentProvider` class with:
- `get_intent(task_idx)` / `get_batch(n)` — per-episode intent sampling
- Episode-level cache (20 slots, TTL=50 episodes) so Ollama is not hit every step
- Graceful synthetic fallback when Ollama is unavailable
- `HANMLPTrainer._inject_lam_intents(state)` — rewrites `delay_intents`,
  `quality_intents`, and `data_sizes` (scaled by MSS `compression_ratio`)
- `config.LAM_TRAINING_MODE = False` — default off; flip to `True` to activate

### Gap 3 — `forward_with_logprob` in DDPMActor
`DDPMActor.forward_with_logprob(graph_emb, message_embs)` — native method
(ported from the `ablation_logpi.py` monkey-patch). Returns `(action, log_pi)`
where `log_pi` sums Gaussian log-prob over all stochastic denoising steps.
`config.USE_ENTROPY_REG = False` — default off; `True` activates Eq. 33 loss.

Outputs are written to `results/final/`. `results/final/SUMMARY.txt` is the
authoritative append-only experiment log.

---

## Known Limitations

- **LAM training mode is off by default.** Set `LAM_TRAINING_MODE = True` in
  `code/config.py` and start Ollama with `qwen2.5vl:3b` to activate real-intent
  training. With the default `False` the training loop uses synthetic intents
  (paper Table II distribution) and all committed results are reproducible
  without Ollama. The `LAMIntentProvider` caches 20 slots and refreshes every
  50 episodes to keep Ollama overhead manageable on the RTX 4050.
- **Qwen2.5-VL-3B, not LLaVA-NeXT-Interleave.** Chosen for the 6 GB VRAM budget.
- **Cohere rerank is optional.** Without `COHERE_API_KEY` the RAG reranker
  degrades to embedding order; ISREL/ISSUP reflection still runs.
- **Simplified channel model.** 3 interference cells vs paper's 6; no full
  3GPP TR 38.901 cluster model. Affects 10-CSCA results most.
- **Single training seed.** Reported ± values are across evaluation episodes,
  not across training runs. Total variance is understated.
- **Static baseline is fixed QPSK.** Not an SINR-adaptive scheduler. At low
  load it is near-optimal; at high load it is weak. Both facts apply when
  reading the +54%/+704% figures.
- **Relay excluded from action space** (`USE_RELAY_ACTION = False`). Relay
  selection is a hand-coded heuristic applied equally to all policies.
- **Relay knowledge is thin.** With `RELAY_KNOWLEDGE_P = 0.475` and seed 42,
  the 5 relays cover text/audio/image with 1/3/2 relays respectively — text is
  served by a single relay. Measured `semantic_conn` edge count at 20 messages /
  5 relays is 39.9 (range 29–53) over 500 episodes; an earlier docstring claimed
  45–50, which was never measured and has been corrected. This is the only
  episode-varying edge type in the CSC graph, which bounds how much relational
  structure the HAN has to attend to.
- **MIM is not an MCS constraint.** `compute_mim` (Eq. 5) exists and is used by
  the semantic layer, but feeding it into MCS selection as Eq. 6 constraint C3
  was implemented and reverted — it cost ~24% ISR across all policies. See the
  NOTE at `sim_channel.py` `simulate_channel`.

Full deviation list with impact assessment: [`AUDIT_NOTES.md`](AUDIT_NOTES.md)

---

## License

Academic reproduction for educational purposes.
