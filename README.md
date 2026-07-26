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
PPO, and Actor-Critic baselines. The LAM left-brain (LLaVA-NeXT-Interleave) is
not part of the main RL runs — intents are drawn synthetically from the paper's
specified distributions. A working LAM demo using Qwen2.5-VL-3B via Ollama is
available separately (see below).

The central qualitative finding reproduces: intent-aware per-task policy
generation strongly outperforms uniform allocation under resource competition.
Three quantitative claims do not reproduce — delay ordering, HAN ablation
magnitude, and 10-CSCA scaling. All deviations are documented with structural
explanations in [`AUDIT_NOTES.md`](AUDIT_NOTES.md).

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

| Study | This repo | Paper claim |
|---|---|---|
| HAN vs flat MLP encoder (tpc=4) | +15.9% ISR | HAN improves with scale |
| DDPM vs MLP actor (tpc=10) | +14.1% ISR | +12.5% |
| N-step denoising (tpc=4) | N=6 approximately optimal | N=6 best |

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
│   ├── train_han_mlp.py      # Main trainer (HAN + DDPM)
│   ├── ddpm_policy.py        # DDPM diffusion policy
│   ├── han_network.py        # Heterogeneous Attention Network
│   ├── csc_graph_builder.py  # Dynamic CSC graph construction
│   ├── mlp_policy.py         # MLP actor/critic (baselines)
│   └── mlp_encoder.py        # Flat encoder (HAN ablation)
├── channel/
│   ├── sim_channel.py        # Wireless channel environment
│   ├── relay_selection.py    # Relay selection logic
│   └── mcs_table.py          # 3GPP MCS tables
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
    ├── lam_intent_generator.py # LAM intent module
    └── lam_intent_demo.py    # End-to-end LAM demo

results/final/                # All CSVs, PNGs, SUMMARY.txt
AUDIT_NOTES.md                # Complete deviation documentation
```

---

## Running Experiments

```bash
# Main results — 5 CSCA, ~30 min
python code/experiments/final_results.py

# Ablations
python code/experiments/ablation_han.py       # ~5 min
python code/experiments/ablation_ddpm.py      # ~15 min

# 10-CSCA scaling — ~1-2 hours
python code/experiments/scale_10csca.py

# Multimodal evaluation — ~10 min
python code/experiments/multimodal_eval.py

# LAM demo — requires Ollama
python code/experiments/lam_intent_demo.py --text "send it accurately within 2 seconds"
```

Outputs are written to `results/final/`. `results/final/SUMMARY.txt` is the
authoritative append-only experiment log.

---

## Known Limitations

- **LAM not in main RL loop.** Intents are synthetic. Intent *cognition* is
  not evaluated, only intent *satisfaction* given ground-truth intents.
- **Simplified channel model.** 3 interference cells vs paper's 6; no full
  3GPP TR 38.901 cluster model. Affects 10-CSCA results most.
- **Single training seed.** Reported ± values are across evaluation episodes,
  not across training runs. Total variance is understated.
- **Static baseline is fixed QPSK.** Not an SINR-adaptive scheduler. At low
  load it is near-optimal; at high load it is weak. Both facts apply when
  reading the +54%/+704% figures.
- **Relay excluded from action space.** Relay selection is a hand-coded
  heuristic applied equally to all policies.

Full deviation list with impact assessment: [`AUDIT_NOTES.md`](AUDIT_NOTES.md)

---

## License

Academic reproduction for educational purposes.
