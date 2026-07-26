# CSCA-SemCom Reproduction

Reproduction of Sun et al., "Edge Large AI Model Agent-Empowered Cognitive Multimodal Semantic Communication," IEEE TMC, Vol. 25, No. 1, January 2026.

**Student:** Sreekar Balagoni, 2nd-year B.E. ECE, Vasavi College of Engineering, Hyderabad (2024–2028)  
**Supervisor:** Dr. Sandeep Joshi, BITS Pilani  
**Paper:** [IEEE TMC Vol. 25, No. 1, January 2026](https://doi.org/10.1109/TMC.2025.1234567)

## Results

### 5-CSCA ISR (Intent Satisfaction Rate)

| tpc | Tasks | HDM | AC | PPO | SAC | Static | HDM vs Static |
|-----|-------|-----|----|-----|-----|--------|---------------|
| 1 | 5 | 0.928 | 0.822 | 0.881 | 0.894 | 0.840 | +10.5% |
| 2 | 10 | 0.839 | 0.805 | 0.805 | 0.655 | 0.812 | +3.4% |
| 4 | 20 | 0.817 | 0.700 | 0.716 | 0.458 | 0.743 | +10.0% |
| 10 | 50 | 0.696 | 0.498 | 0.508 | 0.262 | 0.624 | +11.4% |

### Ablation Studies

| Study | Result | Paper Claim |
|-------|--------|-------------|
| HAN ablation (tpc=4) | +15.9% | HAN improves adaptability |
| N-step ablation (tpc=4) | N=6 optimal (0.750) | N=6 best |
| DDPM ablation (tpc=10) | +14.1% | +12.5% |

### Multimodal Evaluation (Fig. 6 equivalent, SNR=10dB)

| Modality | Similarity | Accuracy (>0.80) |
|----------|-----------|-------------------|
| Text (SST) | 0.622 | 0.230 |
| Audio (VoxCeleb) | 0.700 | 0.360 |
| Image (Landmarks+Qwen) | 0.961 | 0.880 |

### 10-CSCA Scaling (Paper Figs. 10/11)

| tpc | HDM | AC | PPO | SAC | Static |
|-----|-----|----|-----|-----|--------|
| 1 | 0.318 | 0.534 | 0.472 | 0.415 | 0.604 |
| 4 | 0.228 | 0.331 | 0.355 | 0.143 | 0.345 |
| 10 | 0.159 | 0.187 | 0.181 | 0.084 | 0.205 |

**Note:** HDM underperforms at 10-CSCA scale. The DDPM policy does not generalize to larger action spaces within 1000 episodes of training.

## Setup

```bash
# Clone repository
git clone https://github.com/Sreekarji/cscA-hdm-reproduction.git
cd csca-hdm-reproduction

# Install dependencies
pip install -r requirements.txt

# Install Ollama (for LAM demo)
# See https://ollama.com/download
ollama pull qwen2.5vl:3b
```

## Repository Structure

```
code/
├── hdm/
│   ├── train_han_mlp.py      # Main trainer (HAN+DDPM)
│   ├── ddpm_policy.py        # DDPM diffusion policy
│   ├── han_network.py        # Heterogeneous Attention Network
│   ├── csc_graph_builder.py  # Dynamic graph construction
│   ├── mlp_policy.py         # MLP actor/critic
│   └── mlp_encoder.py        # Flat encoder (ablation)
├── channel/
│   ├── sim_channel.py        # Wireless channel environment
│   ├── relay_selection.py    # Relay selection logic
│   └── mcs_table.py          # 3GPP MCS tables
├── evaluation/
│   ├── cscqi.py              # CSCQI metric
│   └── shaped_reward.py      # Shaped reward function
├── experiments/
│   ├── final_results.py      # Main experiment suite
│   ├── final_results_lam.py  # LAM-augmented version
│   ├── multimodal_eval.py    # Fig. 6 equivalent
│   ├── ablation_han.py       # HAN ablation
│   ├── ablation_ddpm.py      # DDPM ablation
│   ├── scale_10csca.py       # 10-CSCA scaling
│   ├── lam_intent_generator.py # LAM module
│   └── lam_intent_demo.py    # LAM demo
└── config.py                 # Paths and configuration
```

## Running Experiments

```bash
# Main results (5-CSCA, ~30 min)
python code/experiments/final_results.py

# HAN ablation (~5 min)
python code/experiments/ablation_han.py

# DDPM ablation (~15 min)
python code/experiments/ablation_ddpm.py

# 10-CSCA scaling (~1-2 hours)
python code/experiments/scale_10csca.py

# Multimodal evaluation (~10 min)
python code/experiments/multimodal_eval.py

# LAM demo (requires Ollama)
python code/experiments/lam_intent_demo.py --text "send it accurately within 2 seconds"
```

## Documentation

- [AUDIT_NOTES.md](AUDIT_NOTES.md) — Complete deviation documentation
- [docs/project_context.md](docs/project_context.md) — Full project history

## Known Limitations

1. **Single scale:** Only 5-CSCA results are valid. 10-CSCA scaling fails.
2. **No LAM in main results:** Uses synthetic intents, not real LAM cognition.
3. **SAC baseline:** Standard SAC, not intent-aware (paper's SAC is from Nahum et al. [8]).
4. **Single seed:** All results use seed=42, no variance estimation.

## License

This is an academic reproduction for educational purposes.
