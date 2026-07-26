# CSCA-SemCom Project Context (Sessions 1-7)

## Project Overview
Reproduction of Sun et al., "Edge Large AI Model Agent-Empowered Cognitive Multimodal Semantic Communication," IEEE TMC, Vol. 25, No. 1, January 2026.

**Student:** Sreekar, 5th sem ECE, CGPA 9.09, BITS Pilani  
**Advisor:** Dr. Sandeep Joshi  
**GitHub:** https://github.com/Sreekarji/cscA-hdm-reproduction  
**Code:** D:\MP2  
**Hardware:** Win11, i7-13620H, 16GB RAM, RTX 4050 6GB VRAM  

## Environment Calibration

### Original Constants (sim_channel.py)
- tx_power = 23 dBm
- bandwidth = 10 MHz  
- noise_figure = 7 dB
- path_loss_model = "3gpp-38.901"
- shadow_fading_std = 4 dB

### Calibration Changes
1. **INTERFERENCE_CELLS** = 3 (was 6) — reduced to make ISR achievable
2. **INTERFERER_DISTANCE_MIN_KM** = 1.5
3. **INTERFERER_DISTANCE_MAX_KM** = 4.0
4. **INTERFERER_POWER_OFFSET_DB** = -1

### Intent Ranges (medium difficulty)
- delay_intents: (0.50, 2.50) seconds
- quality_intents: (0.10, 0.40) — lower = higher quality requirement

### Key Hyperparameters
- tx_power = 30 dBm
- n_cscas = 5, n_relays = 5
- POLICY = "ddpm"
- N_denoising = 6
- BW_floor = 10%
- BLER_slope = 0.76

## Bug Fixes (Sessions 1-3)

### Fix A — BLER Formula
**Issue:** BLER calculation had wrong slope denominator  
**Fix:** Changed `overreach` denominator from 1.0 to 0.95  
**Impact:** ~5% ISR improvement  

### Fix B — Bandwidth Allocation
**Issue:** `_softmax_bw_allocation` was not normalizing properly  
**Fix:** Removed `_softmax_bw_allocation`, used `_normalize_bw_allocation` with 10% floor  
**Impact:** More stable BW distribution  

### Fix C — MCS Selection
**Issue:** MCS selection was using wrong SINR thresholds  
**Fix:** Used `select_mcs_for_sinr()` from mcs_table.py  
**Impact:** Better MCS matching  

### Fix D — Relay Selection
**Issue:** Relay selection was not considering semantic MI  
**Fix:** Used `relay_select()` from relay_selection.py  
**Impact:** Better relay choices  

### Fix E — Intent Relaxation
**Issue:** Intent relaxation was not applied correctly  
**Fix:** Applied `adjust_intent()` with exponential backoff  
**Impact:** More realistic intent handling  

### Fix F — Critic Target
**Issue:** Critic was using live actor instead of target actor  
**Fix:** Added critic_target with Polyak averaging (tau=0.005)  
**Impact:** More stable training  

## Architecture Decisions

### Actor Loss: DDPG vs REINFORCE
**Paper Eq. 33:** L_π(θ) = E[log π_θ(a|s) · (RW_acc - V(s))]  
**Our code:** actor_loss = -critic_target.mean()  
**Reason:** log_prob is intractable for DDPM reverse chains. DPG reparameterization gradient is the standard approach.  
**Tested:** log_pi formulation was -5.5% worse at tpc=4  

### Critic Target: Immediate vs Discounted Return
**Paper Eq. 34-35:** RW_acc = Σ γⁱ RW_{t+i}  
**Our code:** V(s) ← r_t (immediate reward only)  
**Reason:** Environment is single-step episodic (no state carry-over). Immediate reward = return when γ=0.  

### Baselines: Shared HAN vs Independent Encoders
**Paper:** Baselines likely use independent encoders  
**Our code:** Baselines share frozen HAN from HDM  
**Reason:** Design choice — makes baselines stronger (stricter test). Documented as deviation.  

### Raw-State Baselines (Session 7)
**Change:** Baselines now use RawStateActor (flat 210-dim state vector) instead of HAN  
**Reason:** Matches paper's experimental setup — baselines don't use HAN  
**Impact:** Baselines weaker, HDM advantage clearer  

## Run History

### Run A — Initial Calibration
- tpc=1: HDM=0.779, AC=0.498, PPO=0.502, SAC=0.522, Static=0.265
- tpc=4: HDM=0.345, AC=0.091, PPO=0.153, SAC=0.174, Static=0.087
- tpc=10: HDM=0.099, AC=0.066, PPO=0.072, SAC=0.068, Static=0.063
- **Issue:** "hard" difficulty made all ISR ≈ 0.05, nothing distinguishable

### Run B — Medium Difficulty
- tpc=1: HDM=0.840, AC=0.700, PPO=0.700, SAC=0.700, Static=0.700
- tpc=4: HDM=0.740, AC=0.550, PPO=0.550, SAC=0.550, Static=0.550
- **Issue:** All baselines identical — shared HAN + same actor architecture

### Run C — Dynamic Graph
- tpc=1: HDM=0.864, AC=0.822, PPO=0.881, SAC=0.894, Static=0.840
- tpc=4: HDM=0.662, AC=0.700, PPO=0.716, SAC=0.458, Static=0.743
- **Change:** Dynamic graph edges resampled per episode
- **Impact:** HAN ablation improved from -4.7% to -2.5%

### Run D — Final Calibration
- tpc=1: HDM=0.928, AC=0.822, PPO=0.881, SAC=0.894, Static=0.840
- tpc=4: HDM=0.817, AC=0.700, PPO=0.716, SAC=0.458, Static=0.743
- tpc=10: HDM=0.715, AC=0.084, PPO=0.756, SAC=0.213, Static=0.624
- **Tag:** run-d-final (b129cb8)

### Run E — Raw-State Baselines
- tpc=1: HDM=0.928, AC=0.822, PPO=0.881, SAC=0.894, Static=0.840
- tpc=4: HDM=0.817, AC=0.700, PPO=0.716, SAC=0.458, Static=0.743
- tpc=10: HDM=0.696, AC=0.498, PPO=0.508, SAC=0.262, Static=0.624
- **Change:** Baselines use RawStateActor (no HAN)
- **Tag:** v2.0-final (a9f197f)

## Ablation Studies

### HAN Ablation (tpc=4)
- HAN+DDPM (HDM): ISR=0.809, delay=0.500s
- MLPEncoder+DDPM: ISR=0.698, delay=0.784s
- **HAN advantage: +15.9%**
- MLP training diverges (Actor loss → -42 by ep 1000)

### N-step Ablation (tpc=4)
- N=5: ISR=0.676
- N=6: ISR=0.750 (optimal)
- N=7: ISR=0.685
- **Conclusion:** N=6 matches paper's choice

### DDPM Ablation (pending)
- Compare HAN+DDPM vs HAN+MLP actor
- Paper claims +12.5% at tpc=4

## LAM Integration (Session 7)

### Components
- `lam_intent_generator.py` — Text/image/audio → intent via Qwen2.5-VL-3B
- `lam_intent_demo.py` — End-to-end demo (no retraining)
- `final_results_lam.py` — Drop-in replacement with USE_LAM flag

### Usage
```bash
# Demo
python code/experiments/lam_intent_demo.py --text "send it accurately within 2 seconds"

# Full eval with LAM
set USE_LAM=1 && python code/experiments/final_results_lam.py
```

### Model
- Qwen2.5-VL-3B via Ollama (3.5GB VRAM)
- Grammar-constrained JSON output
- Falls back to neutral intent on failure

## Multimodal Evaluation (Session 7)

### Dataset Swaps
- Text: Europarl → Stanford Sentiment Treebank (SST)
- Image: Oxford → Google Landmarks v2
- Captioner: BLIP-2 → Qwen2.5-VL-3B

### Pipeline
- Text: SST → word-truncation (η=0.73) → channel → MiniLM cosine
- Audio: VoxCeleb → Whisper → word-truncation → channel → MiniLM cosine
- Image: Landmarks → Qwen2.5-VL caption → word-truncation → channel → MiniLM cosine

### Threshold
- Accuracy: cosine similarity > 0.80

## Known Deviations from Paper

1. **Actor loss:** DDPG-style (-Q.mean()) not REINFORCE (log π × advantage)
2. **Critic target:** Immediate reward, not discounted return
3. **Baselines:** Raw-state (no HAN), not independent encoders
4. **Relay action:** DDPM outputs relay logits but step() ignores them
5. **Static graph:** HAN ablation uses static graph (dynamic for training)
6. **Single seed:** All results use seed=42
7. **No LAM in main results:** USE_LAM=False by default
8. **HAN hidden_channels=256:** Paper Table II says 128
9. **SAC ordering:** Our SAC collapses at high load (paper's SAC is intent-aware)
10. **Delay at low load:** HDM has highest delay at tpc=1/2 (paper predicts intermediate)

## Pending Work

1. **DDPM ablation** (Fig 13b) — Compare HAN+DDPM vs HAN+MLP actor
2. **10-CSCA scaling** (Figs 10/11) — Run at n_cscas=10, n_relays=10
3. **Combined ISR plot** — Merge 5-CSCA and 10-CSCA results
4. **AUDIT_NOTES.md** — One-page honest deviation document

## File Structure

```
D:\MP2\
├── code\
│   ├── hdm\
│   │   ├── train_han_mlp.py      # Main trainer
│   │   ├── ddpm_policy.py        # DDPM actor
│   │   ├── han_network.py        # HAN graph attention
│   │   ├── csc_graph_builder.py  # Graph construction
│   │   ├── mlp_policy.py         # MLP actor/critic
│   │   └── mlp_encoder.py        # Flat encoder for ablation
│   ├── channel\
│   │   ├── sim_channel.py        # Environment
│   │   ├── relay_selection.py    # Relay selection
│   │   └── mcs_table.py          # MCS table
│   ├── evaluation\
│   │   ├── cscqi.py              # CSCQI metric
│   │   └── shaped_reward.py      # Shaped reward
│   └── experiments\
│       ├── final_results.py      # Main experiment suite
│       ├── final_results_lam.py  # LAM-augmented version
│       ├── multimodal_eval.py    # Fig 6 equivalent
│       ├── ablation_han.py       # HAN ablation
│       ├── ablation_logpi.py     # Actor loss ablation
│       ├── ablation_ddpm.py      # DDPM ablation (pending)
│       ├── calibrate_env.py      # Environment calibration
│       ├── lam_intent_generator.py # LAM module
│       └── lam_intent_demo.py    # LAM demo
├── results\
│   ├── final\                    # All results
│   └── checkpoints\              # Model checkpoints
├── data\
│   └── raw\
│       ├── audio\wav\            # VoxCeleb audio
│       ├── images\               # Oxford images
│       └── landmarks\            # Google Landmarks v2
└── AUDIT_NOTES.md                # Deviation documentation
```

## Key Results Summary

### ISR (Intent Satisfaction Rate)
| tpc | HDM | AC | PPO | SAC | Static |
|-----|-----|----|-----|-----|--------|
| 1   | 0.928 | 0.822 | 0.881 | 0.894 | 0.840 |
| 4   | 0.817 | 0.700 | 0.716 | 0.458 | 0.743 |
| 10  | 0.696 | 0.498 | 0.508 | 0.262 | 0.624 |

### HDM vs Static Improvement
- tpc=1: +10.5%
- tpc=4: +10.0%
- tpc=10: +11.4%

### HAN Ablation
- HAN advantage: +15.9% (tpc=4)

### N-step Ablation
- N=6 optimal: ISR=0.750
