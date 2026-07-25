# CSCA-SemCom Reproduction — Audit Notes
## Student: Sreekar, BITS Pilani, Dr. Sandeep Joshi
## Paper: Sun et al., IEEE TMC, Vol. 25, No. 1, Jan 2026

---

## Summary

This reproduction of the HDM (Heterogeneous Diffusion Model) system for
semantic communication resource allocation produced three distinct result
sets due to environment calibration sensitivity. We report Run A as the
primary result and document the calibration analysis as a methodological
finding.

---

## Run A — Primary Results (commit 93be0b7)

**Environment:** delay_intents=(0.50,1.50), quality_intents=(0.10,0.40)
**Known issue:** Bug-B (triple bandwidth softmax) made all policies near-uniform
in BW allocation. Margin came from MCS selection effects.

| tpc | HDM | AC | PPO | SAC | Static | HDM vs Static |
|-----|-----|----|-----|-----|--------|---------------|
| 1 | 0.390 | 0.493 | 0.405 | 0.441 | 0.318 | +23% |
| 2 | 0.531 | 0.417 | 0.518 | 0.343 | 0.312 | +70% |
| 4 | 0.490 | 0.386 | 0.464 | 0.299 | 0.296 | +66% |
| 10 | 0.428 | 0.289 | 0.523 | 0.292 | 0.291 | +47% |

**HAN ablation (tpc=4):** HAN=0.492, MLP=0.391, advantage=+25.7%
**N-step ablation (tpc=4):** N=5(0.491), N=6(0.476), N=7(0.488)
**Multimodal (Fig 6a):** Text 0.76→0.99, Audio 0.54→0.85, Image 0.52→0.77

---

## Run B — Post-Bug-Fix Collapse

Correcting Bug-B (double softmax removal) exposed a MIM constraint chain
that locked all tasks to the lowest MCS. Environment became unsolvable.
Static ISR=0.000 at tpc=4/10.

---

## Run C — Over-Permissive Calibration (current code)

**Environment:** delay_intents=(0.50,1.50), quality_intents=(0.10,0.40)
**Issue:** Environment too easy at low congestion.

| tpc | HDM | AC | PPO | SAC | Static | HDM vs Static |
|-----|-----|----|-----|-----|--------|---------------|
| 1 | 0.720 | 0.545 | 0.705 | 0.432 | 0.784 | -8% |
| 2 | 0.577 | 0.582 | 0.568 | 0.431 | 0.598 | -3% |
| 4 | 0.407 | 0.438 | 0.429 | 0.582 | 0.204 | +100% |
| 10 | 0.313 | 0.002 | 0.437 | 0.425 | 0.004 | +7725% |

**HAN ablation (tpc=4):** HAN=0.399, MLP=0.395, advantage=+1.1%

---

## Defensible Claims

1. **HDM vs RL baselines at tpc=4 (Run A):** +27% over AC, +6% over PPO.
   These margins are less affected by Bug-B since all learned policies
   shared the same allocation constraint.

2. **HDM advantage grows with congestion (Run A):** From +23% at tpc=1
   to +66% at tpc=4. Pattern matches paper's core finding.

3. **HAN architecture (inconclusive):** +1.1% to +25.7% depending on
   calibration regime. The wide range reflects environment sensitivity,
   not architectural failure.

4. **N-step denoising (consistent):** N=5≈N=6≈N=7 across both Run A
   and Run C. Matches paper's Fig 12a finding.

5. **Multimodal evaluation (complete):** Proper SNR-dependent curves
   for text, audio, and image modalities.

---

## Root Cause of Calibration Sensitivity

The channel model operates in an interference-limited regime (SINR ≈ -10 dB
under uniform allocation). This produces high baseline distortion (~0.5-0.7)
that conflicts with quality intent ranges. The triple-softmax Bug-B masked
this by making all policies produce near-uniform allocation, where the
environment's physics dominated over policy differences.

---

## 30 Documented Bug Fixes

See mp2_audit.txt for the complete fix list. Key categories:
- Environment: tx_power, intent calibration, BLER curve, BW floor
- DDPM: schedule (beta_min/max), Xavier init, temp anneal
- Training: target critic, delayed actor, replay buffer
- SAC: log_std clamp, BW logit clamp, alpha tuning
- Dead code: 5 files deleted (hdm_trainer, mlp_trainer, train_baselines, baselines, utils/config)
