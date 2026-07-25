# AUDIT_NOTES.md

Reproduction of: Y. Sun, Y. Liu, S. Guo, X. Qiu, J. Chen, J. Hao, D. Niyato,
"Edge Large AI Model Agent-Empowered Cognitive Multimodal Semantic Communication,"
IEEE Transactions on Mobile Computing, vol. 25, no. 1, pp. 19-36, Jan. 2026.

Implementation: Sreekar Balagoni, ViSRI Lab, BITS Pilani (supervisor: Dr. Sandeep Joshi).
Hardware: RTX 4050 6 GB, i7-13620H, 16 GB RAM, Windows 11.
Software: Python 3.11, PyTorch 2.6. All results: seed 42, 1000 training episodes,
200 evaluation episodes.

This document records every defect found during the audit, every assumption made
where the paper is underspecified, and every result that does not reproduce.
It is written so that a reader can reject any claim in the report by rerunning a
named experiment.

---

## 1. Defects found and fixed

### Bug A — deadline scaling coupled to task count
`MultiCSCAEnvironment.generate_state()` multiplied `delay_intents` by
`max(1, n_tasks / 5)`. At tpc=10 (50 tasks) deadlines were 10x looser than at
tpc=1 (5 tasks). Intent satisfaction rate was consequently non-monotonic in load,
which is the opposite of the paper's Fig. 7 and Fig. 9(a).
**Fix:** intents sampled i.i.d. uniform, independent of `n_tasks`.
**Impact:** invalidates all Run A numbers as a function of tpc.

### Bug B — urgency feature clipped to zero at high load
`intents_from_state()` computed `urgency = (d - 0.05) / (0.60 - 0.05)` with a
hardcoded denominator. For any delay intent d > 0.60 s the clipped result was
zero. At tpc >= 4 every task's urgency feature was identically zero, so the HAN
was structurally blind to per-task deadlines. This is the most serious defect
found: it silently disabled the mechanism the paper's Fig. 13(a) ablation is
designed to measure.
**Fix:** per-episode min/max normalisation over the sampled delay intents.
**Impact:** invalidates the Run A HAN ablation (+25.7%). See Section 4.

### Bug C — intent attenuation sign inversion
`adjust_intent()` in `code/evaluation/cscqi.py` applied `exp(-omega * tau_w)`,
tightening intents under queuing pressure. Paper Eq. (19)-(20) require
`exp(+omega * tau_w)`, i.e. relaxation. The function was dead code (never
called; the correct form is inlined in `sim_channel.step()`), so no reported
number is affected.
**Fix:** deleted.

### Bug D — dead bandwidth allocator
`_softmax_bw_allocation()` was defined but never called; allocation went through
`_normalize_bw_allocation()`. No numerical impact.
**Fix:** deleted, to prevent future misattribution of results.

### Bug E — deterministic semantic type
`csc_graph_builder.build()` assigned `semantic_type = type_map[i % 3]`, making
modality a deterministic function of task index rather than a random draw. The
HAN could memorise modality from node ordering.
**Fix:** randomised per episode.

### Bug F — inconsistent delay-intent normalisation
`generate_state()` wrote `message_features[i][1] = di / 5.0`;
`sample_eval_state()` wrote the same field as `di` unnormalised. Training and
evaluation therefore presented the encoder with differently scaled inputs.
**Fix:** both use `di / 10.0`.

---

## 2. Channel model recalibration (assumption, not a bug)

The paper specifies path loss `128.1 + 37.6 log10(d_km)` (ref. [48]), shadow
fading `N(0, 8 dB)` (3GPP TR 38.901 Table 7.4.1-1), and inter-cell interference
from "the six cells with the highest RSRP" (Eq. 8, ref. [8]). It does not specify
interferer distances or transmit power offsets.

Our initial choice (6 interferers, uniform 0.5-3.0 km, equal transmit power)
produced SINR ~= -10 dB at tpc=1, i.e. an interference-limited regime in which
the channel, not the policy, determines the outcome. Paper Fig. 9(b) plots
semantic accuracy over an SINR sweep extending to at least 15 dB with high
accuracy at the top end, which is inconsistent with that regime.

**Assumption adopted (Run D):** 3 interferers, uniform 1.5-4.0 km, NLOS,
-1 dB transmit power offset. Resulting SINR range: 1.7-10 dB.

This is a modelling choice made to place the system in the paper's implied
operating region. It is not derived from the paper. All Run D results are
conditional on it. `code/experiments/calibrate_env.py` reproduces the
calibration sweep.

Full Run D constant set:

```
tx_power = 30 dBm, bandwidth_total = 20 MHz
INTERFERENCE_CELLS = 3
INTERFERER_DISTANCE_KM = [1.5, 4.0], NLOS
INTERFERER_POWER_OFFSET_DB = -1
N_CLUSTERS_NLOS = 19, N_RAYS = 20
medium difficulty: delay_intent U(0.50, 2.50) s
quality_intent U(0.10, 0.40)
data_size U(0.1, 0.5) MB
OMEGA1_DELAY = 0.05, OMEGA2_QUALITY = 0.02
BLER = 0.95 * (1 - exp(-0.76 * (overreach - 1))), overreach > 1
bandwidth floor = 10% of fair share per task
```

---

## 3. Run D results (final)

Intent satisfaction rate, 200 evaluation episodes, seed 42:

| tpc | tasks | HDM | AC | PPO | SAC | Static | HDM vs Static |
|-----|-------|-------|-------|-------|-------|--------|---------------|
| 1 | 5 | 0.927 | 0.777 | 0.886 | 0.847 | 0.956 | -3% |
| 2 | 10 | 0.791 | 0.754 | 0.808 | 0.749 | 0.861 | -8% |
| 4 | 20 | 0.815 | 0.864 | 0.751 | 0.740 | 0.529 | +54% |
| 10 | 50 | 0.643 | 0.075 | 0.784 | 0.554 | 0.080 | +704% |

Mean delay (s) / mean distortion:

| tpc | HDM | AC | PPO | SAC | Static |
|-----|-------------|-------------|-------------|-------------|-------------|
| 1 | 0.414/0.334 | 0.060/0.418 | 0.178/0.361 | 0.322/0.364 | 0.349/0.319 |
| 2 | 0.463/0.350 | 0.080/0.423 | 0.125/0.398 | 0.078/0.428 | 0.705/0.318 |
| 4 | 0.508/0.359 | 0.322/0.359 | 0.181/0.427 | 0.183/0.432 | 1.469/0.319 |
| 10 | 0.923/0.399 | 3.655/0.317 | 0.588/0.403 | 1.212/0.421 | 3.657/0.318 |

Static calibration (uniform BW, QPSK): ISR 0.905 / 0.743 / 0.356 / 0.077
at tpc = 1 / 2 / 4 / 10 respectively (independent calibration sweep).

---

## 4. Ablations

### HAN vs flat MLP encoder (paper Fig. 13(a)), tpc=4

| Run | Code state | HAN | MLPEncoder | Delta |
|-----|-----------|-------|------------|-------|
| A | Bugs A,B,E,F present | 0.492 | 0.391 | +25.7% |
| C | Bugs fixed, loose intents | 0.399 | 0.395 | +1.1% |
| D | Bugs fixed, Run D channel | 0.805 +/- 0.083 | 0.845 +/- 0.067 | **-4.7%** |

The Run A figure is discarded for two reasons: (i) Bug B held every urgency
feature at zero at tpc=4, so the ablation did not measure attention over intent
features; (ii) `MLPEncoder(task_input_dim=6)` and
`CSCGraphBuilder(message_feat_dim=4)` were fed different state vectors, so the
comparison was uncontrolled.

**Conclusion: HAN provides no measurable benefit in this implementation.**
Two independent post-fix runs bracket zero (-4.7%, +1.1%) with per-arm standard
deviations of 0.067-0.083.

Probable cause, from `code/hdm/csc_graph_builder.py`: the CSC graph topology is
constructed once in `__init__` (round-robin `arange % n`) and never resampled.
Every episode presents an identical edge set; only node features vary. Node-level
and semantic-level attention exist to exploit relational variation, of which
there is none, so HANConv over a fixed graph with mean pooling reduces to a
shared per-node MLP with fixed aggregation -- i.e. to `MLPEncoder`.
Testing the paper's claim requires episode-varying topology (dynamic CSCA-BS
association, distance-dependent relay reachability). This reproduction does not
implement it, and the ablation should be read as untested rather than refuted.

Note: the MLPEncoder actor loss diverged to -487 by episode 1000. The reported
0.845 is its best evaluation checkpoint (episode 250, ISR 0.844). Divergence
occurred after the peak, so the comparison against HAN's best checkpoint
(episode 600) is valid, but the MLP arm is not a stable-training result.

### Denoising steps N (paper Fig. 12(a)), tpc=4

N=5: 0.843, N=6: 0.817, N=7: 0.831.

Flat within noise. The paper's claimed ordering (N=6 best, N=7 degrading through
over-denoising) is **not reproduced**. Our N=5 is nominally best. With per-arm
std ~0.07 this ablation is underpowered at 200 evaluation episodes and one seed;
it should not be cited in either direction.

### Actor loss form (paper Eq. 33), tpc=4

Eq. (33) requires `L = -E[log pi * (R_acc - V)]`. Our DDPM actor is a
deterministic-at-evaluation reverse chain; we implemented a stochastic-path
log-probability (`ablation_logpi.py`, lambda = 0.01) and measured **-5.5% ISR**
versus the deterministic policy gradient `-Q.mean()`. Reported results use
`-Q.mean()`, i.e. a DDPG/DPG-form update. This is a documented deviation.

---

## 5. Findings that do not reproduce

### 5.1 Delay (paper Fig. 9(d)): reversed

The paper reports HDM achieving the lowest communication delay. In every run,
HDM's mean delay exceeds AC and PPO at every load. Three mechanisms, in order
of magnitude:

1. **MCS conservatism.** At tpc=1 HDM's delay (0.414 s) implies spectral
   efficiency ~0.23, i.e. QPSK -- the same modulation Static hardcodes. AC's
   0.060 s implies ~1.33 (16QAM). HDM buys the lowest distortion of any learned
   policy (0.334 vs AC 0.418) by declining the BLER penalty from over-reaching
   MCS (`BLER = 0.95(1 - exp(-0.76(overreach - 1)))` in `sim_channel.step()`).
2. **Bandwidth convexity.** Mean delay `sum(D_i / (eff * B_i))` is convex in
   B_i under a fixed budget, so any non-uniform allocation raises the *mean*
   delay relative to uniform even while reducing the *count* of missed
   deadlines. HDM optimises CSCQI/ISR, a threshold count; mean delay is the
   wrong statistic against which to judge it, and Static is by construction the
   mean-delay minimiser.
3. **Relay hops.** See 5.2 -- small and load-independent.

HDM does adapt with load: from tpc=1 to tpc=4 its delay rises only 0.414 -> 0.508 s
while Static's rises 0.349 -> 1.469 s, indicating HDM escalates to higher-order MCS
for a subset of tasks under congestion at a cost of +0.025 distortion. AC applies
maximum aggressiveness at all loads and collapses to ISR 0.075 at tpc=10. This
load-conditional behaviour is the clearest positive result of the reproduction.

### 5.2 Semantic relay effectively never fires

`select_relay()` triggers only when `distortion_direct > 1 - intent_quality`.
With `quality_intent` in [0.10, 0.40] the threshold is 0.60-0.90, while the
DeepSC-calibrated distortion proxy yields 0.70 (0 dB) to 0.30 (10 dB). Over the
Run D SINR range the direct path clears the threshold in the large majority of
episodes. Paper Fig. 9(b)'s claim that HDM maintains accuracy at low SINR *by
selecting relays* is therefore untested here.

### 5.3 Low-load regime: uniform QPSK is optimal, not a weak baseline

At tpc=1 all policies clear the 0.50-2.50 s deadlines, so ISR is determined
entirely by the distortion constraint `theta <= 1 - q`. The unique optimal action
is minimum-order MCS, which Static implements exactly. Static's ISR of 0.956 is
an oracle result at that load, not a baseline artefact. HDM's -3% is exploration
noise around the optimum. Learned-policy advantage is meaningful only where
Static collapses (tpc >= 4).

---

## 6. Known deviations from the paper

1. **LAM left brain absent.** No LLaVA-NeXT-Interleave, no LKB, no RAG, no
   CohereRerank, no self-adaptive retrieval. Intents are drawn from
   `np.random.uniform`. Sections III-B.1 and Figs. 3-4 are not implemented.
   Intent *cognition* is therefore not evaluated; only intent *satisfaction*
   given ground-truth intents.
2. **Relay removed from the action space.** `parse_action()` returns
   `relay = zeros`; `action_dim = n_tasks + n_tasks * n_mcs`. Paper Eq. for
   `a_t = {BW_t, Pi_t, Theta_t}` includes relay selection. Relay is instead a
   hand-coded heuristic applied identically to all policies, so it cannot
   differentiate them.
3. **Eq. (15) semantic mutual information is a proxy.** The exact form requires
   `p(alpha)`, `p(beta)`, `f(mu|alpha)` from an LKB symbol probability table.
   We substitute a Jensen-Shannon/cosine blend over sentence embeddings
   (`compute_semantic_mi_approximation`). Labelled as an approximation in code
   and in all outputs.
4. **Eq. (33) actor loss replaced by `-Q.mean()`** (see Section 4).
5. **Contextual bandit, not an MDP.** One `env.step()` per episode; `gamma=0.95`
   is declared but never applied; `tau_w` is a static function of task count
   (`max(0, load - 0.5) * 0.05`), not a dynamic queue. Eq. (34)'s discounted
   return reduces to the immediate reward.
6. **Static baseline is fixed-QPSK.** `mcs = [1/3,1/3,1/3]` with `argmax` selects
   index 0 (QPSK) always. The comparison is therefore HDM vs
   uniform-bandwidth-fixed-QPSK, not vs an SINR-adaptive scheduler. At low load
   this baseline is optimal (Section 5.3); at high load it is weak. Both facts
   must be stated whenever the +54% / +704% figures are quoted.
7. **Algorithm 1 (minimum synonymous subsequence) is not implemented.** Text
   compression is a fixed 73% word truncation, chosen to match the paper's
   reported 73.3% ratio. The compression figure for text is therefore an input,
   not a measurement, and is labelled as such in
   `fig6_multimodal_semcom.csv` (`compression_note`). Audio (0.32) and image
   (0.21) ratios are measured on transcribed/captioned text.
8. **Baselines share the HAN.** AC/PPO/SAC consume frozen HAN embeddings via
   `PerTaskGaussianActor`. The comparison is HAN+DDPM vs HAN+PPO etc., i.e. an
   isolation of the generative policy head, not of the full architecture.
9. **Image captioning** uses BLIP-2 OPT-2.7B, with a filename-derived caption
   fallback on OOM. The paper uses Stable Diffusion for image reconstruction and
   PSNR >= 22 dB as the accuracy criterion; we substitute caption-embedding
   cosine similarity. The image row of Fig. 6a is not directly comparable to the
   paper's.
10. **Single seed.** All headline numbers are seed 42. `joshi_eval_v2.py`
    supports 3-seed evaluation but was not run for Run D. Reported +/- values are
    across evaluation episodes, not across seeds, and understate total variance.

---

## 7. Multimodal evaluation (independent of the RL runs)

Semantic similarity vs SNR, word-erasure channel proxy, MiniLM cosine:

| SNR (dB) | Text (Europarl) | Audio (VoxCeleb+Whisper) | Image (Oxford+BLIP-2) |
|----------|-----------------|--------------------------|------------------------|
| 0 | 0.759 | 0.543 | 0.521 |
| 10 | 0.898 | 0.700 | 0.662 |
| 25 | 0.986 | 0.846 | 0.771 |

The channel here is a word-level erasure/substitution proxy driven by the
`sim_channel` distortion output, not a DeepSC forward pass. Prior to fix M-FIX-1
the same scalar was written into every SNR bucket, producing a flat curve; that
defect is corrected and the monotone trend above is real, but the absolute values
depend on the erasure model and should not be compared numerically to the paper's
Fig. 6(a).

---

## 8. Reproduction commands

```
python code/experiments/calibrate_env.py        # Static ISR calibration sweep
python code/experiments/final_results.py        # Run D main table + N ablation
python code/experiments/ablation_han.py         # HAN vs MLPEncoder, tpc=4
python code/experiments/ablation_logpi.py       # Eq. 33 vs -Q.mean(), tpc=4
python code/experiments/multimodal_eval.py      # Fig. 6 equivalent
```

Outputs are written to `results/final/`. `results/final/SUMMARY.txt` is appended
to by each script and is the authoritative record.

---

## 9. Summary judgement

The paper's central qualitative claim -- that intent-aware, per-task policy
generation outperforms uniform allocation under intent competition -- reproduces,
and does so strongly (+54% at 20 tasks, +704% at 50 tasks). The claim does not
hold under resource abundance, where uniform minimum-order MCS is provably
optimal and no learned policy can exceed it.

Three specific quantitative claims do not reproduce: the delay advantage
(Fig. 9(d), reversed here), the HAN ablation (Fig. 13(a), -4.7% here, with a
structural explanation in Section 4), and the denoising-step ordering
(Fig. 12(a), flat within noise). The headline "+42.19% ISR" is regime-dependent
in our testbed and ranges from -8% to +704% depending solely on task count.
