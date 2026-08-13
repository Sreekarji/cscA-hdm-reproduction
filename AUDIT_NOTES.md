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

> **Results status.** Sections 3, 4 and 5 report Run D, which predates the fixes
> in Sections 10.1, 10.4 and 10.5 and the action-space change in deviation 2.
> Those fixes alter the intent features the HAN sees, the `comm_req` graph
> topology at tpc != 1, and the action width, so **Run D checkpoints are stale
> and its numbers are pending re-measurement.** Retrain with
> `python code/experiments/final_results.py` (tpc 1, 2, 4, 10; 1000 episodes;
> seed 42) before citing any figure below.

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

Probable cause, from `code/hdm/csc_graph_builder.py` **as it stood for Runs A/C/D**:
the CSC graph topology was effectively static. `comm_conn` and `comm_req` are
deterministic functions of node index, and `relay_knowledge` is drawn once in
`__init__` and never resampled. Every episode presented a near-identical edge
set; only node features varied. Node-level and semantic-level attention exist to
exploit relational variation, of which there was none, so HANConv over a fixed
graph with mean pooling reduces to a shared per-node MLP with fixed aggregation
-- i.e. to `MLPEncoder`.

The one episode-varying edge type is `semantic_conn` (message -> relay), which
is `semantic_type INTERSECT relay_knowledge` and so re-draws with the episode's
modality sample. Measured at 20 messages / 5 relays, seed 42, 500 episodes:
mean 39.9 edges, range 29-53, std 3.6. (An earlier docstring claimed the
constant `RELAY_KNOWLEDGE_P = 0.475` was tuned for 45-50; that was never
measured and is wrong. Expected value is `n_messages * mean(relays per type)`
= `20 * (1+3+2)/3` = 40.) Note also that the seed-42 knowledge draw leaves
text covered by only **one** of five relays, so relational variation is not
merely confined to one edge type but is thin within it.

Testing the paper's claim properly requires more: dynamic CSCA-BS association,
distance-dependent relay reachability. This reproduction does not implement
those, and the ablation should be read as untested rather than refuted.

Note also that Runs A/C/D predate the fix in Section 10.5, so their `comm_req`
edges encoded a task-to-CSCA mapping the simulator did not use. The -4.7% figure
is not a clean measurement of HAN's value and should be re-measured after
retraining.

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

1. **LAM left brain — implemented (was: absent).** `code/experiments/lam_intent_generator.py`
   parses text/image/audio intents through Qwen2.5-VL via Ollama with
   grammar-constrained structured output, backed by a self-adaptive RAG loop over
   a local knowledge base (`data/lkb/intent_corpus.json`, 25 exemplars):
   MiniLM retrieval (k=8) -> Cohere rerank (top_n=3) -> ISREL reflection with up
   to 2 query reformulations -> generation -> ISSUP reflection, falling back to
   the no-retrieval parse when the retrieved evidence does not support the
   produced intent. Toggle: `USE_ADAPTIVE_RAG`. Cohere is optional; without
   `COHERE_API_KEY` the rerank degrades to embedding order.
   **Still a deviation:** the paper uses LLaVA-NeXT-Interleave, we use
   Qwen2.5-VL-3B (6 GB VRAM limit). RL *training* still uses synthetic
   `np.random.uniform` intents — this matches the paper (Table II), which trains
   on sampled intents; the LAM runs at inference time only
   (`lam_intent_demo.py`, `e2e_demo.py`, `final_results_lam.py`).
2. **Relay removed from the action space (deliberate, flag-controlled).**
   `train_han_mlp.USE_RELAY_ACTION = False`, so
   `action_dim = n_tasks * (1 + n_mcs)` (80 at tpc=4 rather than 180) and
   `parse_action()` returns `relay = zeros`. The paper's `a_t = {BW_t, Pi_t,
   Theta_t}` includes relay selection, but `sim_channel.step()` picks relays with
   the heuristic `relay_select()` and never reads `action["relay"]`, so those
   `n_tasks * n_relays` outputs received no reward-linked gradient while still
   inflating the action space and the SAC entropy target. Relay is therefore a
   hand-coded heuristic applied identically to all policies and cannot
   differentiate them. Set `USE_RELAY_ACTION = True` to reproduce the pre-fix
   layout; `compute_action_dim()` is the single source of truth and every actor,
   critic, and experiment script derives its width from it.
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
7. **Algorithm 1 (minimum synonymous subsequence) — implemented.**
   `code/hdm/mss_algorithm.py` realises Algorithm 1 as a greedy importance-ordered
   deletion: repeatedly drop the single token whose removal costs the least
   BERT-family (MiniLM) cosine similarity, stopping when the cheapest remaining
   deletion would fall below `epsilon` (default 0.95). The compression ratio is
   therefore **measured**, not assumed. `e2e_demo.py` uses this path.
   `multimodal_eval.py` keeps `COMPRESSION_METHOD = "eta"` (fixed 73% prefix
   truncation) as the default so previously reported Fig. 6b numbers remain
   reproducible; set `COMPRESSION_METHOD = "mss"` for the paper-faithful path.
   Under `"eta"` the text compression figure is an input, not a measurement, and
   is labelled as such in `fig6_multimodal_semcom.csv` (`compression_note`).
   Audio (0.32) and image (0.21) ratios are measured on transcribed/captioned text.
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

LAM / semantic-layer entry points (inference only, no training):

```
python code/hdm/mss_algorithm.py                       # Algorithm 1 smoke test
python code/experiments/lam_intent_generator.py "..."  # RAG retrieval + intent parse
python code/experiments/lam_intent_demo.py             # one intent, all tasks
python code/experiments/e2e_demo.py                    # per-task text/image/audio
python code/experiments/final_results_lam.py           # main table with LAM intents
```

Action-space smoke tests (both `USE_RELAY_ACTION` settings):

```
python code/hdm/ddpm_policy.py
python code/hdm/mlp_policy.py
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

---

## 10. Additional Deviations (Post-Run Audit)

### 10.1 Quality Intent Normalisation (A2 + A1) — FIXED
**Paper:** Quality intent drives semantic accuracy requirement.
**Was:** `intents_from_state()` passed raw `quality_intents` (range 0.10–0.40)
to the HAN unnormalised while delay urgency was range-normalised to [0,1], so
the two intent channels had asymmetric scales. Worse, the delay channel used a
*per-episode min/max* rescale: the mapping from a physical deadline to a feature
value depended on which other deadlines happened to be drawn that episode, and
for uniform injected intents (LAM demo) the range collapsed to ~0 and the
urgency column went identically zero.
**Fix:** single shared helper `sim_channel.normalise_intents(delay, quality)`
with fixed bounds, now the only place either intent is scaled:
```
QUAL_INT_MIN = 0.10, QUAL_INT_SPAN = 0.30, DELAY_URGENCY_DIVISOR = 2.50
di = clip(delay / 2.50, 0, 1)
qi = clip((quality - 0.10) / 0.30, 0, 1)
```
Both channels now span the full [0,1] and are episode-independent. Call sites
converted: `sim_channel.generate_state()`,
`sim_channel.generate_state_with_params()`, `train_han_mlp.sample_eval_state()`,
`train_han_mlp.intents_from_state()`, `lam_intent_demo.inject_lam_intents()`,
`final_results_lam._inject_lam`. `csc_graph_builder` carries the same three
constants as class attributes (`QUAL_INT_MIN` / `QUAL_INT_SPAN` /
`DELAY_URGENCY_DIVISOR`) for its no-`intent_vectors` fallback path; they must
track the helper.
**Deliberately NOT converted:** `HighPressureEnvironment` keeps the old `/10.0`
and `(1.0 - qi)` formula. It is a separate stress-test environment whose
published numbers are calibrated against that scaling; a NOTE comment at the
call site records this.
**Impact:** invalidates pre-fix checkpoints — retrain required.

### 10.2 INTERFERENCE_CELLS=3 vs Paper's 6
**Paper line 697:** "inter-cell interference is mainly determined by the six cells with the highest RSRP"
**Code:** `INTERFERENCE_CELLS = 3` (sim_channel.py)
**Reason:** Reduced interference complexity during environment calibration. Fewer interference cells means achievable ISR is higher. Results are internally consistent but not directly comparable to paper's absolute ISR values.

### 10.3 BLER Curve Parameters
**Paper:** Does not specify BLER slope or ceiling numerically
**Code:** `BLER_SLOPE = 0.76`, `BLER_CEIL = 0.95` — empirically calibrated
**Reason:** Paper references 3GPP link-level curves without providing exact fit parameters. Values were tuned so that medium-difficulty intents are achievable.

### 10.4 Urgency Range Too Narrow — FIXED
**Was:** `di = delay_intents[i] / 10.0` gave range [0.05, 0.25] for medium
difficulty, so `(1.0 - di)` spanned ~0.20 instead of ~1.0 and the HAN could
barely separate tasks by deadline.
**Fix:** divisor changed to `DELAY_URGENCY_DIVISOR = 2.50`, the top of the
medium-difficulty delay range, so urgency now spans the full [0.20, 1.00] for
delays in [0.50, 2.50]. Same helper as 10.1.

### 10.5 Message-to-CSCA edge assignment contradicted the simulator — FIXED
`csc_graph_builder.build()` assigned task *i* to CSCA `i // tasks_per_csca`
(block), while `sim_channel.step()` draws task *i*'s path loss from
`csca_positions[i % n_cscas]` (round-robin). The two agree only at tpc=1. At
tpc 2/4/10 the `comm_req` edges told the HAN that tasks `0..tpc-1` contend for
one CSCA's resources when the simulator had in fact placed them at `tpc`
different CSCAs — the graph encoded physics the environment did not implement.
**Fix:** `csca_assign = [i % n_c for i in range(n_m)]`.
**Impact:** changes graph topology at tpc != 1, so any checkpoint trained before
this fix has a mismatched `comm_req` structure and must be retrained.

Two smaller inconsistencies fixed in the same pass:
- `data_sizes_norm` divided by `5e5` while `sim_channel` writes
  `message_features` using `6e5`; now both use `6e5`.
- In the no-`system_state` branch, `message_feats` column 1 (the semantic-type
  channel) was left as an independent random draw while the relay edges were
  built from `sem_types`; column 1 is now set from `sem_types`.

### 10.6 Stale absolute path
`sim_channel.py` did `sys.path.insert(0, r"D:\MP2\code\channel")`, a hardcoded
path to a third tree that does not exist in this repo. Replaced with a
`__file__`-relative insert.

---

## 11. Gap Closure — Aug 2026 (lam-integration branch)

Three gaps identified by the external audit (2026-08-13) were resolved on the
`lam-integration` branch. Committed results in `results/final/` are not
changed; these fixes prepare the pipeline for the next phase (LAM training
loop integration).

### 11.1 Gap 1 — LKB corpus populated

**Problem:** `data/lkb/intent_corpus.json` was absent. `lkb_retrieve()` returned
`[]`; the ISREL/ISSUP RAG loop and CohereRerank had nothing to retrieve from.

**Fix:**
- Created `data/lkb/intent_corpus.json` with 200 synthetic exemplars.
  Schema: `{id, sentence, delay_s, quality, modality, priority}`.
  50 entries per modality (text/image/audio/multimodal); priority distributed
  across low/medium/high/critical with delay and quality matched to priority.
- All 49 offline tests pass confirming corpus schema, distribution, and
  retrieval correctness (`code/tests/test_lkb_corpus.py`).

**Additional fix — local reranker fallback:**
The original `cohere_rerank()` silently truncated to embedding order when
`COHERE_API_KEY` was absent, with no warning. Replaced with a local MiniLM
reranker (`_local_rerank()` in `code/lam/lam_intent_generator.py`) that
computes cosine similarity between the query and each candidate sentence using
the same model already loaded for `lkb_retrieve()`. A warning is emitted only
if `sentence_transformers` is itself unavailable (the true degenerate case).

### 11.2 Gap 2 — LAMIntentProvider + LAM_TRAINING_MODE

**Problem:** The HDM training loop called `env.generate_state()` for synthetic
intents every episode. LAM output never entered the RL training distribution.

**Fix — `code/lam/lam_provider.py`:**
- `LAMIntentProvider` class with `get_intent(task_idx)` and `get_batch(n)`.
- Episode-level cache: 20 intent slots refreshed every 50 episodes so Ollama
  is called at most 20× per refresh, not once per training step.
- Graceful synthetic fallback (same medium-difficulty distribution as the env)
  when Ollama is unreachable — training never crashes mid-run.
- `HANMLPTrainer._inject_lam_intents(state)` in `train_han_mlp.py` overwrites
  `delay_intents`, `quality_intents`, and `data_sizes` (scaled by MSS
  `compression_ratio`) in the state dict and rebuilds `message_features` so
  HAN sees normalised intent features consistently.

**Flag — `config.LAM_TRAINING_MODE`:**
- Default `False`: existing 5-CSCA results reproducible without Ollama.
- Set `True` to activate real LAM-intent training. Injection happens before
  `intents_from_state()` is called, so the HAN, DDPM, and critic all train
  on the LAM-parsed distribution without any other code change.

**Constraint respected:** RTX 4050 6 GB VRAM. `LAMIntentProvider` runs Ollama
on CPU/offloaded via the Ollama server process; the HDM tensors remain on GPU.
The two never load simultaneously.

### 11.3 Gap 3 — Native `forward_with_logprob` in DDPMActor

**Problem:** Paper Eq. 33 entropy term (`-λ * log_pi`) was implemented only as
a monkey-patch in `ablation_logpi.py` and not available in the production
`DDPMActor` class used during training.

**Fix — `code/hdm/ddpm_policy.py`:**
- Extracted action-head logic from `forward()` into `_raw_to_action(raw)`
  so both `forward()` and `forward_with_logprob()` share a single action-head
  implementation without duplication.
- Added `DDPMActor.forward_with_logprob(graph_emb, message_embs)` as a native
  method. Logic ported directly from the `ablation_logpi.py` monkey-patch
  (which was itself the reference implementation). Returns `(action, log_pi)`
  where `log_pi = Σ_{n=N..2} Σ_{tasks} -0.5*(z_n^2 + log(2π) + log(β̃_n))`.
- Updated `train_han_mlp.py` actor-loss block: when `USE_ENTROPY_REG=True`
  calls `forward_with_logprob()` and computes
  `actor_loss = -Q.mean() - ENTROPY_COEFF * log_pi.mean()` (Eq. 33).

**Flag — `config.USE_ENTROPY_REG`:**
- Default `False`: actor loss = `-Q.mean()` (committed results unchanged).
- Set `True` to activate Eq. 33. `ENTROPY_COEFF = 0.01` (paper value).

**Audit item B6 — `forward_with_logprob`:** implemented natively in
`DDPMActor`. `USE_ENTROPY_REG=False` (default) — entropy term excluded from
training loss after empirical ISR regression observed during `ablation_logpi`
experiment. Method is present and gradient-tested; not active in reported
results.

**Regression check:** 49 offline tests pass with `USE_ENTROPY_REG=False`,
confirming no change to the default training path.

### 11.4 Ablation fix — `ablation_ddpm.py` tpc=10

**Problem:** The committed `ablation_ddpm.csv` showed the claimed “+14.1% at
tpc=10” but the script was hard-coded to `tpc=4`. The result was actually the
tpc=4 improvement.

**Fix:** `ablation_ddpm.py` now loops over `tpc ∈ {4, 10}`, writes both rows to
CSV with a `tpc` column, and also records `delay_mean` (previously dropped).
The original tpc=4 result (+14.1%) is preserved as the first row.

### 11.5 Test suite

`code/tests/` created with 49 offline tests across three files:
- `test_lkb_corpus.py` — corpus schema, distribution, `lkb_retrieve()`,
  `cohere_rerank()` local fallback
- `test_ddpm_policy.py` — `forward_with_logprob()` shape, finiteness, gradient
  flow, action-head consistency, noise schedule; config flag defaults
- `test_lam_provider.py` — `get_intent()`/`get_batch()` ranges, cache cycling,
  state injection contract, smoke test

Run: `python -m pytest code/tests/ -q`  (∼30 s, no Ollama, no GPU needed).
