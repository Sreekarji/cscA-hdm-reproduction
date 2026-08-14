# Audit Fixes

## 1. Normalisation Fix (2026-07-29)

**Deviation 10.1 — Quality intent not normalised before HAN**
- Fix: quality_intents normalised from raw [0.10, 0.40] to [0, 1] via (q - 0.10) / 0.30
- Applied in: `sim_channel.py` generate_state(), `train_han_mlp.py` sample_eval_state() and intents_from_state()

**Deviation 10.4 — Urgency range too narrow**
- Fix: delay normalisation changed from /10.0 (range [0.05, 0.25]) to /2.50 (range [0.20, 1.00])
- Applied in: `sim_channel.py` generate_state(), `train_han_mlp.py` sample_eval_state()
- Urgency formula also standardised to use qi (not 1-qi) for consistency between generate_state() and sample_eval_state()

**Result:** tpc=4 ISR improved from 0.817 (baseline) to 0.8325 (+1.90%). Changes kept and checkpoint saved as `han_ddpm_tpc4_norm_best.pt`.

## 2. MSS Algorithm & MIM-based MCS Constraint (2026-07-29)

**Files created:**
- `code/hdm/mss_algorithm.py` — Minimum Synonymous Subsequence (Algorithm 1) and MIM computation (Eq 5-6)
- Integrated into `multimodal_eval.py` for text compression

**MSS Compression (Part A):**
- Mean compression ratio: **0.8662** ± 0.1149 (vs previous fixed 0.73)
- Range: [0.50, 1.00] — MSS keeps 86.6% of words on average, only removes redundant words
- Previous fixed truncation (0.73) removes 27% of words regardless of semantic content

**MIM Distribution on SST text (Part B):**
- MIM mean: 0.4167 ± 0.2560, range [0.0843, 1.0000]
- At omega=1.0: 83% of texts have max_mcs=2 (unconstrained), 17% constrained (max_mcs<2)
- Only texts with MIM > 0.80 get constrained

**ISR Impact (Part C):**
- Baseline norm-fix: **0.8325 ± 0.0721**
- With MIM constraint (tpc=4, 1000 eps): **0.6330 ± 0.0966**
- Best during training: 0.850 (ep 450)
- Difference: -0.1995 (-23.96%) — far exceeds 3% threshold

**Decision:** Reverted sim_channel.py MIM constraint. MSS code kept in mss_algorithm.py and multimodal_eval.py for future use. Root cause: MIM constraint forces lower-reliability MCS → lower rate → more delay violations → lower ISR.

## 3. Self-Adaptive RAG with ISREL/ISSUP (2026-07-29)

**Paper reference:** Sec III-B.1 — reflection tokens for retrieval quality.

**Files modified:**
- `code/lam_intent_generator.py` — added `USE_ADAPTIVE_RAG` flag, `_ISREL_PROMPT`, `_ISSUP_PROMPT`, `_check_isrel()`, `_check_issup()`, `_left_brain_adaptive()`
- `verify_lam_hdm.py` — fixed sys.path (added `code/`)

**How it works:**
```
user text
  → lkb_retrieve() top-8
  → cohere_rerank() top-3
  → _check_isrel()    — ISREL token: is retrieved content domain-relevant?
      ↓ YES                     ↓ NO (reformulate query + re-retrieve, max 2 iters)
  → _ollama_chat(refined)       generate intent with LKB context
  → _check_issup()    — ISSUP token: is generated intent supported by context?
      ↓ YES                     ↓ NO
  → return                      → _ollama_chat(raw) fallback (no retrieval)
```

**Test results (5 intent sentences):**

| Sentence | Old (d,q) | New (d,q) | Old t | New t |
|---|---|---|---|---|
| send it immediately, quality doesn't matter | (0.600, 0.100) | (0.600, 0.100) | 6.2s | 0.8s |
| accurate delivery within 2 seconds | (2.000, 0.350) | (2.000, 0.350) | 0.4s | 1.0s |
| stream my voice call right now | (0.600, 0.150) | (0.600, 0.150) | 0.4s | 0.8s |
| backup this file overnight | (2.500, 0.350) | (2.500, 0.350) | 0.4s | 0.8s |
| this is urgent, get it there fast | (0.600, 0.100) | (0.600, 0.100) | 0.4s | 0.8s |

**ISREL/ISSUP stats:** ISREL=False=1/5 (reformulated query, succeeded on retry), ISSUP=False=0/5. All values identical — no quality regression from adaptive path.

**Latency impact:** adaptive adds ~0.4-0.6s/call for two extra Ollama checks. Toggle via `USE_ADAPTIVE_RAG` flag.

**verify_lam_hdm.py:** All 25 checks passed. No pipeline regressions.

**Status:** KEPT. Flag-gated, no impact on training or simulation path.

## 4. CSC Graph Topology Fix (Dynamic Per Episode)

**Problem (Bug D):** `csc_graph_builder.py` built edge connections using `torch.randint()` with global RNG. With fixed seed 42, every training episode produced identical relay-message and message-CSCA edge sets. The HAN over this fixed graph degraded to a shared per-node MLP — same as `MLPEncoder`. The previous +15.9% HAN advantage may have been an artifact.

**Fix (File: `code/hdm/csc_graph_builder.py`):**
- Replace global RNG edges with state-content-dependent logic
- **Relay-message edges:** determined by `semantic_type` (from SCt) ∩ `relay_knowledge` (fixed per relay)
- **Message-CSCA edges:** determined by actual CSCA assignment: `message_idx // tasks_per_csca`
- **Relay knowledge sets:** initialised once with `np.random.default_rng(42)` to be deterministic across runs but different per relay
- **semantic_types:** read from SCt (random per episode via `sim_channel.generate_state()`)

**Fix (File: `code/channel/sim_channel.py`):**
- Added `SCt["semantic_types"]` — random 0/1/2 per task per episode — in both `generate_state()` and `generate_state_with_params()`

**Edge count verification (tpc=4, 10 episodes):**
```
ep 0: relay-msg=48  msg-csca=20
ep 1: relay-msg=49  msg-csca=20  <-- relay varied
ep 2: relay-msg=50  msg-csca=20  <-- relay varied
...
```
**Relay-msg edges vary: True** (45-50 range) — dynamic topology confirmed.
**Msg-csca edges vary: False** (always 20, fixed by task count) — but node features on those edges vary per episode.

**Ablation re-run (`ablation_han.py`, both trained from scratch, tpc=4):**
| Model | ISR | vs Previous |
|---|---|---|
| HAN+DDPM (HDM) | 0.8203 ± 0.0829 | — |
| MLPEncoder+DDPM | 0.6900 ± 0.0970 | — |
| **HAN advantage** | **+18.9%** | dynamic topology |

**Verdict:** HAN advantage **confirmed** with dynamic topology (+18.9%). The +15.9% result was NOT an artifact of fixed topology — it holds and slightly improves. HAN graph attention is architecture-justified.

**Checkpoint saved:** `han_ddpm_tpc4_best.pt` (trained with dynamic topology, best ISR=0.564 at ep 600 during training, 0.8203 at eval).

## 5. Relay Selection in DDPM Action Space (Restoration Attempt)

**Context (from Sun et al. 2026 Eq for a_t = {BW_t, Π_t, Θ_t}):**
The paper defines the action as a tuple of bandwidth allocation, relay selection, and MCS selection. The DDPMActor already outputs relay dimensions (`task_dim = 1 + n_relays + n_mcs` = 9 per task, action_dim = 180). `parse_action()` already extracts relay into the action dict. **However,** `sim_channel.step()` was ignoring `action["relay"]` and using the heuristic `relay_select()` from `relay_selection.py` instead — making the DDPM's relay output dead code.

**Action audit (current DDPMActor):**
```
n_tasks  = 20
n_relays = 5
n_mcs    = 3
task_dim = 1 + 5 + 3 = 9     (BW logit + 5 relay scores + 3 MCS scores)
action_dim = 20 * 9 = 180    (already relay-inclusive)
forward() output layout:
  cols 0..19:           BW softmax (sums to 1)
  cols 20..119:         relay sigmoid (5 per task, flattened)
  cols 120..179:        MCS sigmoid (3 per task, flattened)
```

**Wiring attempt:** Modified `sim_channel.step()` relay selection block (lines 581-605) to check for `action["relay"]` and use the DDPM's chosen relay (argmax over 5 scores; confidence threshold 0.5).

**Training result (tpc=4, 1000 episodes, seed 42):**
| Model | ISR | vs heuristics |
|---|---|---|
| **Relay in action space** | **0.7682 ± 0.0934** | — |
| Heuristic relay + dynamic topology | 0.8203 ± 0.0829 | baseline |
| Heuristic relay + norm-fix | 0.8325 ± 0.0721 | norm baseline |

**Decision: REVERTED** — relay in action space degrades ISR by ~5% vs heuristic.

**Probable cause:** Relay selection adds 5 × n_tasks = 100 extra dimensions to the action space (from 80 to 180), increasing exploration difficulty with only 1000 training episodes. The heuristic `relay_select()` incorporates domain knowledge (distance, semantic MI, distortion threshold), which the DDPM would need many more episodes to learn from scratch.

**Checkpoint saved:** `han_ddpm_tpc4_relay_best.pt` (relay-in-action, ISR=0.7682, ep 500)
**Revert action:** `sim_channel.py` returned to heuristic relay. No code change needed to DDPMActor (`parse_action` already extracts relay — just unused).

## isr-v2 Experiments — 2026-08-14

### Baseline (isr-improvements branch)
tpc=1: 0.858 | tpc=2: 0.8725 | tpc=4: 0.7968 | tpc=10: 0.6267

### isr-v2 results (all 5 changes active)
tpc=1: 0.927 | tpc=2: 0.834 | tpc=4: 0.715 | tpc=10: 0.342

### Changes attempted and verdict

Change 1 — Adaptive critic LR (5e-4 if n_tasks>=40)
  Result: REVERTED
  Delta tpc=10: −45.4% (combined effect, cannot isolate)
  Delta tpc=4:  −10.2%
  Reason: Cannot isolate — all 5 changes active simultaneously.
  Interaction with reward normalisation likely dominant.

Change 2 — Scaled replay buffer (n_tasks * 100) + batch size
  Result: REVERTED
  Delta tpc=10: −45.4% (combined)
  Delta tpc=4:  −10.2%
  Reason: Cannot isolate. Larger buffer may have introduced
  stale normalised rewards from early catastrophic episodes.

Change 3 — Reward normalisation (EMA mean/std before buffer insert)
  Result: REVERTED — PRIMARY SUSPECTED CAUSE
  Delta tpc=10: −45.4% (combined)
  Reason: tpc=10 CSCQI goes to −4.0 at ep 100 (visible in logs).
  EMA std is tiny in early training; dividing by it amplifies
  negative rewards by 10-100x. Actor learns to minimise ISR.
  CSCQI going negative is a tpc=10 regime issue (tight deadlines,
  50 tasks) — reward normalisation is unsafe when rewards can be
  strongly negative. Do not re-attempt without reward clamping first.

Change 4 — Wider critic hidden_dim 256 → 512
  Result: REVERTED
  Reason: Wider critic with normalised rewards produced unstable
  Q-estimates. Cannot assess in isolation. Retest only after
  reward normalisation is confirmed safe.

Change 5 — Per-task denoiser heads (shared backbone + task heads)
  Result: REVERTED
  Reason: Architecture change likely needs more than 1000 episodes
  to converge given larger parameter count. May still be valid
  with 2000+ episodes. Retest in isolation if episode budget
  increases.

### Root cause summary
Reward normalisation (Change 3) is the primary failure mode.
EMA-based normalisation is unsafe when episode rewards are
strongly negative (tpc=10 regime). All other changes are
contaminated by this instability and cannot be assessed.
The only clean result: tpc=1 improved 0.858→0.927, suggesting
Changes 1-2 help at low task counts where rewards stay positive.

### Lesson
Never combine reward normalisation with architecture changes
in a single run. Test reward normalisation alone first with
a reward floor (clip CSCQI to [0, inf] before normalising)
or use return normalisation (normalise across batch, not EMA).
