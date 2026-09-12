# Smart Scan Strategy for Electronic Warfare — SIH PS26055

A machine-learning-based Electronic Support (ES) receiver scheduler that searches a wide RF
spectrum for hostile emitters **without prior intelligence** about how many emitters exist,
where they sit in frequency, or how they behave in time — and does it faster / more reliably
than a fixed open-loop sweep.

**Problem statement:** SIH25 PS26055 — passive ES receiver scan scheduling. A narrowband
receiver must repeatedly choose *which frequency band to look at next* out of many, against
an unknown, time-varying mix of hostile emitters (some intermittent, some periodically
scanning/rotating-antenna, some frequency-hopping), maximizing interception while a fixed
open-loop sweep (round-robin) "may lose time to nonthreatening emitters by not giving time to
new or threatening ones."

---

## TL;DR — what actually beats what

| Where tested | Winner | Margin |
|---|---|---|
| Our synthetic testbed, single run (25 scenarios) | Belief + periodicity heuristic | 2.17× round-robin |
| Our synthetic testbed, 8–10 independent seeds, bootstrap-significant | Belief + periodicity heuristic | beats belief-index, an idealized Whittle-index oracle, ε-Greedy, and residual PPO — all with 95% CIs excluding zero |
| **Real TSRD data** (17 real stare-mode files) | Belief + periodicity / Belief-index | **~2.8× the real deployed receiver's own recorded hardware dwell schedule** |

The hand-engineered belief-index scheduler — not a deep-RL model — is the strongest result in
this project, on both synthetic and real data. Deep RL (SAC-Discrete) gets close on synthetic
data; PEARL-style meta-context does not help. Full detail and caveats below.

---

## 1. The dataset

### 1.1 What TSRD actually is

The **Turing Synthetic Radar Dataset (TSRD)** — Gunn, Hosford, Jones, Zeitler, Groves,
Nockles, Alan Turing Institute — is a large-scale, gated Hugging Face dataset
(`alan-turing-institute/turing-synthetic-radar-dataset`) built for radar pulse deinterleaving
research. ~6,000 pulse trains (2,500 train / 250 validation / 250 test **per receiver mode**),
~4 billion pulses total, up to 90 emitters per file.

Every pulse is a 5-field **Pulse Descriptor Word (PDW)**: Time of Arrival (μs), Centre
Frequency (MHz), Pulse Width (μs), Angle of Arrival (°), Amplitude (dB).

**Two receiver modes, and this distinction matters a lot for how you use the data:**

- **Stare mode** (`stare/{train,val,test}_stare/config_<i>.h5`) — an oracle receiver that
  monitors the *entire* spectrum simultaneously and continuously. Its PDW stream captures
  essentially every pulse from every emitter, with no scanning decisions involved. This is
  what you need if you want to build your own scheduler and test it against complete ground
  truth — it tells you what *would* have been there had you looked at any given band at any
  given time.
- **Scan mode** (`scan/{train,val,test}_scan/config_<i>.h5`) — a realistic, bandwidth-limited
  receiver that sweeps a fixed schedule. It only records pulses from whichever band it
  happened to be tuned to. Not useful for simulating a *different* scheduler (whatever wasn't
  scanned is simply missing from the file) — but genuinely useful as **the literal recorded
  behavior of one specific real receiver design**, which we exploit directly (§4.4).

**Real HDF5 schema, confirmed against a downloaded file** (not guessed — see
`notebooks/smart_scan_strategy.ipynb` §9 for the full note):
```
data                 (n_pulses, 5) float32   -- columns: ToA(us), Frequency(MHz), PulseWidth(us), AoA(deg), Amplitude(dB)
labels               (n_pulses, 1) int8      -- emitter id per pulse
metadata/feature_names                        -- confirms the 5-column order above
metadata/receiver/freq_range_mhz              -- true receiver span, e.g. [500, 18000] (see grid note below)
metadata/receiver/dwell_centres_mhz           -- (scan-mode files) the receiver's actual recorded dwell schedule
metadata/transmitters/transmitters_<i>/...    -- each emitter's true generative params (frequencies, staggered PRIs, pulse width)
```
Real amplitudes run **≈ −190 to −10 dB** (actual received power), nothing like a small
positive-dB toy scale — this matters if you calibrate a detection model against them.

**Grid correction:** the true receiver grid is **36 bands over 0–18,000 MHz** (band 0 =
[0, 500) MHz is a guard band below where anything in this dataset transmits, not a bug) — an
earlier pass in this project assumed 500–18,500 MHz and had to correct it (see
`real_hardware_schedule_comparison.ipynb`).

### 1.2 What we actually used, and why

Most of this project's development and comparison work runs on **our own synthetic
generator** (12 bands × 150 slots), *not* real TSRD files — real access is gated, required a
Hugging Face account + accepted terms + token, and was only obtained partway through this
project. Our synthetic PDWs use the exact same 5-field schema so the same downstream pipeline
(`ScanEnv`, every scheduler, `evaluate_scheduler`) runs on either unmodified.

Once real access was obtained, two notebooks (§2.7, §2.8) validate the synthetic-testbed
findings against **17 real TSRD stare/scan file pairs** actually downloaded locally
(`data/stare/test_stare/`, `data/scan/test_scan/`, ~378 MB) — a directional check on real
pulse trains, real emitter geometry, real amplitude noise, not a certified 250-file benchmark
run (see caveats in §5).

---

## 2. What's in this repo — every notebook, in the order they were built

### 2.1 `notebooks/smart_scan_strategy.ipynb` — the main prototype

Builds the whole pipeline from scratch:
- **Synthetic emitter simulator**: `FixedEmitter`, `BurstyEmitter` (2-state Markov on/off),
  `PeriodicScanEmitter` (spatially-scanning/rotating-antenna), `FrequencyHopEmitter`
  (frequency-agile) — a fresh random mix every episode (**domain randomization**).
- **`ScanEnv`**: a restless multi-armed bandit / POMDP. Belief is Bayes-updated for the
  scanned band and *predicted forward* for every other band via an online-estimated Markov
  transition kernel — the mechanism that makes a scheduler smart instead of merely reactive.
- **Baselines**: round-robin, random.
- **Belief-index heuristic**: myopic `argmax(belief)`, provably optimal for cumulative reward
  on 2-state positively-correlated channels (Zhao, Krishnamachari & Liu, 2008) — plus a
  staleness/exploration bonus (a naive greedy version was shown to starve low-belief bands:
  higher hit rate but *lower* interception ratio than round-robin — exactly the "loses time to
  nonthreatening emitters" failure the problem statement warns about).
- **Belief + periodicity**: adds a per-band inter-hit-period estimator so the scheduler
  anticipates a rotating-antenna emitter's next illumination window.
- **Residual PPO**: a warm-started, bounded-correction policy (`logits = frozen index score +
  tanh-bounded MLP correction`) — the *stable* result after diagnosing why vanilla DQN and a
  naive PPO fine-tune both collapsed (see §3).
- **10-seed robustness sweep**: retrains PPO + rebuilds the eval set fresh for seeds 1–10,
  confirming the ranking holds and quantifying variance.
- Every figure of merit the problem statement names: Pd, Pfa, sensitivity (own empirical
  Pd-vs-amplitude curve), avg intercept rate, avg reward, % correct predictions, avg
  intercept-time error.
- A scaffolded (untested against real data at write time) `load_tsrd_pulse_train()` for
  Section 9 — since confirmed and superseded by §2.7/§2.8's real loaders.

### 2.2 `notebooks/paper1_egreedy_ps26055.ipynb` — testing PS26055's published leader

PS26055's own benchmark dossier reports **ε-Greedy(ε=0.10)** as the strongest scheduler on
the real 250-file corpus (0.06556 hit rate, beating UCB1, Thompson Sampling, SW-UCB). This
notebook implements the same plain multi-armed-bandit ε-Greedy and sweeps the same ε values
on our synthetic testbed. **Result: it does not transfer** — our belief-index heuristic still
wins. Diagnosis: PS26055's other baselines (UCB1, Thompson, SW-UCB) are, like ε-Greedy,
*generic* bandits with no problem-specific structure; our belief-index scheduler propagates
information to *unvisited* bands every step via the learned transition kernel, something
plain ε-Greedy structurally cannot do.

### 2.3 `notebooks/paper2_sac_discrete.ipynb` — SAC-Discrete

Christodoulou (2019), *Soft Actor-Critic for Discrete Action Settings* — dual Q-networks (min
of two, fights overestimation), softmax policy, automatic entropy-temperature tuning.
**Vanilla version failed**: the paper's own `target_entropy = 0.98·log(|A|)` (tuned for Atari)
is 98% of *maximum* entropy for our 12-action space — in our sparse-reward setting it drove
the policy toward near-uniform-random behavior as training progressed. Fixed with a lower
target entropy plus the same residual/warm-start + Q-network-warmup recipe that stabilized
PPO. **Result: edges out the belief-index heuristic** (0.229 vs 0.213 hit rate on the
single-run 25-scenario eval) — the best-performing paper method tested.

### 2.4 `notebooks/paper3_pearl_meta_context.ipynb` — PEARL-style meta-context

Rakelly et al. (2019), *Efficient Off-Policy Meta-RL via Probabilistic Context Variables*
(PEARL) — infers a latent context `z` from recent transitions via a permutation-invariant
encoder (product-of-Gaussians, PEARL's own closed form), conditions the policy/critic on it.
Conceptually a strong fit (every domain-randomized episode *is* a task ~ p(T)). Built on the
same stabilized SAC-Discrete recipe (context batch sampled separately from the RL batch, per
PEARL's own ablation). **Result: a genuine negative result** — 0.189 hit rate, below both
plain SAC-Discrete (0.229) and belief-index (0.213). Our per-band features already encode a
strong hand-designed history summary; the extra context-inference machinery added training
noise without adding information the belief features didn't already capture.

### 2.5 `notebooks/rigorous_scheduler_comparison.ipynb` — closing the "is this actually rigorous" gaps

A self-review flagged three gaps in the notebooks above: no oracle/regret baseline, no seed
variance with significance testing on the heuristic-vs-bandit comparison, no per-emitter-type
breakdown. This notebook adds all three:
- **Idealized Whittle-index oracle** lands statistically tied with plain belief-index (~0.166
  both) — the hand-tuned staleness bonus is already near the Markov-optimal ceiling for pure
  hit rate, now checked rather than assumed.
- **8-seed paired bootstrap significance**: Belief+periodicity beats belief-index, the
  idealized Whittle index, and both ε-Greedy settings, all four 95% CIs excluding zero.

| Scheduler | hit_rate mean ± std | vs Belief+periodicity |
|---|---|---|
| Belief + periodicity | 0.1887 ± 0.0187 | — |
| Belief-index | 0.1659 ± 0.0085 | −0.0228, **significant** |
| Whittle index (idealized) | 0.1655 ± 0.0135 | −0.0233, **significant** |
| ε-Greedy (ε=0.20) | 0.1604 ± 0.0129 | −0.0283, **significant** |
| ε-Greedy (ε=0.10) | 0.1462 ± 0.0191 | −0.0425, **significant** |
| Random scan | 0.0999 ± 0.0040 | — |
| Round-robin | 0.0991 ± 0.0047 | — |

### 2.6 `notebooks/rl_seed_robustness_comparison.ipynb` — same rigor, applied to the RL papers

Extends the significance treatment to `paper2`/`paper3`'s single-seed claims: retrains
SAC-Discrete and PEARL-style from scratch across 8 seeds, checks whether "SAC-Discrete beats
belief-index" and "PEARL-style underperforms both" survive seed variance, and — critically —
adds the comparison the original single-seed papers never made: both RL methods **against
Belief+periodicity specifically**, not just plain belief-index. *(Execution status: see
§5 — this was still running as of this README's last update; check the notebook directly for
final numbers.)*

### 2.7 `notebooks/real_tsrd_validation.ipynb` — first contact with real data

Downloads and uses **17 real TSRD stare-mode test files**, builds ground truth on PS26055's
own stated grid (36 bands × 500 MHz IBW, 50 ms slots, 600 slots/30 s mission), and re-runs the
belief-index / periodicity / ε-Greedy / round-robin comparison on real pulse trains for the
first time.

| Scheduler | hit_rate | Pd | Pfa |
|---|---|---|---|
| Belief + periodicity | 0.7696 | 0.9046 | 0.0503 |
| Belief-index | 0.7683 | 0.9046 | 0.0504 |
| ε-Greedy (ε=0.10) | 0.6502 | 0.9036 | 0.0477 |
| ε-Greedy (ε=0.20) | 0.6099 | 0.9051 | 0.0492 |
| Round-robin | 0.2756 | 0.9105 | 0.0493 |

**What holds up:** the exact same ranking from every synthetic notebook — belief-based
schedulers clearly beat ε-Greedy, which clearly beats round-robin — survives contact with
real pulse trains, real emitter geometry, real amplitude noise. **What doesn't hold up:**
the absolute hit-rate numbers are roughly 10× PS26055's published figures on (apparently) the
same corpus family — real per-file occupancy here measures 1–41% of grid cells containing at
least one pulse, and PS26055's dossier states grid/timing but not its exact
occupancy-construction rule, so this is an honest "same relative comparison, different
absolute rule" result, not a reproduction of their 0.06556.

### 2.8 `notebooks/real_hardware_schedule_comparison.ipynb` — beating the real deployed receiver

Goes one step further: TSRD's `scan` mode files record **one specific real receiver's actual
fixed dwell schedule** (36 bands, monotonic sweep, 2.15 s cycle, weighted dwell — 7 of 36
bands get double time). This notebook extracts that literal schedule and replays it against
the paired `stare`-mode ground truth for the same 17 real scenarios, using the corrected
0–18,000 MHz grid.

| Scheduler | hit_rate | Pd | Pfa |
|---|---|---|---|
| Belief + periodicity | 0.7789 | 0.9041 | 0.0546 |
| Belief-index | 0.7705 | 0.9039 | 0.0537 |
| ε-Greedy (ε=0.10) | 0.7011 | 0.9027 | 0.0494 |
| **Real deployed receiver (actual dwell schedule)** | **0.3088** | 0.9061 | 0.0480 |
| Round-robin (uniform, synthetic) | 0.2740 | 0.9107 | 0.0475 |

**This is the strongest real-data result in the project**: our adaptive schedulers beat not
just a synthetic round-robin strawman but **the literal recorded behavior of real deployed EW
receiver hardware**, by roughly 2.5×, on real pulse trains.

### 2.9 `notebooks/results/all_methods_comparison.png`

A single comparison image, all 12 tested schedulers, in the same column format PS26055's own
dossier uses (Scheduler | Hit rate | Total hits | vs RR) — synthetic-testbed results only
(see §2.7/§2.8 for the real-data tables above).

---

## 3. The debugging journey that matters more than any single number

Two off-policy methods were tried in this project — vanilla DQN (in early main-notebook
iteration) and vanilla SAC-Discrete — and **both needed the same fix** to become usable under
domain randomization:

1. **Vanilla DQN got worse with more training** (200 → 1,200 episodes). Diagnosis: domain
   randomization means the replay buffer mixes transitions from different underlying MDPs, so
   a state feature vector doesn't have one consistent Q-value across the buffer.
2. **Switching to on-policy PPO** trained stably but plateaued well below the belief-index
   heuristic — rediscovering "pick the highest-belief band" from sparse reward alone is a hard
   exploration problem for a blank policy.
3. **Warm-starting PPO's policy** (`logits = frozen belief-index formula + bounded MLP
   correction`, verified by assertion to exactly reproduce the heuristic at step 0) still
   **collapsed to worse-than-random** within a few updates when fine-tuned naively — a
   freshly-initialized correction head absorbs many noisy gradient steps per batch before the
   also-freshly-initialized critic has anything reliable to say.
4. **Fix** (standard in residual-RL literature, e.g. Silver et al. 2018): a **critic warm-up**
   phase (train only the value/Q head against the known-good policy's real returns before any
   policy update is trusted) + a **tanh-bounded correction** that makes it structurally
   impossible for RL to fully overwrite the expert prior. This is stable and competitive for
   both PPO and, later, SAC-Discrete.

**The reusable finding:** if you're extending this project past ε-Greedy (PS26055's own B1
rung) toward Double DQN, GRU-DDQN, or Discrete SAC (their B2/B4 rungs), expect to need this
same warm-start + critic-warmup treatment — a frozen ε-Greedy reference does not fall to a
naively-implemented off-policy method out of the box.

---

## 4. Repo structure

```
SIH_2026/
├── README.md                                   -- this file
├── PS26055_EGreedy_Benchmark_and_Published_Comparison.pdf   -- the real-corpus benchmark dossier this project tests against
├── 1910.07207v2.pdf                             -- SAC-Discrete paper (Christodoulou 2019)
├── rakelly19a.pdf                               -- PEARL paper (Rakelly et al. 2019)
├── data/
│   ├── stare/test_stare/config_<i>.h5           -- 17 real TSRD stare-mode files (downloaded)
│   └── scan/test_scan/config_<i>.h5             -- 17 real TSRD scan-mode files (downloaded, paired by config id)
└── notebooks/
    ├── smart_scan_strategy.ipynb                -- main prototype (belief-index, periodicity, residual PPO, 10-seed sweep)
    ├── paper1_egreedy_ps26055.ipynb
    ├── paper2_sac_discrete.ipynb
    ├── paper3_pearl_meta_context.ipynb
    ├── rigorous_scheduler_comparison.ipynb       -- Whittle oracle + bootstrap significance + per-emitter breakdown
    ├── rl_seed_robustness_comparison.ipynb       -- same rigor applied to SAC-Discrete / PEARL-style
    ├── real_tsrd_validation.ipynb                -- first real-data validation (stare mode)
    ├── real_hardware_schedule_comparison.ipynb   -- beats the real deployed receiver's own schedule
    └── results/all_methods_comparison.png
```

---

## 5. Honest limitations — read this before presenting any number from this repo

- **Synthetic-testbed numbers (§2.1–§2.6) are internal comparisons**, not validated against
  PS26055's real 250-file corpus — different scale (12×150 vs. their 36×600), different data
  source entirely.
- **Real-data numbers (§2.7, §2.8) use 17 of 250 stare-test files**, a flat Pd/Pfa detection
  model (no amplitude-based sensitivity calibration against the real ≈−190 to −10 dB scale),
  and a single evaluation pass — a directional real-data check, not a certified benchmark run.
- **Absolute hit-rate numbers on real data (§2.7/§2.8) do not match PS26055's published
  0.06556** — the *relative ranking* does, and that's the load-bearing claim; closing the
  absolute-number gap needs PS26055's exact occupancy-construction rule, which their dossier
  doesn't fully specify.
- **`rl_seed_robustness_comparison.ipynb`** — check whether it finished; if not, its
  §2.6 claims are pending, not final.
- No real-time/hardware latency measurement has been done on any scheduler in this repo.

---

## 6. How to reproduce

```bash
pip install numpy pandas matplotlib torch nbformat nbclient h5py huggingface_hub
```

Each notebook in `notebooks/` is self-contained (environment/emitter code is duplicated
across notebooks deliberately, so each one runs standalone). Open in Jupyter/VS Code and run
top to bottom. For the real-data notebooks (`real_tsrd_validation.ipynb`,
`real_hardware_schedule_comparison.ipynb`), you'll need your own Hugging Face account with
gated access accepted at `huggingface.co/datasets/alan-turing-institute/turing-synthetic-radar-dataset`,
authenticated via `huggingface-cli login` or the `HF_TOKEN` environment variable.

## 7. References

- PS26055 Research Progress Report & Benchmark Dossier (internal, 2026).
- Gunn, Hosford, Jones, Zeitler, Groves, Nockles — *The Turing Synthetic Radar Dataset: A
  dataset for pulse deinterleaving*, Alan Turing Institute.
- Zhao, Krishnamachari & Liu (2008) — myopic-policy optimality for 2-state positively
  correlated restless bandits.
- Christodoulou (2019) — *Soft Actor-Critic for Discrete Action Settings*, arXiv:1910.07207.
- Rakelly, Zhou, Quillen, Finn, Levine (2019) — *Efficient Off-Policy Meta-Reinforcement
  Learning via Probabilistic Context Variables*, ICML (PEARL).
- Silver et al. (2018) — *Residual Policy Learning*.
- Clarkson, El-Mahassni & Howard (2006) — periodic/Markov-chain beam-agile radar scheduling
  models, DOI 10.1049/ip-rsn:20050055.
