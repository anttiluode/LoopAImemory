# LoopAImemory — a little AI with a fading present

**Status: experimental, CPU-only, reproducible.** This repository explores whether a bounded continuous state can make the *recent past* available to a trained AI **without a growing list of snapshots**.

Origin: [MoebiusLoopMemory](https://github.com/anttiluode/MoebiusLoopMemory) and the [Perception Lab ECG feedback experiment](https://github.com/anttiluode/GeometricNeuronOriginReview). Those established *different* ideas: the former is a recurrent numerical state; the latter is a homeostatic measurement-feedback loop. Here we test whether a downstream **learned readout** can use such states.

## Quick start (Windows, Python 3.13)

~~~powershell
git clone https://github.com/anttiluode/LoopAImemory.git
cd LoopAImemory
python3.13 -m pip install -r requirements.txt
python3.13 demo.py --first 1 --second -1 --hold 3
python3.13 benchmark.py --seeds 5 --train 1200 --test 600 --output results.json
python3.13 lag_benchmark.py
python3.13 -m unittest discover -s tests -v
~~~

Change `--first` and `--second` to +1 or -1. The demo trains *only* a linear output head to answer whether two previous events match after they have both disappeared. It reports independent held-out accuracy, too. No LLM or webcam is required.

## The architecture

A single 16D loop uses the same recurrence as MoebiusLoopMemory:

~~~text
h = R @ state
b = B @ input
next_state = tanh(leak * h + b + gate * b * tanh(A @ h))
~~~

`R` can be signed shift, unsigned cyclic shift, random orthogonal matrix or identity. `A,B` are fixed random projections. **Only the linear readout is trained** (ridge regression). This is reservoir-style supervised learning, not end-to-end optimization.

**DualPathMemory (32D)** concatenates two *independent* paths fed by the same input:
- A 16D **linear signed-shift register** for earlier-bit recall (`gate=0`, linear activation).
- A 16D **stationary nonlinear contextual binder** for event comparisons (`gate=1.2`, `tanh`).

This division is *empirical*: neither path alone was best on both tested tasks. **It does not demonstrate that signed/Möbius topology is necessary**, and does not imply the cortex has such a register.

## Experimental results, five seeds (local Python run, 2026-10-29)

Task A: balanced, noisy, antipodal two-event stream. The two events stop before the final input. Train on hold 2–5 steps; test on independent trials; OOD uses hold 7–10 and **2.5× higher noise**. Scores are averages across 5 seeds, each with **1,200 training and 600 held-out examples**.

| Model | Held-out two-event equality | Later + noisier OOD |
|---|---:|---:|
| Last input, linear readout | 50.2% | 49.8% |
| Mean input, linear readout | 50.8% | 48.5% |
| Exponential average, linear readout | 50.9% | 49.3% |
| **Mean input + all quadratic products**, linear readout | **100.0%** | **100.0%** |
| Signed rotating loop, gate | 76.0% | 47.2% |
| Cyclic rotating loop, gate | 72.1% | 51.4% |
| Orthogonal rotating loop, gate | 71.3% | 53.5% |
| Identity recurrence, gate | **99.8%** | **82.3%** |
| Signed rotating loop, no gate | 49.6% | 49.7% |
| Signed **linear** loop | 50.0% | 50.0% |
| **DualPathMemory, 32D** | **99.9%** | **80.7%** |

**Do not over-interpret**: the quadratic-mean comparator trivially solves this synthetic task, so *neither the gate nor the rotation is required for this function in general*. The nonlinear gate makes some relational information accessible to a **linear decoder**, which is a different, narrower claim. The rotated gated loop fails to generalize to the unseen later/noisier range. The stationary gate is the best 16D model here.

On a causal input-ablation test that removes the first event (and all earlier input) while holding the learned readout fixed, the dual path falls to ~49.9% chance. That supports dependency on the earlier cue, not biological causality.

Task B: recover the binary input **six steps earlier** from only the final state of a 12-step stream. Five-seed means:

| Model | Lag-6 accuracy |
|---|---:|
| Last input | 50.0% |
| Mean input | 61.0% |
| Identity gated | 64.9% |
| Signed gated | 74.7% |
| Signed no gate | 98.7% |
| Signed exactly linear | **100.0%** |
| **Dual path** | **100.0%** |

The high linear-recall scores are expected for a 16D noisily encoded 12-step binary stream with fixed random projections. This is **not** evidence of extraordinary compression capacity or superior general-world memory.


## New: predict the next event

Run `python3.13 predict_next.py` to train on a synthetic *second-order* sequence. The stream is constructed with P(next symbol = symbol two observations before next) = 90%. The latest frame alone cannot tell the answer. We train a linear predictor on the final history state, no symbolic knowledge of the generator in the readout.

Five-seed means (1,200 training and 600 independent test sequences per seed):

| Predictor | Accuracy |
|---|---:|
| Latest event alone | 51.2% |
| Exponential moving average | 61.3% |
| Exact preceding-two-step event, oracle | **91.1%** |
| Signed linear loop | **91.1%** |
| Signed gated loop | **91.1%** |
| Identity gated loop | 86.5% |
| Dual-path 32D | **91.1%** |

The loop learned a useful history-to-forecast map with a linear head. However, the **two-frame oracle ties it**. No claim of advantage is justified without harder tasks, comparable budgets, or robustness tests. This is an example of predictive sequence learning, not general temporal understanding.

## Boundaries of the experiment

- Toy vector events, not real webcam gestures, raw wave propagation, language understanding or neuron measurements.
- Fixed known stimulus codebook in train and test; no object, lighting or scene generalization.
- A full explicit history with a quadratic decoder is a strong alternative. The 32D dual path has more compute/state than the 16D baselines.
- `TimeAwareLoop` changes the exponential decay using elapsed wall time but *still rotates once per observation*. It does not implement a physical continuously evolving wave; a proper continuous-time operator is future work.
- The geometric signed shift has no demonstrated special advantage over simple cyclic recurrence.
- Related prior research: [Keller et al. Wave-RNN (ICLR 2024)](https://proceedings.iclr.cc/paper_files/paper/2024/hash/374fc6371864f800f213bf8a248f1117-Abstract-Conference.html), [Karuvally et al. hidden traveling waves (ICML 2024)](https://proceedings.mlr.press/v235/karuvally24a.html), and [Mamba state-space models](https://github.com/state-spaces/mamba). This repo **does not reproduce or outperform** any of these models.

## Next milestones

1. Train and test on **real gesture sequences** from the webcam, with identical final static frames and **disjoint recording sessions**.
2. Train a **future-event prediction head**, not only binary historical equality; compare predictive loss to EMA, full-history linear/quadratic readouts, GRU and Wave-RNN at matched budgets.
3. Handle true irregular time with calibrated continuous-time propagation `R(delta_t)`, not one rotation per received frame.
4. Test contextual `write/read/erase` interventions and generalization to changing symbol codebooks.
5. Explore neural links by comparing formal phase/probe predictions with independent cortex recordings, not visual similarity between heatmaps.

**The research question:** Can a compact history-dependent state become an efficient, robust and trainable interface between continuously sensed events and decisions made later? This repository is where we can find out.
