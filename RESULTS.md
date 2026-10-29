# First experimental ledger — 2026-10-29

## Protocol
CPU NumPy; 16D reservoirs unless otherwise noted; fixed random recurrent projections; linear ridge readout (+ intercept). All models within a seed see the same synthetic input histories, trained on 1,200 independently generated sequences, tested on 600 independent sequences; 5 independent seeds. Two-event task: 24 steps, two signed pulses of the same code vector separated by 3–5 ticks, then 2–5 silent/noisy ticks (hold); no event at last timestep. Equal frequencies of all four sign pairs; labels ask whether the two pulse signs match. Noise SD=.025, OOD noise SD=.0625 with 7–10 ticks after last pulse. Independent test streams with no tuning on OOD.

## Nonlinear historical comparison — mean accuracy

| Model | Test | Later and noisier OOD |
|---|---:|---:|
| Last input, linear head | 50.2% | 49.8% |
| Mean input, linear head | 50.8% | 48.5% |
| Exponential average, linear head | 50.9% | 49.3% |
| Mean plus quadratic products, linear head | **100.0%** | **100.0%** |
| Signed shift gated (16D) | 76.0% | 47.2% |
| Cyclic shift gated (16D) | 72.1% | 51.4% |
| Random orthogonal gated (16D) | 71.3% | 53.5% |
| Identity gated (16D) | **99.8%** | **82.3%** |
| Signed shift ungated (16D) | 49.6% | 49.7% |
| Signed shift exactly linear (16D) | 50.0% | 50.0% |
| Dual-path shift+identity gated (32D) | **99.9%** | **80.7%** |

**Causal input deletion:** remove all input before the second event in the held-out stream and apply the frozen trained head. Dual path accuracy goes to **49.9%**, roughly chance. This does not imply the machine has a memory mechanism that biology shares. Strong negative: the simple quadratic-mean baseline is perfectly accurate, so the recurrence is not necessary for the task in general. The signed-gated model fails the OOD transfer.

## Earlier symbol recall (lag 6 of 12) — 5-seed mean

| Model | Test |
|---|---:|
| Last input | 50.0% |
| Mean input | 61.0% |
| Identity gated | 64.9% |
| Signed gated | 74.7% |
| Cyclic gated | 71.6% |
| Signed ungated | **98.7%** |
| Signed exactly linear | **100.0%** |
| Dual path | **100.0%** |

### Interpretation and unresolved questions

The rotating path provides temporal separation for independent older symbols, while a stationary nonlinear gate provides multiplicative binding of two pulses. Combining the two preserves both capabilities, but uses twice the state width. A downstream quadratic decoder can also perform binding; no efficiency advantage has been established against fair-budget trained recurrent networks, SSMs or Wave-RNNs.

Experiments are all synthetic, with fixed input encoding, no true sensory scenes, no trained video encoder, no explicit seconds-based phase propagation, no physical waves, and no neural data. The binary-symbol tests are valuable diagnostics, not machine understanding of time. Do not claim novelty, biological implementation or extraordinary memory compression.

Run the experiment using `python benchmark.py --seeds 5`, `python lag_benchmark.py` and `python -m unittest discover -s tests -v`.

## Predicting an unseen next event

A second independent synthetic benchmark was added in `predict_next.py`. Generator: balanced two-step stochastic Markov rule (90% copy from `t-2`, 10% flip), 16 sequential observations each containing the signed vector plus Gaussian noise SD=.02, then forecast the next symbol. Five seeds, independent 1,200/600 train/test sequences per seed.

| Model | Five-seed mean |
|---|---:|
| Latest input only | 51.2% |
| Exponential average (decay .75) | 61.3% |
| Preceding event oracle (two-frame buffer) | **91.1%** |
| Signed exactly linear recurrent state | **91.1%** |
| Identity nonlinear gate | 86.5% |
| Signed nonlinear gate | **91.1%** |
| Dual path 32D | **91.1%** |

A state captures the short history needed to forecast, but a standard two-frame buffer does just as well here. This experiment supports a practical learning API, not an architectural breakthrough. Results are neither a Wave-RNN reproduction nor an evaluation on real sensory events.
