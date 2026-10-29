# Visual temporal memory — 2026-10-29

**Question:** Can a 16-number current state be used to recognize motion after it ends, with a learned linear head, without giving the readout the whole video?

## Reproduction

    python3.13 vision_train.py --simulate --clips-per-session 140 --data synthetic_vision.npz --model models/synthetic_loop.npz
    python3.13 -m unittest discover -s tests -v

Four independently randomized synthetic sessions; 140 clips per session, balanced left-to-right/right-to-left. Thirty 16x16 grayscale frames per clip, one moving bright patch visible only in frames 3–15, then stationary background with weak noise. Frame differences are projected by a fixed seeded matrix from 256 pixels to 16 features. Training 420 clips, test session 3 held out entirely (140 clips). Each model has a trained regularized linear ridge readout, except that the feature extractor varies.

| Features | Training | Held-out |
|---|---:|---:|
| Last compressed frame **difference** | 58.6% | 58.6% |
| Last three differences | 61.7% | 52.1% |
| Mean of changes | 89.3% | 79.3% |
| EMA of changes | 86.9% | 72.1% |
| 16D signed linear loop | 100.0% | **100.0%** |
| 16D signed gated loop | 100.0% | **100.0%** |
| 16D stationary gate | 96.9% | 87.1% |
| 32D dual-path loop | 100.0% | **100.0%** |
| 480D full compressed history | 100.0% | **100.0%** |

The loops and full history solve this simple synthetic task. This does **not** establish a real-webcam result, superiority over explicit video buffers or a unique Möbius/apical mechanism. The simpler **linear 16D loop** matches the dual-path system and is therefore the default model. Different generator seeds are not the same as independent physical recording days. All inference is per frame but the displayed class decision is after each 30-frame capture; no continuous event parser yet. Timing is frame-index based, not elapsed seconds.

Next: collect real gestures on at least two occasions, hold out an entire session, compare same models and a full-frame or optical-flow baseline, then test under changed light, speed, view and delays. The saved .npz stores numerical features only, not camera images; treat it as potentially sensitive nevertheless.