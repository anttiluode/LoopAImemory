# Real temporal vision: a 16-number fading memory

This is an experiment, not a verified real-world gesture recognizer. Training requires your own webcam sessions. Only the **linear readout** is fitted; the random 16D visual projection and memory dynamics are fixed.

## Install (Python 3.13, Windows)

```powershell
python3.13 -m pip install -r requirements-vision.txt
```

## Try without a camera

```powershell
python3.13 vision_train.py --simulate --clips-per-session 140 --data synthetic_vision.npz --model models/synthetic_loop.npz
python3.13 -m unittest discover -s tests -v
```

Synthetic clips move a bright patch left/right, then return to the *same final background*. The trained classifier sees only the **final recurrent state** of 30 compressed observations.

## Record real gestures

```powershell
python3.13 webcam_gestures.py record --session day1 --data gestures.npz --fps 8
python3.13 webcam_gestures.py record --session day2 --data gestures.npz --fps 8
python3.13 vision_train.py --data gestures.npz --test-session day2 --model models/gesture_loop.npz --model-name signed_linear
python3.13 webcam_gestures.py predict --model models/gesture_loop.npz --fps 8
```

In the recorder, `L` starts a labeled **left-to-right** clip and `R` a **right-to-left** clip. The label describes motion in the *displayed camera image*. Press `Q` to quit. Allow 1.5 seconds for countdown, then move across the image early, remove your hand, and leave the final part quiet. Aim for **15–25 clips per direction per session**. Capture `day2` separately, ideally with varied lighting and distance. Keep `--mirror` and `--fps` settings the same when recording and predicting; if using mirror, apply it in all commands. For another camera use `--source 1`.

The `predict` mode uses **Space** to capture one unlabeled 30-frame sequence. It shows a signed linear margin (not a probability) and a directional classification after capture ends. It does not detect actions continuously, and it does not learn from the prediction trials.

Store the `.npz` files locally. They contain 16D features, labels, and recording-session names, **not raw photos or video**. Projection is lossy, but data can still be sensitive; do not publish it casually. The camera feed is never transmitted by the program.

## What is and isn't demonstrated

The numerical loop follows exactly the signed linear recurrence from the existing `temporal_ai.py` implementation: `state_next=0.92*R*state+B*u`. The nonlinear gated and dual-path alternatives also remain available through `--model-name signed_gate`, `identity_gate`, and `dual`. All are compared using the same session-disjoint split and trained linear output heads.

- Input is a **16×16 difference image** projected by a fixed, seeded 256-to-16 matrix. It discards most information.
- The last scene is stationary in the synthetic benchmark. `last_change` baseline uses only the latest compressed difference signal; `explicit_history` can access every 16D input in the clip and is a strong comparator.
- The loop's useful behavior here does **not** require special Möbius topology or a nonlinear apical gate. Signed and ordinary recurrence comparisons in the original repository did not establish a special advantage.
- Synthetic sessions are independent generator seeds and modest noise changes, *not real webcam domain generalization*. The real evaluation requires recording two or more disjoint sessions.
- Frame-index memory is not true physical time. The nominal leak factor is per processed frame; variable frame rates can change the number of seconds represented.
- Simple full-history models solve the synthetic motion task too, so currently we have a compact implementation, **not a proof of superior performance**.

## Next experiments

1. Evaluate on genuine camera clips with disjoint recording days and changed viewpoint, but matched gesture labels.
2. Add a small online temporal-event parser that reports `event_start`, `direction`, and `event_end` to a local language model, *only after verifying predictive accuracy*.
3. Benchmark GRUs, Wave-RNN and temporal CNNs on the same real sessions, matched for parameters and state size.
4. Replace frame-step rotation with a true elapsed-time operator. Compare response after standardized delays and explicit state erasures.