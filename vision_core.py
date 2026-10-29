"""Small streaming visual encoder and synthetic gesture benchmark.

Only a fixed 16x16 change image and a random 256->16 projection feed
LoopAImemory. Nothing here recognizes people, faces, or gestures a priori.
"""
from dataclasses import dataclass
import math
import numpy as np

GRID = 16
DIM = 16
STEPS = 30


@dataclass(frozen=True)
class VisionConfig:
    dim: int = 16
    gain: float = 4.0
    deadband: float = 0.025
    seed: int = 2026


class FrameToVector:
    def __init__(self, config: VisionConfig = VisionConfig()):
        self.config = config
        rng = np.random.default_rng(config.seed)
        self.projection = rng.normal(size=(GRID * GRID, config.dim)) / math.sqrt(GRID * GRID)
        self.previous = None

    def reset(self):
        self.previous = None

    def push_gray(self, gray: np.ndarray) -> np.ndarray:
        """Accept grayscale 16x16 array in [0,1]. First frame is reference."""
        gray = np.asarray(gray, dtype=float)
        if gray.shape != (GRID, GRID):
            raise ValueError('Expected 16x16 grayscale float image')
        gray = np.clip(gray, 0., 1.)
        signal = (gray - self.previous) if self.previous is not None else np.zeros_like(gray)
        signal[np.abs(signal) < self.config.deadband] = 0.
        self.previous = gray.copy()
        return self.config.gain * (signal.ravel() @ self.projection)

    def push_frame(self, frame: np.ndarray) -> np.ndarray:
        import cv2
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        small = cv2.resize(gray, (GRID, GRID), interpolation=cv2.INTER_AREA)
        return self.push_gray(small.astype(float) / 255.0)


def synth_clip(rng: np.random.Generator, direction: int, steps: int = STEPS,
               difficulty: float = 1.0) -> np.ndarray:
    """Synthetic moving bright patch followed by silence. Both directions
    have the same motion amount and final static frame distribution.
    Output is (steps,16,16) of analog gray levels, not stored video.
    """
    if direction not in (-1, 1) or steps < 22:
        raise ValueError('direction +/-1, steps >=22 required')
    y = rng.uniform(4., 12.)
    radius = rng.uniform(.9, 1.8)
    brightness = rng.uniform(.35, .65)
    bg = rng.uniform(.08, .23, (16, 16))
    noise = .006 * difficulty
    out = np.empty((steps, 16, 16), dtype=np.float64)
    yy, xx = np.mgrid[:16, :16]
    for t in range(steps):
        frame = bg + rng.normal(0, noise, size=(16, 16))
        if 3 <= t <= 15:
            frac = (t-3)/12
            center_x = (2.1+11.8*frac) if direction == 1 else (13.9-11.8*frac)
            frame += brightness * np.exp(-((xx-center_x)**2+(yy-y)**2)/(2*radius**2))
        out[t] = np.clip(frame, 0, 1)
    return out


def synth_dataset(n_per_session: int = 120, sessions: int = 4,
                  steps: int = STEPS, seed: int = 20261029):
    """Entirely disjoint randomized sessions; balanced directions per session.
    Each sample is compressed online to a 16D vector per frame.
    """
    if n_per_session % 2 or n_per_session < 2:
        raise ValueError('Need even n_per_session >=2')
    streams, labels, ids = [], [], []
    encoder = FrameToVector()
    for session in range(sessions):
        rng = np.random.default_rng(seed + session*701)
        yy = np.array([-1, 1]*(n_per_session//2))
        rng.shuffle(yy)
        for direction in yy:
            movie = synth_clip(rng, int(direction), steps, difficulty=1.+session*.15)
            encoder.reset()
            streams.append(np.stack([encoder.push_gray(frame) for frame in movie]))
            labels.append(direction)
            ids.append(session)
    return np.stack(streams), np.asarray(labels, dtype=int), np.asarray(ids, dtype=int)


def save_dataset(path, x, y, session, **metadata):
    """Only compressed features, labels, and session metadata; no images."""
    np.savez_compressed(path, x=np.asarray(x, dtype=np.float32),
                        y=np.asarray(y, dtype=np.int8),
                        session=np.asarray(session, dtype='U50'),
                        **metadata)


def load_dataset(path):
    with np.load(path, allow_pickle=False) as d:
        x,y,session = d['x'].astype(float), d['y'].astype(int), d['session'].astype(str)
    if x.ndim != 3 or x.shape[-1] != DIM or len(y) != len(x) or len(session) != len(x):
        raise ValueError('Invalid compressed gesture data dimensions')
    if not np.isin(y, [-1, 1]).all() or not np.isfinite(x).all():
        raise ValueError('Invalid data or class labels')
    return x,y,session
