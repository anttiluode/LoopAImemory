"""Minimal task-driven AI from a fixed circulating history plus a trained linear head.

No physical traveling wave, brain model, or uniquely Moebius topological claim.
Matches the MoebiusLoopMemory core recurrence under standard 1-step updates.
"""
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class Config:
    dim: int = 16
    leak: float = 0.92
    gate: float = 1.2
    geometry: str = 'signed_shift'
    activation: str = 'tanh'
    seed: int = 99


def shift_matrix(dim: int, geometry: str, rng: np.random.Generator) -> np.ndarray:
    if geometry == 'identity':
        return np.eye(dim)
    if geometry in ('signed_shift', 'cyclic'):
        m = np.roll(np.eye(dim), 1, axis=0)
        if geometry == 'signed_shift':
            m[0, -1] = -1.
        return m
    if geometry == 'orthogonal':
        q, t = np.linalg.qr(rng.normal(size=(dim, dim)))
        q *= np.where(np.diag(t) < 0., -1., 1.)
        return q
    raise ValueError(geometry)


class TemporalLoop:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        rng = np.random.default_rng(cfg.seed)
        d = cfg.dim
        self.R = shift_matrix(d, cfg.geometry, rng)
        self.A = rng.normal(size=(d,d))/np.sqrt(d)
        self.B = rng.normal(size=(d,d))/np.sqrt(d)

    def step(self, state: np.ndarray, inp: np.ndarray) -> np.ndarray:
        h = np.matmul(state, self.R.T)
        b = np.matmul(inp, self.B.T)
        z = self.cfg.leak * h + b + self.cfg.gate*b*np.tanh(h @ self.A.T)
        return np.tanh(z) if self.cfg.activation == 'tanh' else z

    def encode(self, streams: np.ndarray, erase_at: int | None = None) -> np.ndarray:
        """(batch,time,dim) -> (batch,dim). Only final recurrent state retained."""
        streams = np.asarray(streams, dtype=np.float64)
        if streams.ndim != 3 or streams.shape[-1] != self.cfg.dim:
            raise ValueError('Expected (batch,steps,dim) inputs')
        s = np.zeros((streams.shape[0], self.cfg.dim))
        for t in range(streams.shape[1]):
            if erase_at is not None and t == erase_at:
                s[:] = 0
            s = self.step(s, streams[:, t])
        return s

    def predict_state(self, stream: np.ndarray) -> np.ndarray:
        if stream.shape != (self.cfg.dim,):
            raise ValueError('Expected a single input vector')
        return self.step(np.zeros(self.cfg.dim), stream)


class LinearHead:
    """Only this readout learns. Ridge regression with an unpenalized bias."""
    def __init__(self, ridge: float = 1e-2):
        self.ridge = ridge
        self.weights = None

    def fit(self, x, y):
        x = np.asarray(x, dtype=np.float64)
        z = np.column_stack([x, np.ones(len(x))])
        reg = np.eye(z.shape[1]) * self.ridge
        reg[-1,-1] = 0.
        self.weights = np.linalg.solve(z.T@z+reg,z.T@y)
        return self

    def score(self, x):
        if self.weights is None:
            raise RuntimeError('Fit before using the head')
        return x@self.weights[:-1]+self.weights[-1]

    def predict(self, x):
        return np.where(self.score(x)>=0, 1, -1)

    def accuracy(self,x,y):
        return float(np.mean(self.predict(x)==y))


class TimeAwareLoop:
    """Exploratory real-time version: elapsed-time decay but discrete rotation per event.
    NOT a physically correct variable-time propagation operator; use only as
    simple robustness comparator and do not infer phase from elapsed wall time.
    """
    def __init__(self, loop: TemporalLoop, tau: float = 8.3):
        self.loop = loop
        self.tau = tau
        self.state = np.zeros(loop.cfg.dim)

    def reset(self):
        self.state.fill(0.)

    def advance(self,u,dt=1.):
        if dt<0:
            raise ValueError('Negative elapsed time')
        decay = np.exp(-dt/self.tau)
        h = self.loop.R @ self.state
        basal = self.loop.B @ u
        z = decay*h + basal + self.loop.cfg.gate * basal*np.tanh(self.loop.A@h)
        self.state = np.tanh(z) if self.loop.cfg.activation == 'tanh' else z
        return self.state.copy()


class DualPathMemory:
    """Research architecture: 16D chronological register + 16D nonlinear binder.
    These are independent 16D states. Both readouts receive the same inputs;
    only linear output weights are trained.
    """
    def __init__(self, dim=16, seed=99):
        self.timeline = TemporalLoop(Config(dim=dim,gate=0.,geometry='signed_shift',activation='linear',seed=seed))
        self.binder = TemporalLoop(Config(dim=dim,gate=1.2,geometry='identity',activation='tanh',seed=seed))
        self.dim=dim

    def encode(self, streams):
        return np.column_stack((self.timeline.encode(streams), self.binder.encode(streams)))

    def step(self, state, inp):
        return np.concatenate((self.timeline.step(state[..., :self.dim],inp),
                               self.binder.step(state[..., self.dim:],inp)),axis=-1)
