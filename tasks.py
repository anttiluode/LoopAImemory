"""Counterbalanced, same-final-input temporal reasoning datasets.

A cue is a signed vector pulse, not a literal image or EEG measurement.
Test samples are independent, all classes equally numerous, and the final
frames are silence plus label-independent nuisance. These toy tasks probe
whether a simple readout can access nonlinear history information.
"""
import numpy as np


def generate(n=1600, steps=24, dim=16, seed=1, task='xor', noise=0.025,
             lag_range=(2, 5), fixed_code=None):
    if n % 4: raise ValueError('n must be divisible by 4')
    if steps < 15: raise ValueError('steps must be >= 15')
    rng=np.random.default_rng(seed)
    code=np.asarray(fixed_code if fixed_code is not None else np.random.default_rng(1618).normal(size=dim),dtype=float)
    code *= .65/(np.linalg.norm(code)/np.sqrt(dim))
    signs = np.tile(np.array([[-1,-1],[-1,1],[1,-1],[1,1]]), (n//4,1))
    rng.shuffle(signs, axis=0)
    x=rng.normal(0,noise,(n,steps,dim))
    # The two pulses end before the final frame. All four class combinations
    # are equally common, so the last input has no task signal.
    gap = rng.integers(3, 6, size=n)
    hold=rng.integers(lag_range[0],lag_range[1]+1,size=n)
    second=steps-1-hold
    first=second-gap
    if first.min() < 1: raise ValueError('lag range too long for steps')
    for i in range(n):
        t2=int(second[i])
        t1=int(first[i])
        x[i,t1] += signs[i,0]*code
        x[i,t2] += signs[i,1]*code
    if task=='xor':
        y=signs[:,0]*signs[:,1]
    elif task=='first':
        y=signs[:,0]
    elif task=='second':
        y=signs[:,1]
    else:
        raise ValueError(task)
    return x, y.astype(int),dict(first=first,second=second,hold=hold,signs=signs)


def counterbalanced_events(n=1600, steps=24, dim=16, seed=1,
                           noise=.025, gap_range=(3,6), blank=4):
    """A->B vs B->A: exactly two opposing pulses, equal composition.
    Paired examples use identical noise, positions and pulse energies.
    Strong linear rotated reservoir should solve this, without any apical gate.
    """
    if n % 2: raise ValueError('n must be even')
    rng=np.random.default_rng(seed)
    code=np.random.default_rng(1618).normal(size=dim)
    code*=.65/(np.linalg.norm(code)/np.sqrt(dim))
    x=np.zeros((n,steps,dim))
    y=np.tile(np.array([1,-1]),n//2)
    for pair in range(n//2):
        a=rng.integers(2,steps-blank-8)
        b=a+rng.integers(gap_range[0],gap_range[1]+1)
        nuisance=rng.normal(scale=noise,size=(steps,dim))
        x[2*pair]=nuisance.copy()
        x[2*pair+1]=nuisance.copy()
        x[2*pair,a]+=code
        x[2*pair,b]-=code
        x[2*pair+1,a]-=code
        x[2*pair+1,b]+=code
    return x,y


def lag_stream(n=1200, steps=12, dim=16, seed=1, lag=6, noise=.01):
    """Input train of independently balanced bits; recover an earlier bit."""
    if lag>=steps: raise ValueError('lag too long')
    rng=np.random.default_rng(seed)
    code=np.random.default_rng(1618).normal(size=dim)
    code*=.60/(np.linalg.norm(code)/np.sqrt(dim))
    bits=rng.choice([-1,1],size=(n,steps))
    x=bits[:,:,None]*code[None,None,:] + rng.normal(0,noise,(n,steps,dim))
    return x,bits[:,-1-lag]
