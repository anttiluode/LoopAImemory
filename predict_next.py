"""Can a compact recurrent state forecast the next event?

Synthetic second-order process: with 90% probability x[t+1]=x[t-1].
The latest frame x[t] is uninformative by design. A two-frame oracle
is a strong, simple baseline; this is NOT a novel predictive task.
"""
import numpy as np
from temporal_ai import Config, TemporalLoop, DualPathMemory, LinearHead


def streams(n=1200, steps=16, seed=0, flip=.1, dim=16, noise=.02):
    rng=np.random.default_rng(seed)
    b=np.empty((n,steps+1),dtype=int)
    b[:,:2]=rng.choice([-1,1],size=(n,2))
    for t in range(2,steps+1):
        flips=rng.random(n)<flip
        b[:,t]=np.where(flips,-b[:,t-2],b[:,t-2])
    code=np.random.default_rng(1618).normal(size=dim)
    code*=.60/(np.linalg.norm(code)/np.sqrt(dim))
    x=b[:,:-1,None]*code[None,None,:]+rng.normal(scale=noise,size=(n,steps,dim))
    y=b[:,-1]
    return x,y


def evaluate(seeds=5,train=1200,test=600):
    models=['current','ema','previous_frame_oracle','signed_linear','identity_gated','signed_gated','dual_path']
    out={m:[] for m in models}
    for k in range(seeds):
        x,y=streams(n=train,seed=2026+110*k)
        xt,yt=streams(n=test,seed=2026+110*k+50000)
        features={
            'current': (x[:,-1,:],xt[:,-1,:]),
            'ema':(sum(x[:,t,:]*.75**(x.shape[1]-1-t) for t in range(x.shape[1])),
                   sum(xt[:,t,:]*.75**(xt.shape[1]-1-t) for t in range(xt.shape[1]))),
            'previous_frame_oracle':(x[:,-2,:],xt[:,-2,:]),
        }
        for m,(a,b) in features.items():
            out[m].append(LinearHead().fit(a,y).accuracy(b,yt))
        for m,cfg in {
            'signed_linear': Config(gate=0,activation='linear',seed=707+k*110),
            'identity_gated': Config(geometry='identity',gate=1.2,seed=707+k*110),
            'signed_gated': Config(gate=1.2,seed=707+k*110)
        }.items():
            net=TemporalLoop(cfg)
            out[m].append(LinearHead().fit(net.encode(x),y).accuracy(net.encode(xt),yt))
        net=DualPathMemory(seed=707+k*110)
        out['dual_path'].append(LinearHead().fit(net.encode(x),y).accuracy(net.encode(xt),yt))
    return {m:{'mean':float(np.mean(v)),
               'sd':float(np.std(v,ddof=1)) if seeds>1 else 0.} for m,v in out.items()}


if __name__=='__main__':
    print('Next-event forecast: P(x[next]=x[previous])=90%; chance=50%; previous-frame oracle ~90%')
    for m,stat in evaluate().items():
        print(f'{m:<24} {stat["mean"]:6.1%} ± {stat["sd"]:5.1%}')
