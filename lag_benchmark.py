"""Counterpart to XOR: recover an individual bit 6 steps ago."""
import numpy as np
from tasks import lag_stream
from temporal_ai import Config,TemporalLoop,LinearHead,DualPathMemory

MODES={'signed_gated':('signed_shift',1.2,'tanh'),
       'cyclic_gated':('cyclic',1.2,'tanh'),
       'identity_gated':('identity',1.2,'tanh'),
       'signed_no_gate':('signed_shift',0.,'tanh'),
       'signed_linear':('signed_shift',0.,'linear')}

def run(seeds=5,train=1200,test=600):
    scores={k:[] for k in ['last_input','mean_input',*MODES,'dual_path']}
    for i in range(seeds):
        x,y=lag_stream(n=train,seed=2026+i*103)
        xt,yt=lag_stream(n=test,seed=2026+i*103+50000)
        for name,a,b in [('last_input',x[:,-1],xt[:,-1]),('mean_input',x.mean(1),xt.mean(1))]:
            scores[name].append(LinearHead().fit(a,y).accuracy(b,yt))
        for name,(geom,gate,activation) in MODES.items():
            net=TemporalLoop(Config(geometry=geom,gate=gate,activation=activation,seed=777+i*103))
            head=LinearHead().fit(net.encode(x),y)
            scores[name].append(head.accuracy(net.encode(xt),yt))
        dual=DualPathMemory(seed=777+i*103)
        head=LinearHead().fit(dual.encode(x),y)
        scores['dual_path'].append(head.accuracy(dual.encode(xt),yt))
    return {name:dict(mean=float(np.mean(v)),sd=float(np.std(v,ddof=1)) if len(v)>1 else 0.) for name,v in scores.items()}

if __name__=='__main__':
    for n,x in run().items():print(f'{n:<20} {x["mean"]:6.1%} ± {x["sd"]:5.1%}')
