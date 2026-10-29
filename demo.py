"""Train a 32D streaming AI to classify two vanished events.

The trained linear head sees only the final recurrent state. No equality
logic is used during inference.
"""
import argparse
import numpy as np
from temporal_ai import DualPathMemory, LinearHead
from tasks import generate

def run(first=1,second=-1,hold=3,seed=2026):
    if first not in (-1,1) or second not in (-1,1):
        raise ValueError('first/second must be -1 or +1')
    if not 0<=hold<=12: raise ValueError('hold between 0 and 12')
    rng=np.random.default_rng(seed)
    x,y,_=generate(2400,seed=seed)
    xt,yt,_=generate(800,seed=seed+100003)
    net=DualPathMemory(seed=seed)
    head=LinearHead().fit(net.encode(x),y)
    validation=head.accuracy(net.encode(xt),yt)
    code=np.random.default_rng(1618).normal(size=16)
    code*=.65/(np.linalg.norm(code)/np.sqrt(16))
    inputs=rng.normal(scale=.025,size=(1,24,16))
    t2=23-hold
    t1=t2-4
    if t1<1: raise ValueError('cue too early')
    inputs[0,t1]+=first*code
    inputs[0,t2]+=second*code
    result=int(head.predict(net.encode(inputs))[0])
    print(f'First event: {first:+d} at t={t1}. Second: {second:+d} at t={t2}.')
    print('Current input is noise, no event visible at final frame.')
    print('AI remembers:', 'SAME' if result==1 else 'DIFFERENT')
    print('Correct label:', 'SAME' if first==second else 'DIFFERENT')
    print('Held-out validation:',f'{validation:.1%}')
    return result, validation

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--first',type=int,choices=(-1,1),default=1)
    p.add_argument('--second',type=int,choices=(-1,1),default=-1)
    p.add_argument('--hold',type=int,default=3)
    a=p.parse_args()
    run(a.first,a.second,a.hold)
