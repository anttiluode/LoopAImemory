"""Reproducible CPU-only nonlinear temporal comparison benchmark.

Run: python benchmark.py --seeds 5 --train 1200 --test 600
"""
import argparse
import json
from pathlib import Path
import time
import numpy as np
from tasks import generate
from temporal_ai import Config, TemporalLoop, LinearHead, DualPathMemory

MODELS={
    'signed_gated': dict(geometry='signed_shift',gate=1.2),
    'cyclic_gated': dict(geometry='cyclic',gate=1.2),
    'orthogonal_gated': dict(geometry='orthogonal',gate=1.2),
    'identity_gated': dict(geometry='identity',gate=1.2),
    'signed_no_gate': dict(geometry='signed_shift',gate=0.),
    'signed_linear': dict(geometry='signed_shift',gate=0.,activation='linear'),
}

def assess(seed,n_train,n_test,noise=.025):
    xtr,ytr,_=generate(n_train,seed=seed,noise=noise)
    xte,yte,meta_te=generate(n_test,seed=seed+100003,noise=noise,lag_range=(2,5))
    xood,yood,_=generate(n_test,seed=seed+200003,noise=noise*2.5,lag_range=(7,10))
    baselines={}
    def quadratic_mean(v):
        mean=v.mean(axis=1)
        iu=np.triu_indices(mean.shape[1])
        return np.column_stack([mean, (mean[:,:,None]*mean[:,None,:])[:,iu[0],iu[1]]])
    for name,features_tr,features_te,features_ood in [
        ('quadratic_mean',quadratic_mean(xtr),quadratic_mean(xte),quadratic_mean(xood)),
        ('last_input',xtr[:,-1,:],xte[:,-1,:],xood[:,-1,:]),
        ('mean_input',xtr.mean(axis=1),xte.mean(axis=1),xood.mean(axis=1)),
        ('exp_avg',np.stack([xtr[:,t,:]*(.92**(xtr.shape[1]-1-t)) for t in range(xtr.shape[1])]).sum(axis=0),
         np.stack([xte[:,t,:]*(.92**(xte.shape[1]-1-t)) for t in range(xte.shape[1])]).sum(axis=0),
         np.stack([xood[:,t,:]*(.92**(xood.shape[1]-1-t)) for t in range(xood.shape[1])]).sum(axis=0))]:
        w=LinearHead().fit(features_tr,ytr)
        baselines[name]={'test':w.accuracy(features_te,yte),'ood':w.accuracy(features_ood,yood),'width':features_tr.shape[-1]}
    outcomes=dict(baselines)
    for model,settings in {**MODELS,'dual_path': None}.items():
        net=DualPathMemory(seed=seed+300) if settings is None else TemporalLoop(Config(**settings,seed=seed+300))
        t0=time.perf_counter()
        tr=net.encode(xtr);te=net.encode(xte);oo=net.encode(xood)
        elapsed=time.perf_counter()-t0
        head=LinearHead().fit(tr,ytr)
        result={'test':head.accuracy(te,yte),'ood':head.accuracy(oo,yood),
                'width':tr.shape[-1], 'encode_seconds':elapsed}
        # Causal knockout of all input prior to the second event.
        # Keep trained model/readout frozen. It must lose a required cue.
        absent_first=xte.copy()
        for i,t_second in enumerate(meta_te['second']):
            absent_first[i,:int(t_second)] = 0.0
        result['remove_first_event']=head.accuracy(net.encode(absent_first),yte)
        outcomes[model]=result
    return outcomes

def run(seeds=3,train=1200,test=600,noise=.025):
    if train%4 or test%4: raise ValueError('train/test sizes multiples of 4')
    all_runs=[assess(20261029+127*i,train,test,noise) for i in range(seeds)]
    summary={}
    for name in all_runs[0]:
        summary[name]={k:{'mean':float(np.mean([s[name][k] for s in all_runs])),
                           'sd':float(np.std([s[name][k] for s in all_runs],ddof=1)) if seeds>1 else 0.}
                          for k in ('test','ood')}
    return {'config':dict(seeds=seeds,train=train,test=test,noise=noise),
            'runs':all_runs,'summary':summary}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--seeds',type=int,default=3)
    ap.add_argument('--train',type=int,default=1200)
    ap.add_argument('--test',type=int,default=600)
    ap.add_argument('--noise',type=float,default=.025)
    ap.add_argument('--output',default='results.json')
    a=ap.parse_args()
    result=run(a.seeds,a.train,a.test,a.noise)
    print('Task: was first cue same sign as second cue? Two vanished events, balanced XOR labels.')
    print('OOD test: later perturbation, 2.5x noise; no OOD tuning.')
    print('model                  heldout   delayed+noise')
    for k,v in result['summary'].items():
        print(f'{k:<23} {v["test"]["mean"]:7.1%}   {v["ood"]["mean"]:7.1%}')
    Path(a.output).write_text(json.dumps(result,indent=2),encoding='utf8')
    print('Saved',a.output)

if __name__=='__main__':main()
