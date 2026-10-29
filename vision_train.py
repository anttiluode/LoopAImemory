"""Train and evaluate small temporal AI using session-disjoint gesture clips.

Only the output linear head is trained, with the same fixed vision projection.
Use python vision_train.py --simulate to generate controlled motion examples.
Use python vision_train.py --data my_gestures.npz --test-session day2
"""
import argparse
from pathlib import Path
import numpy as np
from temporal_ai import TemporalLoop, DualPathMemory, Config, LinearHead
from vision_core import synth_dataset, save_dataset, load_dataset


def features(model, x, seed=99):
    if model == 'last_change': return x[:, -1, :]
    if model == 'last_three': return x[:, -3:, :].reshape((len(x), -1))
    if model == 'mean': return x.mean(axis=1)
    if model == 'ema':
        weights = .86**np.arange(x.shape[1]-1,-1,-1)
        return np.einsum('ntd,t->nd',x,weights)/weights.sum()
    if model == 'signed_linear':
        return TemporalLoop(Config(seed=seed,gate=0.,activation='linear')).encode(x)
    if model == 'identity_gate':
        return TemporalLoop(Config(seed=seed,geometry='identity',gate=1.2)).encode(x)
    if model == 'signed_gate':
        return TemporalLoop(Config(seed=seed,gate=1.2)).encode(x)
    if model == 'dual': return DualPathMemory(seed=seed).encode(x)
    if model == 'explicit_history': return x.reshape((len(x), -1))
    raise ValueError(model)


MODELS = ['last_change','last_three','mean','ema','signed_linear',
          'signed_gate','identity_gate','dual','explicit_history']


def evaluate(x, y, session, test_session=None, seed=99, ridge=.1):
    session = np.asarray(session).astype(str)
    sessions = list(np.unique(session))
    if len(sessions) < 2:
        raise ValueError('Need at least TWO different recording sessions for disjoint testing')
    held = str(test_session) if test_session is not None else str(sessions[-1])
    test = session == held
    train = ~test
    if not test.any() or not train.any(): raise ValueError('Invalid held-out session')
    for mask,name in [(train,'train'),(test,'test')]:
        if set(y[mask]) != {-1,1}:
            raise ValueError(f'{name} has only one gesture class: collect both directions')
    results, trained = {}, {}
    for model in MODELS:
        z = features(model,x,seed)
        head = LinearHead(ridge=ridge).fit(z[train], y[train])
        results[model] = {'train':head.accuracy(z[train],y[train]),
                          'test':head.accuracy(z[test],y[test]),
                          'features':z.shape[-1]}
        trained[model] = head.weights
    return results,trained,held,int(train.sum()),int(test.sum())


def train_and_save(path, model_path, test_session=None, seed=99, ridge=.1, model_name="signed_linear"):
    x,y,s=load_dataset(path)
    results,heads,held,ntr,nte=evaluate(x,y,s,test_session,seed,ridge)
    print(f'TRAIN {ntr} clips | HELD-OUT SESSION {held}: {nte} clips')
    print('model                 train   heldout    state/features')
    for k,v in results.items():
        print(f'{k:<23} {v["train"]:6.1%}  {v["test"]:6.1%}      {v["features"]}')
    pathout=Path(model_path)
    pathout.parent.mkdir(parents=True,exist_ok=True)
    # Training data intentionally NOT included in the model artifact.
    if model_name not in heads: raise ValueError("Unknown model " + model_name)
    if model_name not in ("signed_linear", "signed_gate", "identity_gate", "dual"):
        raise ValueError("Only recurrent memory models can be used for webcam inference")
    np.savez_compressed(pathout, weight=heads[model_name], seed=np.array(seed),
                        frames=np.array(x.shape[1]), model=np.array(model_name),
                        encoder=np.array('change_v1'))
    print('Saved',model_name,'model',pathout,'(trained only on non-held-out sessions)')
    return results


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--simulate',action='store_true',help='create four synthetic sessions')
    p.add_argument('--clips-per-session',type=int,default=160)
    p.add_argument('--data',default='gestures.npz')
    p.add_argument('--test-session',default=None)
    p.add_argument('--model',default='models/gesture_loop.npz')
    p.add_argument('--model-name',choices=['signed_linear','signed_gate','identity_gate','dual'],default='signed_linear')
    args=p.parse_args()
    if args.simulate:
        x,y,s=synth_dataset(args.clips_per_session)
        save_dataset(args.data,x,y,s)
        print('Generated synthetic, balanced gestures:',x.shape,'sessions:',np.unique(s))
    train_and_save(args.data,args.model,args.test_session,model_name=args.model_name)


if __name__=='__main__':main()
