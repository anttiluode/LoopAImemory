"""Local webcam gesture experiment. Stores only compressed 16D motion features.

  python webcam_gestures.py record --session day1
  python webcam_gestures.py record --session day2
  python vision_train.py --data gestures.npz --test-session day2
  python webcam_gestures.py predict --model models/gesture_loop.npz

The camera is never uploaded; projected feature vectors still deserve privacy.
"""
from pathlib import Path
import argparse
import time
import cv2
import numpy as np
from vision_core import FrameToVector, VisionConfig, save_dataset, load_dataset
from temporal_ai import DualPathMemory, TemporalLoop, Config, LinearHead


def camera(index):
    idx = int(index) if str(index).isdigit() else index
    if isinstance(idx, int):
        import sys
        cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW) if sys.platform == 'win32' else cv2.VideoCapture(idx)
        if not cap.isOpened():
            cap.release()
            cap = cv2.VideoCapture(idx)
    else:
        cap = cv2.VideoCapture(idx)
    if not cap.isOpened():
        raise RuntimeError('Camera unavailable; try --source 1 or close other webcam apps')
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    return cap


def show(cap, title, mirror=False, line2=''):
    ok, frame = cap.read()
    if not ok:
        raise RuntimeError('Camera read failed')
    if mirror:
        frame = cv2.flip(frame,1)
    view = frame.copy()
    cv2.rectangle(view, (0,0), (view.shape[1],90), (20,25,30), -1)
    cv2.putText(view,title,(16,35),cv2.FONT_HERSHEY_SIMPLEX,.75,(230,230,230),2,cv2.LINE_AA)
    cv2.putText(view,line2,(16,72),cv2.FONT_HERSHEY_SIMPLEX,.48,(170,245,150),1,cv2.LINE_AA)
    cv2.imshow('LoopAImemory - local webcam',view)
    key=cv2.waitKey(1) & 0xFF
    return frame,key


def capture(cap, frames, fps, mirror=False):
    """Target cadence; actual intervals vary with camera/CPU. Temporal model
    currently interprets frame index, not true elapsed seconds."""
    encoder=FrameToVector(VisionConfig())
    x=[]
    start=time.monotonic()
    for i in range(frames):
        due=start+i/fps
        if due > time.monotonic():
            time.sleep(due-time.monotonic())
        frame,key=show(cap,f'RECORDING  {i+1}/{frames}',mirror,
                       'Move early in one direction, then leave the scene empty.')
        if key in (27,ord('q')):
            return None
        x.append(encoder.push_frame(frame))
    return np.asarray(x,dtype=np.float32)


def record(args):
    if not args.session:
        raise ValueError('record requires --session NAME; use a new name on a future day')
    if args.frames < 22 or args.fps <= 0:
        raise ValueError('frames >=22 and fps>0 required')
    cap=camera(args.source)
    data_path=Path(args.data)
    if data_path.exists():
        x,y,s=load_dataset(data_path)
        if x.shape[1] != args.frames:
            raise ValueError(f'Existing dataset has {x.shape[1]} frames; use --frames {x.shape[1]}')
        stored = [v for v in (x,y,s)]
    else:
        stored=[]
    collected=[]
    print('L = record LEFT-to-RIGHT | R = record RIGHT-to-LEFT | Q = quit')
    print('Keep mirror setting, camera position and recording cadence consistent.')
    try:
        while True:
            _,key=show(cap,'L/R start labeled clip  |  Q quit',args.mirror,
                       f'Session {args.session} | {len(collected)} new clips')
            if key in (27,ord('q')):break
            if key not in (ord('l'),ord('r')):continue
            label = 1 if key==ord('l') else -1
            description = 'LEFT to RIGHT' if label==1 else 'RIGHT to LEFT'
            start=time.monotonic()
            while time.monotonic()-start < 1.5:
                elapsed=time.monotonic()-start
                _,stop=show(cap,description+' - GET READY',args.mirror,
                            f'Recording in {max(0,1.5-elapsed):.1f}s. Clear background first.')
                if stop in (27,ord('q')):return
            print('Capture',description)
            sample=capture(cap,args.frames,args.fps,args.mirror)
            if sample is None:break
            collected.append((sample,label))
            cx=np.stack([v for v,_ in collected]); cy=np.array([v for _,v in collected],dtype=np.int8)
            cs=np.array([args.session]*len(collected))
            if stored:
                xx=np.concatenate([stored[0],cx]); yy=np.concatenate([stored[1],cy]); ss=np.concatenate([stored[2],cs])
            else: xx,yy,ss=cx,cy,cs
            data_path.parent.mkdir(parents=True,exist_ok=True)
            save_dataset(data_path,xx,yy,ss)
            print(f'Saved {len(yy)} total clips (compressed 16D features only): {data_path}')
    finally:
        cap.release();cv2.destroyAllWindows()
    print('Aim for 15-25 clips per direction per session, then repeat in another session.')


def predict(args):
    with np.load(args.model,allow_pickle=False) as d:
        weights=d['weight'].copy()
        seed=int(d['seed'])
        frames=int(d['frames'])
        model=str(d['model'])
        encoder=str(d['encoder'])
    if encoder!='change_v1':
        raise ValueError('Model encoder incompatible with webcam')
    if model=='dual': net=DualPathMemory(seed=seed)
    elif model=='signed_linear': net=TemporalLoop(Config(seed=seed,gate=0.,activation='linear'))
    elif model=='signed_gate': net=TemporalLoop(Config(seed=seed,gate=1.2))
    elif model=='identity_gate': net=TemporalLoop(Config(seed=seed,geometry='identity',gate=1.2))
    else: raise ValueError('Unknown model architecture: '+model)
    head=LinearHead();head.weights=weights
    cap=camera(args.source)
    print('Press SPACE for an unlabeled clip, then classify from final memory; Q to quit.')
    try:
        while True:
            _,key=show(cap,'SPACE: ask memory about motion | Q: quit',args.mirror,
                       'Remove your hand before the clip ends.')
            if key in (27,ord('q')):break
            if key!=32:continue
            start=time.monotonic()
            while time.monotonic()-start < 1.5:
                show(cap,'GET READY',args.mirror,'Move after the countdown; finish early.')
            x=capture(cap,frames,args.fps,args.mirror)
            if x is None:break
            state=net.encode(x[None,:])[0]
            score=float(head.score(state[None,:])[0])
            print('Temporal decision:', 'LEFT -> RIGHT' if score>=0 else 'RIGHT -> LEFT',
                  f'(uncalibrated linear margin {score:.3f})')
            deadline=time.monotonic()+1.5
            while time.monotonic()<deadline:
                show(cap,'Prediction: '+('LEFT to RIGHT' if score>=0 else 'RIGHT to LEFT'),
                     args.mirror, f'Linear margin {score:.3f} (not a probability)')
    finally:
        cap.release();cv2.destroyAllWindows()


def main():
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    modes=p.add_subparsers(dest='command',required=True)
    for name in ('record','predict'):
        a=modes.add_parser(name)
        a.add_argument('--source',default='0')
        a.add_argument('--fps',type=float,default=8.)
        a.add_argument('--mirror',action='store_true',help='flip horizontally; use same choice in every session')
        a.add_argument('--frames',type=int,default=30)
        if name=='record':
            a.add_argument('--session',required=True)
            a.add_argument('--data',default='gestures.npz')
        else:
            a.add_argument('--model',default='models/gesture_loop.npz')
    args=p.parse_args()
    (record if args.command=='record' else predict)(args)


if __name__=='__main__':main()
