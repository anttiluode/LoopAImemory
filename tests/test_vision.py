import unittest
import tempfile
from pathlib import Path
import numpy as np
from vision_core import FrameToVector, synth_dataset, save_dataset, load_dataset
from vision_train import evaluate
from temporal_ai import DualPathMemory


class VisionTests(unittest.TestCase):
    def test_identical_frame_is_zero_input(self):
        e=FrameToVector()
        g=np.zeros((16,16))+0.2
        self.assertTrue(np.allclose(e.push_gray(g),0))
        moved=g.copy();moved[3:7,3:7]+=.5
        self.assertGreater(np.linalg.norm(e.push_gray(moved)),0)
        self.assertTrue(np.allclose(e.push_gray(moved),0))

    def test_synthetic_sessions_balanced_and_final_silent(self):
        x,y,s=synth_dataset(n_per_session=12,sessions=3,seed=12)
        self.assertEqual(x.shape,(36,30,16))
        for session in np.unique(s):
            self.assertEqual(int(sum(y[s==session]==1)),6)
        self.assertLess(np.linalg.norm(x[:,-1],axis=1).mean(),.15)

    def test_session_guard(self):
        x,y,s=synth_dataset(n_per_session=6,sessions=1)
        with self.assertRaisesRegex(ValueError,'TWO'):
            evaluate(x,y,s)

    def test_dual_matches_step_by_step(self):
        x,y,s=synth_dataset(n_per_session=4,sessions=2)
        model=DualPathMemory(seed=99)
        b=model.encode(x[:1])[0]
        m=np.zeros(32)
        for vec in x[0]:m=model.step(m,vec)
        self.assertTrue(np.allclose(m,b,atol=1e-12))

    def test_session_split_and_synthetic_temporal_learning(self):
        x,y,s=synth_dataset(n_per_session=36,sessions=3,seed=8)
        results,heads,held,ntr,nte=evaluate(x,y,s)
        self.assertEqual((ntr,nte),(72,36))
        self.assertGreaterEqual(results['dual']['test'],.85)
        self.assertGreaterEqual(results['signed_linear']['test'],.85)
        self.assertLess(results['last_change']['test'],.8)
        self.assertEqual(len(heads['dual']),33)

    def test_save_and_load_without_pickle(self):
        x,y,s=synth_dataset(n_per_session=4,sessions=2)
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'test.npz'
            save_dataset(path,x,y,s)
            x2,y2,s2=load_dataset(path)
            self.assertTrue(np.allclose(x,x2,atol=1e-6))
            self.assertTrue(np.array_equal(y,y2))
            self.assertTrue(np.array_equal(s2,s.astype(str)))


if __name__=='__main__':unittest.main()
