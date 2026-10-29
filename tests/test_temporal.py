import unittest
import numpy as np
from temporal_ai import Config,TemporalLoop,DualPathMemory,LinearHead,shift_matrix
from tasks import generate,lag_stream,counterbalanced_events


class TemporalTests(unittest.TestCase):
    def test_signed_and_cyclic_return(self):
        rng=np.random.default_rng(1)
        signed=shift_matrix(16,'signed_shift',rng)
        cyclic=shift_matrix(16,'cyclic',rng)
        self.assertTrue(np.allclose(np.linalg.matrix_power(signed,16),-np.eye(16)))
        self.assertTrue(np.allclose(np.linalg.matrix_power(signed,32),np.eye(16)))
        self.assertTrue(np.allclose(np.linalg.matrix_power(cyclic,16),np.eye(16)))

    def test_zero_step_matches_core_formula(self):
        net=TemporalLoop(Config())
        s=np.linspace(-.3,.2,16);u=np.linspace(-.1,.5,16)
        h=net.R@s;b=net.B@u
        expected=np.tanh(.92*h+b+1.2*b*np.tanh(net.A@h))
        self.assertTrue(np.allclose(net.step(s,u),expected))

    def test_dual_step_matches_batch(self):
        net=DualPathMemory()
        rng=np.random.default_rng(7)
        inputs=rng.normal(size=(12,16))*.03
        s=np.zeros(32)
        for u in inputs:s=net.step(s,u)
        self.assertTrue(np.allclose(s,net.encode(inputs[None])[0]))

    def test_labels_balanced_and_last_image_uninformative(self):
        a,y,meta=generate(400,seed=9)
        self.assertEqual(int((y==1).sum()),200)
        self.assertEqual(int((y==-1).sum()),200)
        self.assertTrue(np.all(meta['first']<meta['second']))
        self.assertTrue(np.all(meta['second']<=21))
        self.assertTrue(np.all(meta['second']>=18))
        self.assertEqual(a.shape,(400,24,16))

    def test_linear_readout_cannot_solve_pure_xor(self):
        x=np.array([[-1,-1],[-1,1],[1,-1],[1,1]],float)
        y=x[:,0]*x[:,1]
        self.assertLessEqual(LinearHead().fit(x,y).accuracy(x,y),.75)

    def test_identity_gated_learns_history_nonlinearity(self):
        x,y,_=generate(600,seed=1)
        z,v,_=generate(200,seed=2)
        net=TemporalLoop(Config(geometry='identity',gate=1.2))
        acc=LinearHead().fit(net.encode(x),y).accuracy(net.encode(z),v)
        self.assertGreater(acc,.95)

    def test_causal_first_cue_removal_hurts(self):
        x,y,_=generate(800,seed=4)
        z,v,meta_z=generate(200,seed=5)
        net=DualPathMemory(seed=19)
        head=LinearHead().fit(net.encode(x),y)
        normal=head.accuracy(net.encode(z),v)
        z_no=z.copy()
        for i,t in enumerate(meta_z['second']):z_no[i,:int(t)]=0
        erased=head.accuracy(net.encode(z_no),v)
        self.assertGreater(normal,.90)
        self.assertLess(erased,.64)

    def test_lag_data_is_independent_of_last_bit(self):
        x,y=lag_stream(400,seed=50)
        self.assertEqual(x.shape,(400,12,16))
        self.assertLess(abs(np.mean(y)),.2)

    def test_order_pairs_have_same_total_composition(self):
        x,y=counterbalanced_events(50,seed=3)
        self.assertTrue(np.allclose(x[0].sum(axis=0),x[1].sum(axis=0)))
        self.assertEqual(list(y[:2]),[1,-1])


if __name__=='__main__':unittest.main()
