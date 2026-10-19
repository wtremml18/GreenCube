"""Numerical physics and data-split tests for the real-data extension."""
import unittest
import numpy as np
from model import covariance, PotentialGP, contiguous_splits
from data import plane, image_ridge, electric_output_scale, decode_rocket_metadata, DATA


class ModelTests(unittest.TestCase):
    def setUp(self):
        self.rng=np.random.default_rng(431)
        self.x=self.rng.uniform(-80,80,(14,2))
        self.p=[15.,-.2,.002]

    def test_curved_covariance_psd_and_reciprocity(self):
        for flow in [False,True]:
            k=covariance(self.x,self.x,30,4,self.p,flow=flow)
            np.testing.assert_allclose(k,k.T,atol=1e-13)
            self.assertGreater(np.linalg.eigvalsh(k).min(),-1e-12)

    def test_physical_derivative_constraint(self):
        query=self.rng.uniform(-60,60,(12,2));alpha=self.rng.normal(size=28)
        h=1e-3
        def derivative(axis,flow):
            delta=np.zeros_like(query);delta[:,axis]=h
            a=covariance(query+delta,self.x,30,4,self.p,flow=flow)@alpha
            b=covariance(query-delta,self.x,30,4,self.p,flow=flow)@alpha
            return ((a-b)/(2*h)).reshape(-1,2)
        curl=derivative(0,False)[:,1]-derivative(1,False)[:,0]
        div=derivative(0,True)[:,0]+derivative(1,True)[:,1]
        self.assertLess(np.max(abs(curl)),1e-9)
        self.assertLess(np.max(abs(div)),1e-9)

    def test_permutation_and_conditional_covariance(self):
        y=self.rng.normal(size=(14,2))*20
        noise=np.tile(np.array([[9.,1.],[1.,16.]]),(14,1,1));perm=self.rng.permutation(14)
        a=PotentialGP(self.x,y,noise,30,30,2,4,self.p)
        b=PotentialGP(self.x[perm],y[perm],noise[perm],30,30,2,4,self.p)
        q=self.rng.uniform(-120,120,(20,2))
        ya,ca=a.predict(q,True);yb,cb=b.predict(q,True)
        np.testing.assert_allclose(ya,yb,atol=1e-9)
        np.testing.assert_allclose(ca,cb,atol=1e-8)
        self.assertGreaterEqual(np.linalg.eigvalsh(ca).min(),-1e-8)

    def test_exact_constant_background(self):
        y=np.tile([30.,-17.],(14,1));noise=np.tile(np.eye(2),(14,1,1))
        model=PotentialGP(self.x,y,noise,30,30,2,4,self.p)
        q=np.array([[400.,300.],[-500.,-200.],[0.,0.]])
        np.testing.assert_allclose(model.predict(q),np.tile(y[0],(3,1)),atol=1e-10)

    def test_guarded_nested_splits_with_missing_rows(self):
        ids=np.array([1,2,3,4,5,6,7,8,9,10,11,13,14,15,17,18])
        tested=[]
        for train,test in contiguous_splits(ids,4,1):
            self.assertGreater(np.min(abs(ids[train,None]-ids[test][None,:])),1)
            tested.extend(test)
            for inner_train,inner_test in contiguous_splits(ids[train],3,1):
                self.assertFalse(set(ids[train[inner_train]]) & set(ids[test]))
                self.assertGreater(np.min(abs(ids[train[inner_train],None]-ids[train[inner_test]][None,:])),1)
        self.assertEqual(sorted(tested),list(range(len(ids))))

    def test_image_ridge_known_geometry(self):
        xx,yy=np.meshgrid(np.linspace(-150,150,120),np.linspace(-100,180,180))
        true=20.-.15*xx+.001*xx**2
        im=5+100*np.exp(-.5*((yy-true)/8)**2)
        p,_,error=image_ridge(np.stack([xx,yy],axis=-1),im,(-145,145),(-100,180))
        np.testing.assert_allclose(np.polynomial.Polynomial(p)(np.linspace(-120,120,20)),np.polynomial.Polynomial([20.,-.15,.001])(np.linspace(-120,120,20)),atol=1.)
        self.assertLess(error,1.)

    def test_spherical_electric_metric(self):
        # A potential proportional to longitude has physical east field 1/cos(lat).
        q=plane(np.array([-147.,-147.]),np.array([64.,69.]))
        physical=np.column_stack([1/np.cos(np.deg2rad([64.,69.])),[0.,0.]])
        chart=physical/electric_output_scale(q)
        np.testing.assert_allclose(chart[:,0],np.repeat(1/np.cos(np.deg2rad(65.12)),2))

    @unittest.skipUnless((DATA/'flow_data_ig.mat').exists(),'Pinned public data not downloaded')
    def test_decoded_rocket_time_against_elapsed(self):
        from scipy.io import loadmat
        f=loadmat(DATA/'flow_data_ig.mat',simplify_cells=True)
        unix,notes=decode_rocket_metadata(f)
        np.testing.assert_allclose(unix-f['time'],1488441000.,atol=.001,rtol=0)
        self.assertEqual(len(unix),195)
        self.assertIn('apex',notes[1])

    def test_longitude_wrap(self):
        np.testing.assert_allclose(plane(np.array([-147.,213.]),np.array([65.,65.]))[0],plane(np.array([-147.,213.]),np.array([65.,65.]))[1])

if __name__=='__main__':unittest.main()
