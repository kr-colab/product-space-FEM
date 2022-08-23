import numpy as np
import product_fem as pf
from product_fem.equations import HittingTimes
from fenics import UnitIntervalMesh, UnitSquareMesh, FunctionSpace
import pytest


class TestGradient:
    
    def test_1d_gradient(self):
        # function spaces
        n = 25
        mesh = UnitIntervalMesh(n-1)
        V = FunctionSpace(mesh, 'CG', 1)
        W = pf.ProductFunctionSpace(V)
        
        # pde constraint
        u_bdy = 0.5
        epsilon = 1e-2
        eqn = HittingTimes(W, u_bdy, epsilon)
        m = eqn.control
        
        # loss functional
        data = eqn.solve()
        m.update(np.ones(m.dim()))
        reg = {'l2': [1e-6, 1e-4], 'smoothing': [1e-1, 1e-1]}
        loss = pf.LossFunctional(data, m, reg)
        
        # inverse problem grads
        Jhat = pf.ReducedLossFunctional(eqn, loss)
        invp = pf.InverseProblem(eqn, loss)
        
        Jm = invp.compute_loss(m)
        dJ = invp.compute_gradient(m)
        e = np.eye(m.dim())
        
        # taylor remainder |Jp - Jm - h * dJ| = O(h^2)
        def taylor_remainder(h):
            Jp = []
            for i in range(m.dim()):
                m.update(np.ones(m.dim()) + h * e[i])
#                 if i==0:
#                     m.update(m.array() + h * e[i])
#                 else:
#                     m.update(m.array() - h * e[i-1] + h * e[i])
                Jp.append(invp.compute_loss(m))
                          
            m.update(np.ones(m.dim()))
            
            return [abs(Jp[i] - Jm - h * dJ[i]) for i in range(m.dim())]
        
        c = 10
        hs = [1. / (c**k) for k in range(1, 4)]
        rs = [taylor_remainder(h) for h in hs]
        
        # rates should all be close to 2
        rates = [np.log(np.divide(rs[i-1], rs[i])) / np.log(c) for i in range(1, len(rs))]
        twos = 2 * np.ones(m.dim())
        
        # mean square errors
        mse0 = np.mean((rates[0] - twos)**2)
        mse1 = np.mean((rates[1] - twos)**2)
        
#         assert mse1 < mse0
        assert mse0 < 1e-3
        assert mse1 < 2e-4
        
        # relative percent abs errors
        rpe0 = 50 * np.mean(np.abs(rates[0] - twos))
        rpe1 = 50 * np.mean(np.abs(rates[1] - twos))
        
#         assert rpe1 < rpe0
        assert rpe0 < 0.1
        assert rpe1 < 0.1
        
        
    def test_2d_gradient(self):
        # function spaces
        n = 8
        mesh = UnitSquareMesh(n-1, n-1)
        V = FunctionSpace(mesh, 'CG', 1)
        W = pf.ProductFunctionSpace(V)
        
        # pde constraint
        u_bdy = 0.5
        epsilon = 1e-2
        eqn = HittingTimes(W, u_bdy, epsilon)
        m = eqn.control
        
        # loss functional
        data = eqn.solve()
        m.update(np.ones(m.dim()))
        reg = {'l2': [1e-6, 1e-4], 'smoothing': [1e-1, 1e-1]}
        loss = pf.LossFunctional(data, m, reg)
        
        # inverse problem grads
        invp = pf.InverseProblem(eqn, loss)        
        Jm = invp.compute_loss(m)
        dJ = invp.compute_gradient(m)
        e = np.eye(m.dim())
            
        # taylor remainder |J - Jp - h * dJ| = O(h^2)
        def taylor_remainder(h):
            Jp = []
            for i in range(m.dim()):
                if i==0:
                    m.update(m.array() + h * e[i])
                else:
                    m.update(m.array() - h * e[i-1] + h * e[i])

                Jp.append(invp.compute_loss(m))
            
            m.update(m.array() - h * e[i])
            return [abs(Jp[i] - Jm - h * dJ[i]) for i in range(m.dim())]
        
        c = 10
        hs = [1. / (c**k) for k in range(1, 4)]
        rs = [taylor_remainder(h) for h in hs]
        
        # rates should all be close to 2
        rates = [np.log(np.divide(rs[i-1], rs[i])) / np.log(c) for i in range(1, len(rs))]
        twos = 2 * np.ones(m.dim())
        
#         # relative percent abs errors
#         rpe0 = 50 * np.mean(np.abs(rates[0] - twos))
#         rpe1 = 50 * np.mean(np.abs(rates[1] - twos))
        
#         assert rpe1 < rpe0
#         assert rpe0 < 0.1
#         assert rpe1 < 0.05
        
        # mean square errors
        mse0 = np.mean((rates[0] - twos)**2)
        mse1 = np.mean((rates[1] - twos)**2)
        
        assert mse1 < mse0
        assert mse0 < 1e-3
        assert mse1 < 1e-4