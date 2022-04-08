import numpy as np
from product_fem import product_fem as pf
from product_fem.equations import HittingTimes
from fenics import *
import pytest

class TestHittingTimes:
    
    def setup_eqn(self, n=22, u_d=None):
        # set up function space W
        mesh = UnitIntervalMesh(n-1)
        V = FunctionSpace(mesh, 'CG', 1)
        W = pf.ProductFunctionSpace(V)
        
        # forcing function
        f = (['-1.0'], ['1.0']) 
        
        # constant boundary conditions
        def on_product_boundary(x, y):
            eps = 0.05
            return np.abs(x - y) <= eps
        bc = pf.ProductDirichletBC(W, 0.35, on_product_boundary)
        
        if u_d is not None:
            return HittingTimes(W, f, bc, u_d)
        else:
            return HittingTimes(W, f, bc)
        
    @pytest.mark.parametrize("n", [23, 32])
    def test_solution(self, n):
        # generate ground truth data u_d
        eqn = self.setup_eqn(n)
        mu_true = eqn.mu_str_to_array('-2 / (1 + exp(-50 * (x[0] - 0.5))) + 1')
        sig = 2.5e-1 # diffusion coefficient
        u = eqn.solve(sig, mu_true)

        # compare Au to f to verify solution
        assert np.allclose(eqn.stiffness.dot(u), eqn.rhs)   
        
    # NOTE: this grad test will fail when n is odd (since x=0.5 will be a mesh node)
    @pytest.mark.parametrize("n", [22, 32])
    def test_gradient(self, n):
        alpha, beta = (1e-6, 1e-8), 0. # regularization parameters
        eqn = self.setup_eqn(n)
        
        # diffusion coef, drift coef, data
        sig = 2.5e-1 
        mu_true = eqn.mu_str_to_array('-2 / (1 + exp(-50 * (x[0] - 0.5))) + 1')
        u_d = eqn.solve(sig, mu_true)
        
        # new instance with data u_d
        del eqn
        eqn = self.setup_eqn(n, u_d)
        
        diff_quots = np.zeros_like(mu_true) # forward differences
        mu0 = np.zeros_like(mu_true) # initial mu
        h, eye = 0.001, np.eye(len(mu0))
        for i in range(len(eye)):
            mu_pert = mu0 + h * eye[i]
            J = eqn.loss_functional(sig, mu0, alpha, beta)
            Jpert = eqn.loss_functional(sig, mu_pert, alpha, beta)
            diff_quots[i] = Jpert / h - J / h

        J_grads = eqn.compute_gradient(sig, mu0, alpha, beta)[0]
        rel_percent_err = 100 * np.abs((diff_quots - J_grads) / diff_quots)
        
        # require no more than 3% error
        assert np.max(rel_percent_err) < 3.0 