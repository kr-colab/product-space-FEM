import numpy as np
import product_fem as pf
from product_fem.equations import HittingTimes
from fenics import UnitIntervalMesh, UnitSquareMesh, FunctionSpace
import pytest
from petsc4py import PETSc


class TestHittingTimes:

    def setup_1d_eqn(self, n=22, u_d=None):
        # set up function space W
        mesh = UnitIntervalMesh(n-1)
        V = FunctionSpace(mesh, 'CG', 1)
        W = pf.ProductFunctionSpace(V)
        return HittingTimes(W, u_bdy=0.35, epsilon=0.05)

    def setup_2d_eqn(self, n=11):
        mesh = UnitSquareMesh(n-1, n-1)
        V = FunctionSpace(mesh, 'CG', 1)
        W = pf.ProductFunctionSpace(V)
        return HittingTimes(W, u_bdy=0.35, epsilon=0.05)

    @pytest.mark.parametrize("n", [23, 32])
    def test_1d_solution(self, n):
        # generate ground truth data u_d
        eqn = self.setup_1d_eqn(n)
        V = eqn.lhs.function_space()
        mu = pf.to_Function('-2 / (1 + exp(-50 * (x[0] - 0.5))) + 1', V)
        sig = pf.to_Function('0.25', V)
        m = pf.Control([mu, sig])
        u = eqn.solve(m).array
        A = eqn.A
        b = eqn.b
        if isinstance(A, PETSc.Mat):
            A = pf.transforms.PETSc_to_sparse(A)
            b = b[:]
        assert np.allclose(A.dot(u), b, atol=1e-6)

    @pytest.mark.parametrize("n", [23, 32])
    def test_1d_control_update(self, n):
        eqn = self.setup_1d_eqn(n)
        u = eqn.solve().array

        # now update control
        mu, sig = m = eqn.control
        mu.vector()[:] = np.linspace(0, 1, mu.dim())
        u_new = eqn.solve(m).array

        assert not np.allclose(u, u_new)

    def test_2d_control_update(self):
        eqn = self.setup_2d_eqn(11)
        u = eqn.solve().array

        # now update control
        mu, sig = m = eqn.control
        mu.vector()[:] = np.linspace(0, 2, mu.dim())
        u_new = eqn.solve(m).array

        assert not np.allclose(u, u_new)

#     @pytest.mark.parametrize("n", [8, 11])
#     def test_2d_solution(self, n):
#         # generate ground truth
#         eqn = self.setup_2d_eqn(n)
#         V = eqn.lhs.function_space()

#         # initialize mu
#         mu = pf.Function(V, dim=2, name='mu')
#         mu_arr = np.arange(0, 2, mu.dim())
#         mu.vector()[:] = mu_arr

#         # initialize sigma
#         sig = pf.Function(V, dim=3, name='sig')
#         sig_arr = np.ones(sig.dim())
#         sig_arr[2::3] = 0.
#         sig.vector()[:] = sig_arr

#         m = pf.Control([mu, sig])
#         u = eqn.solve(m).array

#         assert np.allclose(eqn.A.dot(u), eqn.b)

#     # NOTE: this grad test will fail when n is odd (since x=0.5 will be a mesh node)
#     @pytest.mark.parametrize("n", [22, 32])
#     def test_gradient(self, n):
#         alpha, beta = (1e-6, 1e-8), 0. # regularization parameters
#         eqn = self.setup_eqn(n)

#         # diffusion coef, drift coef, data
#         sig = 2.5e-1
#         mu_true = eqn.mu_str_to_array('-2 / (1 + exp(-50 * (x[0] - 0.5))) + 1')
#         u_d = eqn.solve(sig, mu_true)

#         # new instance with data u_d
#         del eqn
#         eqn = self.setup_eqn(n, u_d)

#         diff_quots = np.zeros_like(mu_true) # forward differences
#         mu0 = np.zeros_like(mu_true) # initial mu
#         h, eye = 0.001, np.eye(len(mu0))
#         for i in range(len(eye)):
#             mu_pert = mu0 + h * eye[i]
#             J = eqn.loss_functional(sig, mu0, alpha, beta)
#             Jpert = eqn.loss_functional(sig, mu_pert, alpha, beta)
#             diff_quots[i] = Jpert / h - J / h

#         J_grads = eqn.compute_gradient(sig, mu0, alpha, beta)[0]
#         rel_percent_err = 100 * np.abs((diff_quots - J_grads) / diff_quots)

#         # require no more than 3% error
#         assert np.max(rel_percent_err) < 3.0
