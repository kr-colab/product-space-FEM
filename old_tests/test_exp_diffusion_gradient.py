import numpy as np
from product_fem import product_fem as pf
from product_fem.equations import ExpDiffusion
from fenics import *
import pytest


class TestExpDiffusion:
    
    def compute_analytic_grads(self, n, alpha):
        """computes int _m e^m nabla(u) dot nabla(p) dxdy
        + alpha int nabla(m) dot nabla(_m) dxdy
        where _m is a placeholder for each basis element"""

        # given m, solve F(u, m) = 0 for u
        mesh = UnitSquareMesh(n-1, n-1)
        V = FunctionSpace(mesh, 'CG', 1)
        v2d = vertex_to_dof_map(V)
        bc = DirichletBC(V, 0, 'on_boundary')

        m = interpolate(Expression('cos(x[0]) + sin(x[1])', element=V.ufl_element()), V)
        f = interpolate(Expression('exp(-x[0]) * cos(x[1])', element=V.ufl_element()), V)
        u, v = Function(V), TestFunction(V)

        F = exp(m) * inner(grad(u), grad(v)) * dx - f * v * dx
        solve(F==0, u, bc)

        # given m and u, solve adjoint eq for p
        p, v = Function(V), TestFunction(V)
        u_d = Function(V) # let data=0
        F_adj = exp(m) * inner(grad(p), grad(v)) * dx + (u - u_d)**2 * v * dx
        solve(F_adj==0, p, bc)

        # given m, u and p, compute analytic gradient
        grads = np.zeros(V.dim())
        for i in range(V.dim()):
            _m = Function(V)
            _m.vector()[i] = 1.
            grad_i = _m * exp(m) * inner(grad(u), grad(p)) * dx + alpha * inner(grad(m), grad(_m)) * dx
            grads[i] = assemble(grad_i)
        return grads[v2d]

    def verify_gradient(self, product_grads, n):
        analytic_grads = self.compute_analytic_grads(n, alpha=1)
        grad_error = np.linalg.norm(product_grads - analytic_grads)
        assert np.log(grad_error) < -10
    
    @pytest.mark.parametrize("n", [11, 17])
    def test_exp_diffusion_grad(self, n):
        mesh = UnitIntervalMesh(n-1)
        V = FunctionSpace(mesh, 'CG', 1)
        W = pf.ProductFunctionSpace(V)
        bc = pf.ProductDirichletBC(W, 0, 'on_boundary')

        eqn = ExpDiffusion(W, ['exp(-x[0])', 'cos(x[0])'], bc)
        grads = eqn.compute_gradient(['cos(x[0])', 'sin(x[0])'], alpha=1)
        p2f = [-i-j*n for i in range(1,n+1) for j in range(n)]
        grads = grads.flatten()[p2f]
        self.verify_gradient(grads, n)
        