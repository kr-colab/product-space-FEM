import numpy as np
from product_fem import product_fem as pf
from product_fem.equations import DriftDiffusion
from fenics import *
import pytest

class TestDriftDiffusion:
    
    def setup_W(self, n=22):
        # set up function space
        mesh = UnitIntervalMesh(n-1)
        h = mesh.hmax()
        V = FunctionSpace(mesh, 'CG', 1)
        W = pf.ProductFunctionSpace(V)
        return W, V
    
    def verify_solution(self, W, eps, b, f, bc, soln):
        # fem solution
        dd = DriftDiffusion(W, f, bc)
        u_h = dd.solve(eps, b)

        # analytic solution
        u = pf.to_array(soln, W)
        
        L2_error = np.sqrt(W.integrate((u - u_h)**2))
        assert np.log(L2_error) < -10.
        
    def test_gradient(self):
        n = 22
        W, V = self.setup_W(n)
        f = (['-1.0'], ['1.0']) # forcing function
        bc = pf.ProductDirichletBC(W, 0, 'on_boundary')
        eqn = DriftDiffusion(W, f, bc)
        
        # regularization parameters
        alpha, beta = 1.0e-08, 0.
        
        # diffusion and drift coefficients
        eps = -9.0e-02 
        b_true = eqn.b_str_to_array(['-2*sin(2*pi*x[0])', '-2*cos(2*pi*x[0])+2'])
        
        # data
        u_d = eqn.solve(eps, b_true)
        del eqn
        eqn = DriftDiffusion(W, f, bc, u_d)

        h = 0.001
        b_dim = 2 * V.dim()
        eye = np.eye(b_dim)
        diff_quots = np.zeros(b_dim)
        b0 = np.zeros(b_dim)
        for i in range(b_dim):
            bpert = b0 + h * eye[i]
            J = eqn.loss_functional(eps, b0, alpha, beta)
            Jpert = eqn.loss_functional(eps, bpert, alpha, beta)
            diff_quots[i] = Jpert / h - J / h

        J_grads = eqn.compute_gradient(eps, b0, alpha, beta)[0]
        rel_percent_err = 100 * np.abs((diff_quots - J_grads) / diff_quots)
        b1_err, b2_err = np.split(rel_percent_err, [n])
        
        # require no more than 3% error
        assert max(np.max(b1_err), np.max(b2_err)) < 3.0 
        
    @pytest.mark.parametrize("n", [27, 32])
    def test_sinusoid(self, n):
        W, V = self.setup_W(n)
        
        # diffusion and drift coefs
        eps = 1
        b = ('x[0]', 'x[0]')

        # force function f = sum_i X_iY_i
        X = ['2 * eps * sin(x[0])', 'x[0] * cos(x[0])', 'sin(x[0])']
        X = [Expression(x, element=V.ufl_element(), eps=eps) for x in X]
        Y = ['sin(x[0])', 'sin(x[0])', 'x[0] * cos(x[0])']
        Y = [Expression(y, element=V.ufl_element()) for y in Y]
        f = (X, Y)

        # boundary conditions
        u_bdy = lambda x, y: np.sin(x) * np.sin(y)
        bc = pf.ProductDirichletBC(W, u_bdy, 'on_boundary')

        self.verify_solution(W, eps, b, f, bc, u_bdy)
        
    @pytest.mark.parametrize("n", [27, 32])
    def test_polynomial(self, n):
        W, V = self.setup_W(n)
        
        eps = 1
        b = ('x[0]', 'x[0]')
        
        # force function f = sum_i X_iY_i
        X = ['x[0] * x[0]', 'x[0]', '1', '-2 * eps']
        X = [Expression(x, element=V.ufl_element(), eps=eps) for x in X]

        Y = ['1', 'x[0]', 'x[0] * x[0]', '1']
        Y = [Expression(y, element=V.ufl_element()) for y in Y]
        f = (X, Y)
        
        # boundary condition is analytic solution
        u_bdy = lambda x, y: 0.5 * (x**2 + x*y + y**2)
        bc = pf.ProductDirichletBC(W, u_bdy, 'on_boundary')
        
        self.verify_solution(W, eps, b, f, bc, u_bdy)
        
    @pytest.mark.parametrize("n", [27, 32])
    def test_exponential(self, n):
        W, V = self.setup_W(n)
        
        # parameters
        eps = 1
        b = ('x[0]', 'x[0]')
        b1 = Expression(b[0], element=V.ufl_element())
        b2 = Expression(b[1], element=V.ufl_element())
        
        # force function f = sum_i X_iY_i
        X = ['exp(x[0]) * 2 * eps * x[0] * x[0]', 
             'exp(x[0]) * 2 * eps * x[0]', 
             'exp(x[0]) * 4 * eps * x[0]', 
             '-exp(x[0]) * b1 * (x[0] * x[0] + x[0] - 1)', 
             'exp(x[0]) * x[0] * (x[0] - 1)']
        X = [Expression(x, element=V.ufl_element(), eps=eps, b1=b1) for x in X]

        Y = ['exp(-x[0]) * (x[0] - 1) * (x[0] - 2)', 
             'exp(-x[0]) * x[0] * (x[0] - 1)', 
             'exp(-x[0]) * (x[0] - 1)', 
             'exp(-x[0]) * x[0] * (x[0] - 1)', 
             'exp(-x[0]) * b2 * (x[0] * x[0] - 3 * x[0] + 1)']
        Y = [Expression(y, element=V.ufl_element(), b2=b2) for y in Y]
        f = (X, Y)
        
        # boundary condition is analytic solution
        u_bdy = lambda x, y: -x * (1-x) * y * (1-y) * np.exp(x-y)
        bc = pf.ProductDirichletBC(W, 0, 'on_boundary')
        
        self.verify_solution(W, eps, b, f, bc, u_bdy)
