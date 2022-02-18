import numpy as np
from product_fem import product_fem as pf
from product_fem.equations import DriftDiffusion
from fenics import *
import pytest

class TestDriftDiffusion:
    
    def setup_W(self, n=21):
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
        u = pf.Function(W)
        u.assign(soln)
        u = u.array
        
        L2_error = np.sqrt(W.integrate((u - u_h)**2))
        assert np.log(L2_error) < -10.
        
    @pytest.mark.parametrize("n", [27, 32])
    def test_sinusoid(self, n):
        W, V = self.setup_W(n)
        
        # drift coefs
        eps = 1
        b1 = Expression('x[0]', element=V.ufl_element())
        b2 = Expression('x[0]', element=V.ufl_element())
        b = (b1, b2)

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
        b1 = Expression('x[0]', element=V.ufl_element())
        b2 = Expression('x[0]', element=V.ufl_element())
        b = (b1, b2)
        
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
        
        eps = 1
        b1 = Expression('x[0]', element=V.ufl_element())
        b2 = Expression('x[0]', element=V.ufl_element())
        b = (b1, b2)
        
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
