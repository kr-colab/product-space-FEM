import pytest
from numpy import e, exp, log, sin, cos, corrcoef
from product_fem import ProductFunctionSpace, to_Function
from product_fem.loss_functionals import L2Error
from fenics import UnitIntervalMesh, UnitSquareMesh, FunctionSpace, \
     assemble, dx


class TestFunctionals:
    
    def test_1d_l2_error(self):
        """Here we test the L2 error Functional 
        E(u;d) := 1/2 int (u-d)^2 dxdy
        We use the fact that when u(x,y) = cos(x-y) and d(x,y) = sin(x-y)
        the functional evaluates to exactly 1/2
        """
        n = 15
        mesh = UnitIntervalMesh(n-1)
        V = FunctionSpace(mesh, 'CG', 1)
        W = ProductFunctionSpace(V)
        
        d_eval = lambda x,y: sin(x-y)
        d = to_Function(d_eval, W)
        E = L2Error(d)
        
        u_eval = lambda x,y: cos(x-y)
        u = to_Function(u_eval, W)
        Eu = E(u)
        
        assert abs(Eu - 1/2) < 1e-12
        
    def test_1d_l2_error_derivative(self):
        h = 1e-9
        n = 25
        mesh = UnitIntervalMesh(n-1)
        V = FunctionSpace(mesh, 'CG', 1)
        W = ProductFunctionSpace(V)
        
        d = to_Function(lambda x,y: exp(-x-y), W)
        u = to_Function(lambda x,y: exp(x+y), W)
        E = L2Error(d)
        
        def central_diff(i, j):
            phi_ij = W._basis_ij(i, j)
            E_right = E(u + h * phi_ij)
            E_left = E(u - h * phi_ij)
            
            return (E_right - E_left) / (2 * h)
        
        diffs = [central_diff(i,j) for i,j in W.dofs()]
        dE = E.derivative(u)
        
        assert abs(corrcoef(diffs, dE)[0,1] - 1) < 1e-4
        
        
    def test_2d_l2_error(self):
        """When u(x1,x2,y1,y2) = exp(x1+x2+y1+y2) and 
        d(x1,x2,y1,y2) = exp(-x1-x2-y1-y2)
        we can analytically integrate 
        E(u;d) := 1/2 (u - d)^2 over the unit hypercube.
        Since (u-d)^2 = u^2 + d^2 - 2ud and 
        u^2 = exp(2(x1+x2+y1+y2)) we have 
        int u^2 dxdy = ((exp(2) - 1) / 2)^4,
        int d^2 dxdy = ((1 - exp(-2)) / 2)^4,
        int 2ud dxdy = 2
        """
        n = 25
        mesh = UnitSquareMesh(n-1, n-1)
        V = FunctionSpace(mesh, 'CG', 1)
        W = ProductFunctionSpace(V)
        
        def d_eval(x, y):
            return exp(-x[0] - x[1] - y[0] - y[1])
        d = to_Function(d_eval, W)
        E = L2Error(d)
        
        def u_eval(x, y):
            return exp(x[0] + x[1] + y[0] + y[1])
        u = to_Function(u_eval, W)
        Eu = E(u)
        
        Eu_true = (exp(2) - 1)**4 / 32 + (1 - exp(-2))**4 / 32 - 1
        
        assert abs((Eu - Eu_true)/Eu_true) < 5e-3
        
    def test_2d_l2_error_derivative(self):
        pass