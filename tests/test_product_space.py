import pytest
from numpy import e, exp, log, sin, cos
from product_fem import ProductFunctionSpace, to_Function
from fenics import UnitIntervalMesh, UnitSquareMesh, FunctionSpace, \
assemble, dx
from scipy.integrate import dblquad, nquad


class TestProductSpaceIntegrate:
    
    def test_1d_analytic_integral(self):
        """
        """
        n = 15
        mesh = UnitIntervalMesh(n-1)
        V = FunctionSpace(mesh, 'CG', 1)
        W = ProductFunctionSpace(V)
        
        def fx_eval(x):
            return cos(x) * exp(x)
        fx_int = (e * (sin(1) + cos(1)) - 1) / 2
        
        def fy_eval(y):
            return sin(y) * exp(y)
        fy_int = (1 + e * sin(1) - e * cos(1)) / 2
        
        def f_eval(x,y):
            return fx_eval(x) * fy_eval(y)
        f_int = (-1 + 2 * e * cos(1) - e**2 * cos(2)) / 4
        
        fx = to_Function(fx_eval, V)
        fy = to_Function(fy_eval, V)
        f = to_Function(f_eval, W)
        
        value = W.integrate(f)
        fx_val = assemble(fx * dx) * assemble(fy * dx)
        
        sp_val, _ = dblquad(f_eval, 0, 1, lambda x: 0, lambda x: 1)
        
        assert abs(f_int - fx_val) < 1e-3
        assert abs(fx_val - value) < 1e-3
        assert abs(f_int - value) < 1e-3
        
    def test_1d_integral(self):
        n = 15
        mesh = UnitIntervalMesh(n-1)
        V = FunctionSpace(mesh, 'CG', 1)
        W = ProductFunctionSpace(V)
        
        def f_eval(x, y):
            return sin(x-y) * log(x + y + 1)
        
        f = to_Function(f_eval, W)
        value = W.integrate(f)
        sp_val, _ = dblquad(f_eval, 0, 1, lambda x: 0, lambda x:1)
        
        assert abs(value - sp_val) < 1e-3
        
    def test_2d_analytic_integral(self):
        n = 15
        mesh = UnitSquareMesh(n-1, n-1)
        V = FunctionSpace(mesh, 'CG', 1)
        W = ProductFunctionSpace(V)
        
        def fx_eval(*x):
            x1, x2 = x
            return cos(x1) * sin(x2) * exp(x1 + x2)
        fx_int = 0.25 * (-1 + 2 * e * cos(1) - e**2 * cos(2))
        
        def fy_eval(*y):
            y1, y2 = y
            return log(y1 + 1) * y2**2
        fy_int = (log(4) - 1) / 3
        
        def f_eval(x, y):
            return fx_eval(*x) * fy_eval(*y)
        f_int = fx_int * fy_int
        
        fx = to_Function(fx_eval, V)
        fy = to_Function(fy_eval, V)
        f = to_Function(f_eval, W)
        
        value = W.integrate(f)
        fx_val = assemble(fx * dx) * assemble(fy * dx)
        
        def f_quad(x1, x2, y1, y2):
            return f_eval([x1, x2], [y1, y2])
        sp_val, _ = nquad(f_quad, [[0,1], [0,1], [0,1], [0,1]])
        
        assert abs(f_int - fx_val) < 1e-3
        assert abs(fx_val - sp_val) < 1e-3
        assert abs(sp_val - value) < 1e-3
        assert abs(f_int - value) < 1e-3
        
    def test_2d_integral(self):
        n = 15
        mesh = UnitSquareMesh(n-1, n-1)
        V = FunctionSpace(mesh, 'CG', 1)
        W = ProductFunctionSpace(V)
        
        def f_quad(x1, x2, y1, y2):
            return sin(x1 - y1) * log(x2 + y2 + 1)
        sp_val, _ = nquad(f_quad, [[0,1], [0,1], [0,1], [0,1]])
        
        def f_eval(x, y):
            x1, x2 = x
            y1, y2 = y
            return f_quad(x1, x2, y1, y2)
        f = to_Function(f_eval, W)
        value = W.integrate(f)
        
        assert abs(sp_val - value) < 1e-3