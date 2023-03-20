import pytest
from numpy import e, eye, exp, isnan, mean, ones_like, log, divide, sin, cos, corrcoef
from numpy.random import randn, uniform, seed
from product_fem import ProductFunctionSpace, to_Function, Control, SpatialData
from product_fem.loss_functionals import L2Error, L2Regularizer, SmoothingRegularizer
from fenics import UnitIntervalMesh, UnitSquareMesh, FunctionSpace, \
     assemble, dx, Function


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
        nx, ny = 25, 20
        mesh = UnitSquareMesh(nx-1, ny-1)
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
        n = 15
        mesh = UnitSquareMesh(n-1, n-1)
        V = FunctionSpace(mesh, 'CG', 1)
        W = ProductFunctionSpace(V)
        h = mesh.hmin()
        
        def d_eval(x, y):
            return exp(-x[0] - x[1] - y[0] - y[1])
        d = to_Function(d_eval, W)
        E = L2Error(d)
        
        def u_eval(x, y):
            return exp(x[0] + x[1] + y[0] + y[1])
        u = to_Function(u_eval, W)
        
        def central_diff(i, j):
            phi_ij = W._basis_ij(i, j)
            E_right = E(u + h * phi_ij)
            E_left = E(u - h * phi_ij)
            
            return (E_right - E_left) / (2 * h)
        
        diffs = [central_diff(i,j) for i,j in W.dofs()]
        dE = E.derivative(u)
        
        assert abs(corrcoef(diffs, dE)[0,1] - 1) < 5e-4
    
    def test_2d_l2_error_sum_derivative(self):
        seed(1234)
        points = uniform(0, 1, 10).reshape(5, 2)
        N = int(len(points) * (len(points) + 1) / 2)
        data = randn(N)
        
        mesh = UnitSquareMesh(9, 9)
        V = FunctionSpace(mesh, 'CG', 1)
        W = ProductFunctionSpace(V)
        
        xy0 = points[[i for i in range(points.shape[0]) for j in range(points.shape[0]) if i <= j], :]
        xy1 = points[[j for i in range(points.shape[0]) for j in range(points.shape[0]) if i <= j], :]
        data = SpatialData(data=data, xy0=xy0, xy1=xy1, W=W)
        J = L2Error(data)
        
        def u_func(x, y):
            return exp(x[0] - y[1]) - exp(3 * y[0] - 5 * x[1])
        u = to_Function(u_func, W)
        
        # taylor test dJdu 
        def taylor_test(J, u):
            Ju = J.evaluate(u)
            dJ = J.derivative(u)
            e = eye(u.dim())

            # taylor remainder |J - Jp - h * dJ| = O(h^2)
            def remainder(h):
                Jp = []
                for i in range(u.dim()):
                    if i==0:
                        u.assign(u.array + h * e[i])
                    else:
                        u.assign(u.array - h * e[i-1] + h * e[i])

                    Jp.append(J.evaluate(u))

                u.assign(u.array - h * e[i])
                return [abs(Jp[i] - Ju - h * dJ[i]) for i in range(u.dim())]

            c = 2
            hs = [1. / (c**k) for k in range(1, 4)]
            rs = [remainder(h) for h in hs]

            # rates should all be close to 2
            rates = [log(divide(rs[i-1], rs[i])) / log(c) 
                     for i in range(1, len(rs))]
            return rates
        
        rates = taylor_test(J, u)
        rates = [r[~isnan(r)] for r in rates]
        
        twos = 2 * ones_like(rates[0])
        assert mean((rates[0] - twos)**2) < 5e-16
        
        twos = 2 * ones_like(rates[1])
        assert mean((rates[1] - twos)**2) < 5e-15

    def test_1d_l2_reg(self):
        n = 25
        mesh = UnitIntervalMesh(n-1)
        V = FunctionSpace(mesh, 'CG', 1)
        
        m = to_Function(lambda x: sin(x), V)
        m = Control(m)
        R = L2Regularizer(m, 2.0)
        
        Rm = R(m)
        Rm_true = (2 - sin(2)) / 4
        
        assert abs(Rm - Rm_true) < 1e-4
        
    def test_2d_l2_reg(self):
        n = 15
        mesh = UnitSquareMesh(n-1, n-1)
        V = FunctionSpace(mesh, 'CG', 1)
        
        m = to_Function(lambda x,y: sin(x-y), V)
        m = Control(m)
        R = L2Regularizer(m, 2.0)
        
        Rm = R(m)
        Rm_true = cos(1)**2 / 2
        
        assert abs(Rm - Rm_true) < 1e-3
    
    def test_1d_smoothing_reg(self):
        n = 25
        mesh = UnitIntervalMesh(n-1)
        V = FunctionSpace(mesh, 'CG', 1)
        
        m = to_Function(lambda x: sin(x), V)
        m = Control(m)
        R = SmoothingRegularizer(m, 2.0)
        
        Rm = R(m)
        Rm_true = (2 + sin(2)) / 4
        
        assert abs(Rm - Rm_true) < 1e-4
    
    def test_2d_smoothing_reg(self):
        n = 25
        mesh = UnitSquareMesh(n-1, n-1)
        V = FunctionSpace(mesh, 'CG', 1)
        
        m = to_Function(lambda x,y: sin(x-y), V)
        m = Control(m)
        R = SmoothingRegularizer(m, 2.0)
        
        Rm = R(m)
        Rm_true = 1 + sin(1)**2
        
        assert abs(Rm - Rm_true) < 1e-3
    
#     def test_1d_l2_reg_derivative(self):
#         pass
    
#     def test_2d_l2_reg_derivative(self):
#         pass
    
#     def test_1d_smoothing_reg_derivative(self):
#         pass
    
#     def test_2d_smoothing_reg_derivative(self):
#         pass
