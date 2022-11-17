from fenics import assemble, inner, dx, grad, Function, Constant
from ufl import derivative
from .transforms import to_Function
from .functions import ProductFunction, SpatialData
import numpy as np


class Functional:
    """Functional defined from ufl.Form with Control m"""
    
    def __init__(self, control, ufl_form):
        self.control = control
        assert not ufl_form.arguments()
        self.ufl_form = ufl_form
        for m in control:
            assert m in self.ufl_form.coefficients()
        self._derivative_forms = None
        
    def __call__(self, control):
        return self.evaluate(control)
        
    def evaluate(self, control):
        self.control.update(control)
        return assemble(self.ufl_form)
    
    def derivative_forms(self):
        if self._derivative_forms is None:
            self._derivative_forms = self._get_derivative_forms()
        return self._derivative_forms
    
    def _get_derivative_forms(self):
        dJ_forms = []
        for m in self.control:
            dJdm = []
            for basis_func_i in self.control.get_basis(m):
                dJdm.append(derivative(self.ufl_form, m, basis_func_i))
                
            dJ_forms.append(dJdm)
        return dJ_forms
    
    def derivative_component(self, i, m):
        assert m in self.ufl_form.coefficients()
        dJ_forms = self.derivative_forms()
        j = self.control.argwhere(m)
        return dJ_forms[j][i]
    
    def derivative(self, control):
        dJ = []
        self.control.update(control)
        for m in self.control:
            assert m in self.ufl_form.coefficients()
            for i in range(m.function_space().dim()):
                dm_i = self.derivative_component(i, m)
                dJ.append(dm_i)
        return dJ
    
    
class L2Regularizer(Functional):
    """Defined by J(m) = alpha/2 int inner(m, m) dx"""
    
    def __init__(self, control, alpha):
        def l2_reg(m, a):
            return Constant(a/2) * inner(m, m) * dx
        
        if isinstance(alpha, (int, float)):
            assert len(control)==1
            forms = [l2_reg(m, alpha) for m in control]
        elif isinstance(alpha, (list, tuple)):
            assert len(control)==len(alpha)
            forms = [l2_reg(m, a) for m, a in zip(control, alpha)]
        super().__init__(control, sum(forms))


class SmoothingRegularizer(Functional):
    """Defined by J(m) = alpha/2 int inner(grad(m), grad(m)) dx"""
    
    def __init__(self, control, alpha):
        def smooth_reg(m, a):
            return Constant(a/2) * inner(grad(m), grad(m)) * dx
        
        if isinstance(alpha, (int, float)):
            assert len(control)==1
            forms = [smooth_reg(m, alpha) for m in control]
        elif isinstance(alpha, (list, tuple)):
            assert len(control)==len(alpha)
            forms = [smooth_reg(m, a) for m, a in zip(control, alpha)]
        super().__init__(control, sum(forms))


class L2ErrorSum:
    """Defined by J(u) = 1/2 sum (u - u_d)^2 from SpatialData u_d
    and the sum is over all sample points
    """
    
    def __init__(self, data):
        self.data = data.data
        self.W = data.W
        self.sample_points = data.points
        self.eval_matrix = data.eval_matrix
    
    def __call__(self, u):
        return self.evaluate(u)
    
    def evaluate(self, u):
        # evaluate u at sample points
        P = self.eval_matrix
        u_eval = P.dot(u.array)
        sq_error = np.sum(1/2 * (u_eval - self.data)**2) / len(P)
        return sq_error

    def derivative(self, u):
        """The ith derivative component is 
            (dJ/du)_i = sum_{sample points} (u - data) * phi_i
        where phi_i is the ith basis element in W
        """
        P = self.eval_matrix
        u_eval = P.dot(u.array)
        dJdu = P.T.dot(u_eval - self.data) / len(P)
        return dJdu
    
    
class L2ErrorIntegral:
    """Defined by J(u) = 1/2 int (u - u_d)^2 dxdy 
    given ProductFunction u_d"""
    
    def __init__(self, data, weights=None):
        self.data = data
        if weights is None:
            weights = np.ones(len(data))
        else:
            assert isinstance(weights, np.ndarray)
            assert len(weights)==len(data)
        self.weights = to_Function(weights, data.function_space())
        
    def __call__(self, u):
        return self.evaluate(u)
        
    def evaluate(self, u):
        W = u.function_space()
        error = 1/2 * (self.weights * (u - self.data)).array**2
        return W.integrate(error)
    
    def derivative(self, u):
        """"Returns the directional derivative of J wrt u
        in the direction of the ij basis element for u, 
        which is of the form phi_i(x)phi_j(y).
        
        Here there's a little trickery: When we integrate f(x,y)
        against a product basis element phi_i(x)phi_j(y) we have
        int f phi_i phi_j dxdy = sum_kl f_kl int phi_k phi_i dx phi_l phi_j dy
        this is just f dotted with the product mass 
        M_ik,jl := (int phi_k phi_i dx) (int phi_l phi_j dy)"""
        r = (self.weights * (u - self.data)).as_matrix()
        pmass = u.function_space().product_mass()
        dJdu = pmass.dot(r.dot(pmass)).flatten()
        assert len(dJdu)==len(u)
        return dJdu

    
class L2Error:
    def __new__(cls, data):
        if isinstance(data, ProductFunction):
            return L2ErrorIntegral(data)
        elif isinstance(data, SpatialData):
            return L2ErrorSum(data)
        
        
class LossFunctional:
    """The default loss functional has 3 parts: ``L2 error + smoothing reg + L2 reg``
    ```
    Loss(u_d, m) = int (u - u_d)^2 dx + alpha int (grad(m)^2 + m^2) dx
    ```
    So ``dJdu = int (u - u_d) u_ dx``
    and ``dJdm = int grad(m) m_ dx + int m * m_ dx``.
    """
    
    def __init__(self, data, control, reg):
        self.data = data
        self.control = control
        self.reg_constants = reg
        self.l2_error = L2Error(data)
        self.l2_reg = L2Regularizer(control, reg['l2'])
        self.smoothing_reg = SmoothingRegularizer(control, reg['smoothing'])
        
    def __call__(self, u, m):
        return self.evaluate(u, m)
        
    def evaluate(self, u, m):
        loss = self.l2_error.evaluate(u)
        loss += self.l2_reg.evaluate(m)
        loss += self.smoothing_reg.evaluate(m)
        return loss
    
    def derivative_component(self, i, m):
        dJdm = self.l2_reg.derivative_component(i, m)
        dJdm += self.smoothing_reg.derivative_component(i, m)
        dJdm = assemble(dJdm)
        assert isinstance(dJdm, float)
        return dJdm
    
    def derivative(self, control):
        dJdm = []
        for m in control:
            for i in range(m.function_space().dim()):
                dJ_i = self.derivative_component(i, m)
                dJdm.append(dJ_i)
        return dJdm
    
    def partial_u(self, u):
        return self.l2_error.derivative(u)
    

class ReducedLossFunctional:
    
    def __init__(self, equation, loss):
        self.eqn = equation
        self.loss = loss
        
    def __call__(self, m):
        u = self.eqn.solve(m)
        return self.loss(u, m)
        
    def derivative(self, control):
        return self.loss.derivative(control)
    
    