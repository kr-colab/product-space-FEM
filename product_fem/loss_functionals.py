from fenics import assemble, inner, dx, grad
from ufl import derivative
from numpy import array, dot


class Functional:
    """Functional defined from ufl.Form with Control m"""
    
    def __init__(self, ufl_form):
        assert not ufl_form.arguments()
        self.ufl_form = ufl_form
        
    def __call__(self, control):
        return self.evaluate(control)
        
    def evaluate(self, control):
        for m in control:
            assert m in self.ufl_form.coefficients()
        return assemble(self.ufl_form)
    
    def derivative_component(self, i, m):
        assert m in self.ufl_form.coefficients()
        return derivative(self.ufl_form, m, m.basis[i])
    
    def derivative(self, control):
        dJ = []
        for m in control:
            assert m in self.ufl_form.coefficients()
            for i in range(m.dim()):
                dm_i = self.derivative_component(i, m)
                dJ.append(dm_i)
        return dJ
    
    
class L2Regularizer(Functional):
    """Defined by J(m) = alpha/2 int inner(m, m) dx"""
    
    def __init__(self, control, alpha):
        def l2_reg(m, a):
            return a/2 * inner(m, m) * dx
        
        if isinstance(alpha, (int, float)):
            assert len(control)==1
            forms = [l2_reg(m, alpha) for m in control]
        elif isinstance(alpha, (list, tuple)):
            assert len(control)==len(alpha)
            forms = [l2_reg(m, a) for m, a in zip(control, alpha)]
        super().__init__(sum(forms))


class SmoothingRegularizer(Functional):
    """Defined by J(m) = alpha/2 int inner(grad(m), grad(m)) dx"""
    
    def __init__(self, control, alpha):
        def smooth_reg(m, a):
            return a/2 * inner(grad(m), grad(m)) * dx
        
        if isinstance(alpha, (int, float)):
            assert len(control)==1
            forms = [smooth_reg(m, alpha) for m in control]
        elif isinstance(alpha, (list, tuple)):
            assert len(control)==len(alpha)
            forms = [smooth_reg(m, a) for m, a in zip(control, alpha)]
        super().__init__(sum(forms))


class L2Error:
    """Defined by J(u) = 1/2 int (u - u_d)^2 dxdy 
    given SpatialData u_d"""
    
    def __init__(self, spatial_data):
        self.data = spatial_data
        
    def __call__(self, u):
        return self.evaluate(u)
        
    def evaluate(self, u):
        W = u.function_space()
        error = 1/2 * (u - self.data).array**2
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
        pmass = u.function_space().product_mass()
        r = (u - self.data).as_matrix()
        dJdu = pmass.dot(r.dot(pmass)).flatten()
        assert len(dJdu)==len(u)
        return dJdu

    
class LossFunctional:
    """
    The default loss functional has 3 parts: ``L2 error + smoothing reg + L2 reg``
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
            for i in range(m.dim()):
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
    
    