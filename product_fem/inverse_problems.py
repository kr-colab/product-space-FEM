from .transforms import tensordot
import scipy.optimize as opt
import numpy as np


class InverseProblem:
    
    def __init__(self, equation, loss):
        self.equation = equation
        self.loss = loss
        self.solver = equation.solver
        self.assembler = equation.assembler
        
    # solving adjoint requires knowledge of dJdu    
    def solve_adjoint(self, u):
        # adjoint lhs is transpose(lhs from equation)
        A = self.equation.A.T
        
        # adjoint rhs is dJdu
        b = -self.loss.partial_u(u)
        return self.solver.solve(A, b)

#     def assemble_partials(self, m):
#         """Here we assemble dJ/dm for loss J"""
#         dAdm, dbdm = self.equation.partial_m(m)
#         dJdm = self.loss.partial_m(m)
#         return dAdm, dbdm, dJdm
        
    def _gradient_component(self, p, i, u, m):
        dAdm, dbdm = self.equation.derivative_component(i, m)
        dJdm = self.loss.derivative_component(i, m)
        
        dFdm = dbdm - dAdm.dot(u.array)
        gradient = -p.dot(dFdm) + dJdm
        return gradient
        
    def compute_gradient(self, control):
        u = self.equation.solve(control)
        p = self.solve_adjoint(u)
        gradient = []
        for m in control:
            for i in range(m.dim()):
                grad_i = self._gradient_component(p, i, u, m)
                gradient.append(grad_i)
        return np.array(gradient)
        
    def _compute_gradient(self, control):
        u = self.equation.solve(control)
        p = self.solve_adjoint(u)
        grad = np.zeros(control.dim())
        
        dAdm, dbdm = self.equation.derivative(control)
        dFdm = dbdm - tensordot(dAdm, u, axes=(1,0))
        dJdm = self.loss.derivative(control)
        grad = -p.dot(dFdm) + dJdm
        return grad
        
    # in case we want to update regularization parameters
    def set_loss(self, loss):
        self.loss = loss
        
    def compute_loss(self, m):
        u = self.equation.solve(m)
        return self.loss.evaluate(u, m)
    
    def loss_and_grad(self, m):
        # assume m is array
        control = self.equation.control
        control.update(m)
        loss = self.compute_loss(control)
        grad = self.compute_gradient(control)
        return loss, grad
    
    # do scipy optimize
    def optimize(self, m0, method='L-BFGS-B', *args, **kwargs):
        fun = self.loss_and_grad
        jac = True
        options = kwargs.get('options')
        results = opt.minimize(fun, m0.array(), args, method, jac, options)
        m0.update(results['x'])
        return m0, results
        
    def plot_results(self):
        """This can be different for each inverse equation
        but should include plots of m_hat and m_true and
        plots of u_hat and u_true and/or residuals."""
        raise NotImplementedError