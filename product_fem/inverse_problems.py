from .transforms import dense_to_PETSc
from .functions import Control
from .boundary_conditions import ProductDirichletBC
import scipy.optimize as opt
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from fenics import plot
from .plotting import plot_ellipse_field


def taylor_test(invp):
    m = invp.equation.control
    Jm = invp.compute_loss(m)
    dJ = invp.compute_gradient(m)
    e = np.eye(m.dim())

    # taylor remainder |J - Jp - h * dJ| = O(h^2)
    def remainder(h):
        Jp = []
        for i in range(m.dim()):
            if i==0:
                m.update(m.array() + h * e[i])
            else:
                m.update(m.array() - h * e[i-1] + h * e[i])

            Jp.append(invp.compute_loss(m))

        m.update(m.array() - h * e[i])
        return [abs(Jp[i] - Jm - h * dJ[i]) for i in range(m.dim())]

    c = 10
    hs = [1. / (c**k) for k in range(1, 4)]
    rs = [remainder(h) for h in hs]

    # rates should all be close to 2
    rates = [np.log(np.divide(rs[i-1], rs[i])) / np.log(c) for i in range(1, len(rs))]
    return rates


class InverseProblem:
    """
    :param Equation equation:
    :param LossFunctional loss:
    """
    
    def __init__(self, equation, loss):
        self.equation = equation
        self.loss = loss
        self.solver = equation.solver
        self.assembler = equation.assembler
        self.adjoint_bc = self._set_adjoint_bc(equation.bc)
        
    def _set_adjoint_bc(self, bc):
        W = self.equation.bc.product_function_space
        u_bdy = 0
        on_boundary = bc.on_boundary
        return ProductDirichletBC(W, u_bdy, on_boundary)
    
    # solving adjoint requires knowledge of dJdu    
    def solve_adjoint(self, u):
        # adjoint lhs is transpose(lhs from equation)
        A = self.equation.A.transpose()
        
        # adjoint rhs is dJdu
        b = -self.loss.partial_u(u)
        
        # enforce adjoint boundary condition: p=0 on boundary
        A, b = self.adjoint_bc.apply(A, b)
        
        return self.solver.solve(A, b)
        
    def _gradient_component(self, p, i, u, m):
        dAdm, dbdm = self.equation.derivative_component(i, m)
        dJdm = self.loss.derivative_component(i, m)
        
        u = dense_to_PETSc(u.array)
        dFdm = dbdm - dAdm * u
        
        p = dense_to_PETSc(p.array)
        gradient = -p.dot(dFdm) + dJdm
        return gradient
        
    def compute_gradient(self, control):
        u = self.equation.solve(control)
        p = self.solve_adjoint(u)
        gradient = []
        for m in control:
            for i in range(m.function_space().dim()):
                grad_i = self._gradient_component(p, i, u, m)
                gradient.append(grad_i)
        return np.array(gradient)
        
    def _compute_gradient(self, control):
        u = self.equation.solve(control)
        p = self.solve_adjoint(u)
        grad = np.zeros(control.dim())
        
        dAdm, dbdm = self.equation.derivative(control)
        dFdm = dbdm - np.tensordot(dAdm, u, axes=(1,0))
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
    
    def optimize(self, m0, method='L-BFGS-B', *args, **kwargs):
        allvecs = [m0.array()]
        def callback(m):
            allvecs.append(m)
            control = self.equation.control
            control.update(m)
            print(self.compute_loss(control))
            
        fun = self.loss_and_grad
        jac = True
        options = kwargs.get('options', {})
        results = opt.minimize(fun, m0.array(), args, method, jac, callback=callback, options=options)
        m0.update(results['x'])
        return allvecs, results
        
    # NOTE: this isn't a great place for this function since this only plots 
    # the control in the HittingTimes2D equation
    def animate(self, m_hats, save_as, **kwargs):
        # initialize figure and axes 
        m = self.equation.control
        m.update(m_hats[0])
        fig, ax = plt.subplots(1, 2, dpi=150)

        q = ax[1].quiver(np.empty(49), np.empty(49), np.empty(49), np.empty(49))
        s = ax[0].scatter([], [])

        # plot frame i
        def animate(i):
            nonlocal q, s
            if i % 10 == 0: print(f'animating frame {i} / {len(m_hats)}')
            # update to control at ith iteration
            m.update(m_hats[i])
            q.axes.clear()
            s.axes.clear()
            q = plot(m[0])
            ax[0] = plot_ellipse_field(m[1], ax[0])
            return [q] + ax[0].get_children()

        anim = FuncAnimation(fig, animate, frames=len(m_hats), interval=1, blit=True)
        
        writer = kwargs.get('writer', 'ffmpeg')
        fps = kwargs.get('fps', 18)
        anim.save(save_as, writer='ffmpeg', fps=18)

    def plot_results(self):
        """This can be different for each inverse equation
        but should include plots of m_hat and m_true and
        plots of u_hat and u_true and/or residuals."""
        raise NotImplementedError