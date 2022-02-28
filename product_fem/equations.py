from abc import ABC, abstractmethod
import numpy as np
import product_fem as pf
from fenics import *


# BASE CLASS
# L(m)(u) = f
# parameters: m
class Equation(ABC):
    def __init__(self, W, f, bc):
        self.W 
        self.V = W.marginal_function_space
        self.fx, self.fy = f
        self.bc = bc
        
    def solve(self, m, *args):
        u, v = TrialFunction(self.V), TestFunction(self.V)
        
        Ax_forms = [] # int dx
        Ay_forms = [] # int dy
        A_forms = list(zip(Ax_forms, Ay_forms))
        
        bx_forms = [fx * v * dx for fx in self.fx]
        by_forms = [fy * v * dx for fy in self.fy]
        b_forms = list(zip(bx_forms, by_forms))
        
        A, b = pf.assemble_product_system(A_forms, b_forms, self.bc)
        U = np.linalg.solve(A, b)
        self.stiffness = A
        return U
    
    @abstractmethod
    def solve_adjoint(self, m, u, u_d=None):
        p, v = TrialFunction(self.V), TestFunction(self.V)
        mass = assemble(p * v * dx).array()
        u_d = np.zeros_like(u) if u_d is None else u_d
        resid = -(u - u_d)
        b_adj = resid.T.dot(np.kron(mass, mass))

        A_adj, b_adj = self.bc.apply(self.stiffness.T, b_adj)
        p = np.linalg.solve(A_adj, b_adj).T
        return p
    
    # meat and potatoes
    def assemble_partials(self, m, alpha=0):
        pass
    
    # can probably inherit this 
    @abstractmethod
    def compute_gradient(self):
        # given m solve for u
        u = self.solve(m)
        
        # given m, u solve for p
        p = self.solve_adjoint(m, u)
        
        # given m, u, p compute ∂A/∂m and ∂J/∂m
        dAdm, dbdm, dJdm = self.assemble_partials(m, alpha)
        
        # compute gradient dJ/dm = p * (∂b/∂m - ∂A/∂m * u) + ∂J/∂m
        dFdm = dbdm - np.tensordot(dAdm, u, axes=(1,0))
        grad = p.dot(dFdm) + dJdm
        return grad

    
# -Laplacian(u) = f
# parameters: none
class Poisson:
    def __init__(self, W, f, bc):
        self.W = W
        self.V = W.marginal_function_space
        self.fx, self.fy = f
        self.bc = bc
        
    def solve(self):
        u, v = TrialFunction(self.V), TestFunction(self.V)
        
        Ax_forms = [u.dx(0) * v.dx(0) * dx, u * v * dx]
        Ay_forms = [u * v * dx, u.dx(0) * v.dx(0) * dx]
        A_forms = list(zip(Ax_forms, Ay_forms))
        
        bx_forms = [fx * v * dx for fx in self.fx]
        by_forms = [fy * v * dx for fy in self.fy]
        b_forms = list(zip(bx_forms, by_forms))
        
        A, b = pf.assemble_product_system(A_forms, b_forms, self.bc)
        U = np.linalg.solve(A, b)
        return U
    

# -grad•(exp(m) grad(u)) = f 
# parameters: m (array)
class ExpDiffusion:
    def __init__(self, W, f, bc):
        self.W = W
        self.V = W.marginal_function_space()
        self.f = f # f(x,y) = f1(x)f2(y)
        self.bc = bc
        
    def solve(self, m):
        # m = (mx, my)
        u, v = TrialFunction(self.V), TestFunction(self.V)
        
        # when m is given as a string
        if isinstance(m[0], str):
            mx, my = [interpolate(Expression(_m, element=self.V.ufl_element()), self.V) for _m in m]
            self.ms = (mx, my)
        # when m is given as an array
        elif isinstance(m[0], np.ndarray):
            mx, my = Function(self.V), Function(self.V)
            mx.vector()[:] = m[0]
            my.vector()[:] = m[1]
            self.ms = (mx, my)
        
        # compute int grad(m)*grad(m) dxdy for self.loss_functional()
        gmgm = assemble(mx.dx(0)**2 * dx) * assemble(1 * dx(domain=self.V.ufl_domain()))
        gmgm += assemble(1 * dx(domain=self.V.ufl_domain())) * assemble(my.dx(0)**2 * dx)
        self.gmgm = gmgm
        
        fx, fy = [interpolate(Expression(f, element=self.V.ufl_element()), self.V) for f in self.f]
        
        Ax_forms = [exp(mx) * u.dx(0) * v.dx(0) * dx,
                   exp(mx) * u * v * dx]
        Ay_forms = [exp(my) * u * v * dx,
                   exp(my) * u.dx(0) * v.dx(0) * dx]
        A_forms = list(zip(Ax_forms, Ay_forms))
        
        bx_forms = [fx * v * dx]
        by_forms = [fy * v * dx]
        b_forms = list(zip(bx_forms, by_forms))
        
        A, b = pf.assemble_product_system(A_forms, b_forms, self.bc)
        U = np.linalg.solve(A, b)
        self.stiffness = A
        return U
    
    def loss_functional(self, m, u_d, alpha):
        # J = int (u - u_d)^2 dx + alpha/2 int grad(m)•grad(m) dx
        u = self.solve(m)
        J = self.W.integrate((u - u_d)**2) + alpha/2 * self.gmgm
        return J
        
    def solve_adjoint(self, m, u, u_d=None):
        p, v = TrialFunction(self.V), TestFunction(self.V)
        mass = assemble(p * v * dx).array()
        u_d = np.zeros_like(u) if u_d is None else u_d
        resid = -(u - u_d)
        b_adj = resid.T.dot(np.kron(mass, mass))

        A_adj, b_adj = self.bc.apply(self.stiffness.T, b_adj)
        p = np.linalg.solve(A_adj, b_adj).T
        return p
    
    def assemble_partials(self, m, alpha):
        Wdim = self.W.dim()
        mx, my = self.ms
        my_dim = mx_dim = len(mx.vector()[:])
        # assemble dbdm, dAdm, dJdm
        # here b is independent of m so dbdm = 0
        dbdm = np.zeros((Wdim, Wdim))

        # compute gradients of A and J wrt m_rs(x,y) = phi_r(x) phi_s(y)
        dofs = self.W.dofmap.marginal_dofs
        dAdm = np.zeros((Wdim, Wdim, Wdim))
        dJdm = np.zeros((1, Wdim))
        
        # TODO: speed up this nested loop
        u, v = TrialFunction(self.V), TestFunction(self.V)
        for r in dofs:
            phi_r = Function(self.V)
            phi_r.vector()[r] = 1.
            for s in dofs:
                rs = s + r * len(dofs)
                phi_s = Function(self.V)
                phi_s.vector()[s] = 1.

                # dAdm = (phi_r phi_s e^m) inner(grad(u), grad(p)) dxdy
                dAdm_forms = list(zip(
                    [phi_r * exp(mx) * u.dx(0) * v.dx(0) * dx, phi_r * exp(mx) * u * v * dx], 
                    [phi_s * exp(my) * u * v * dx, phi_s * exp(my) * u.dx(0) * v.dx(0) * dx]
                ))
                dAdm[:,:,rs] = pf.assemble_kron(dAdm_forms)

                # dJdm = alpha inner(grad(m), grad(phi_r phi_s)) dxdy
                dJdm_forms = list(zip(
                    [phi_r.dx(0) * mx.dx(0) * dx, phi_r * dx], 
                    [phi_s * dx, phi_s.dx(0) * my.dx(0) * dx]
                ))
                dJdm[0,rs] = alpha * pf.assemble_kron(dJdm_forms)
                
        return dAdm, dbdm, dJdm
    
    def compute_gradient(self, m, alpha):
        # given m solve for u
        u = self.solve(m)
        
        # given m, u solve for p
        p = self.solve_adjoint(m, u)
        
        # given m, u, p compute ∂A/∂m and ∂J/∂m
        dAdm, dbdm, dJdm = self.assemble_partials(m, alpha)
        
        # compute gradient = p * (dbdm - dAdm * u) + dJdm
        dFdm = dbdm - np.tensordot(dAdm, u, axes=(1,0))
        grads = p.dot(dFdm) + dJdm
        
        return grads

    
# NOTE: this is not functioning properly 
# -eps Laplacian(u) + b•grad(u) = f
# parameters: eps (float), b tuple(b1, b2), b1 (array), b2 (array)
# assumes drift b = (b_1(x), b_2(y))
class DriftDiffusion:
    def __init__(self, W, f, bc):
        self.W = W            # product function space
        self.V = W.V          # marginal function space
        self.fx, self.fy = f  # forcing function
        self.bc = bc          # boundary conditions
        
    def solve(self, eps, b):
        u, v = TrialFunction(self.V), TestFunction(self.V)
        b1, b2 = self.b_as_function(b)
        
        fxs = [pf.to_Function(fx, self.V) for fx in self.fx]
        fys = [pf.to_Function(fy, self.V) for fy in self.fy]
        
        Ax_forms = [eps * u.dx(0) * v.dx(0) * dx,
                   u * v * dx,
                   b1 * u.dx(0) * v * dx,
                   u * v * dx]
        Ay_forms = [u * v * dx,
                   eps * u.dx(0) * v.dx(0) * dx,
                   u * v * dx,
                   b2 * u.dx(0) * v * dx]
        A_forms = list(zip(Ax_forms, Ay_forms))
        
        bx_forms = [fx * v * dx for fx in fxs]
        by_forms = [fy * v * dx for fy in fys]
        b_forms = list(zip(bx_forms, by_forms))
        
        A, b = pf.assemble_product_system(A_forms, b_forms, self.bc)
        U = np.linalg.solve(A, b)
        self.stiffness = A
        return U
    
    def b_as_function(self, b):
        b1, b2 = b
        b1 = pf.to_Function(b1, self.V)
        b2 = pf.to_Function(b2, self.V)
        return b1, b2
            
    def loss_functional(self, eps, b, u_d, alpha, beta):
        b1, b2 = self.b_as_function(b)
        u = self.solve(eps, (b1, b2))
        J = 1/2 * self.W.integrate((u - u_d)**2) 
        J += alpha/2 * assemble((b1**2 + b2**2) * dx) 
        J += beta/2 * eps**2
        return J
        
    def solve_adjoint(self, eps, b, u, u_d=None):
        p, v = TrialFunction(self.V), TestFunction(self.V)
        mass = assemble(p * v * dx).array()
        u_d = np.zeros_like(u) if u_d is None else u_d
        resid = -(u - u_d)
        b_adj = resid.T.dot(np.kron(mass, mass))

        A_adj, b_adj = self.bc.apply(self.stiffness.T, b_adj)
        p = np.linalg.solve(A_adj, b_adj).T
        return p
    
    # meat and potatoes: assemble dbdm, dAdm, dJdm
    def assemble_partials(self, eps, b, alpha, beta):
        b1, b2 = b
        b1 = pf.to_Function(b1, self.V)
        b2 = pf.to_Function(b2, self.V)
        
        V_dim = self.V.dim()
        b_dim = 2 * V_dim
        W_dim = self.W.dim()
        dom = self.V.ufl_domain() # for integrating 1 * dx
        
        # here b is independent of m so dbdm = 0
        dbdb = np.zeros((self.W.dim(), b_dim))

        # compute gradients of A and J wrt eps and 
        # b_r(x,y) = (phi_r(x), 0) for 1 <= r <= n
        # b_r(x,y) = (0, phi_r(y)) for n < r <= 2n
        dAde = np.zeros((W_dim, W_dim))
        dAdb = np.zeros((W_dim, W_dim, b_dim))
        dJde = beta * eps
        dJdb = np.zeros((1, b_dim))
        
        uu, v = TrialFunction(self.V), TestFunction(self.V)
        dAde_forms = list(zip([uu.dx(0) * v.dx(0) * dx, uu * v * dx], 
                              [uu * v * dx, uu.dx(0) * v.dx(0) * dx]))
        dAde = pf.assemble_kron(dAde_forms)
        for r in range(V_dim):
            rr = r + V_dim # to index (0, phi_r) assembly
            phi_r = Function(self.V)
            phi_r.vector()[r] = 1.

            # dAdb1 = (phi_r u' v dx) (u v dy)
            # dAdb2 = (u v dx) (phi_r u' v dy)
            dAdb1_forms = list(zip([phi_r * uu.dx(0) * v * dx], [uu * v * dx]))
            dAdb2_forms = list(zip([uu * v * dx], [phi_r * uu.dx(0) * v * dx]))
            dAdb[:,:,r] = pf.assemble_kron(dAdb1_forms)
            dAdb[:,:,rr] = pf.assemble_kron(dAdb2_forms)

            # dJdb1 = (b1 * phi_r dx) (1 dy)
            # dJdb2 = (1 dx) (b2 * phi_r dx)
            dJdb1_forms = list(zip([b1 * phi_r * dx], [1 * dx(domain=dom)]))
            dJdb2_forms = list(zip([1 * dx(domain=dom)], [b2 * phi_r * dx]))
            dJdb[0,r] = alpha * pf.assemble_kron(dJdb1_forms)
            dJdb[0,rr] = alpha * pf.assemble_kron(dJdb2_forms)
        
        return dAdb, dbdb, dJdb
    
    # can probably inherit this 
    def compute_gradient(self, eps, b, alpha, beta):
        # given m solve for u
        u = self.solve(eps, b)
        
        # given m, u solve for p
        p = self.solve_adjoint(eps, b, u)
        
        # given m, u, p compute ∂A/∂m, ∂b/∂m, and ∂J/∂m
        dAdm, dbdm, dJdm = self.assemble_partials(eps, b, alpha, beta)
        
        # compute gradient = p * (∂b/∂m - ∂A/∂m * u) + ∂J/∂m
        dFdm = dbdm - np.tensordot(dAdm, u, axes=(1,0))
        grad = p.dot(dFdm) + dJdm
        return grad
    
    def b_str_to_array(self, b):
        # b = [b1, b2] as strings
        b1, b2 = b
        b1 = pf.to_Function(b1, self.V)
        b2 = pf.to_Function(b2, self.V)
        return np.concatenate((b1.vector()[:], b2.vector()[:]))
    
    def verify_gradient(self, h, eps, b, b_, alpha=0.1, beta=0.1, u_d=None):
        # assumes b, b_ are lists of strings, need to accept ndarrays
        if u_d is None:
            u_d = np.zeros(self.W.dim())
        def grad_error(h, b, b_):
            b_hb = [f'{b[i]} + {h} * ({b_[i]})' for i in range(len(b))]
            _J = self.loss_functional(eps, b_hb, u_d, alpha, beta)
            J = self.loss_functional(eps, b, u_d, alpha, beta)
            grads = self.compute_gradient(eps, b, alpha, beta)
            
            _b_ = self.b_str_to_array(b_)
            error = _J - J - h * np.dot(grads, _b_)
            return error
        
        hs = [h / (2**k) for k in range(5)]
        es = [grad_error(h_, b, b_) for h_ in hs]
        err_rates = [(es[i] / es[i+1]).item() for i in range(len(es)-1)]
        return err_rates
    