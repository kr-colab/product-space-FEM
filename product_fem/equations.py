from abc import ABC, abstractmethod
import numpy as np
import product_fem as pf
from fenics import *


# BASE CLASS
# L_m(u) = f
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
        Wdim = self.W.dim
        mx, my = self.ms
        my_dim = mx_dim = len(mx.vector()[:])
        # assemble dbdm, dAdm, dJdm
        # here b is independent of m so dbdm = 0
        dbdm = np.zeros((Wdim, Wdim))

        # compute gradients of A and J wrt m_rs(x,y) = phi_r(x) phi_s(y)
        dofs = self.W.dofmap.marginal_dofs
        dAdm = np.zeros((Wdim, Wdim, Wdim))
        dJdm = np.zeros((1, Wdim))
        
#         # same but with mx and my
#         dbdmx = np.zeros((Wdim, mx_dim))
#         dbdmy = np.zeros((Wdim, my_dim))
#         dAdmx = np.zeros((Wdim, Wdim, mx_dim))
#         dAdmy = np.zeros((Wdim, Wdim, my_dim))
#         dJdmx = np.zeros((1, mx_dim))
#         dJdmy = np.zeros((1, my_dim))
        
        # TODO: speed up this nested loop
        u, v = TrialFunction(self.V), TestFunction(self.V)
        for r in dofs:
            phi_r = Function(self.V)
            phi_r.vector()[r] = 1.
            
#             dAdmx_forms = list(zip(
#                     [phi_r * exp(mx) * u.dx(0) * v.dx(0) * dx, phi_r * exp(mx) * u * v * dx], 
#                     [exp(my) * u * v * dx, exp(my) * u.dx(0) * v.dx(0) * dx]))
#             dAdmy_forms = list(zip(
#                     [exp(mx) * u.dx(0) * v.dx(0) * dx, exp(mx) * u * v * dx], 
#                     [phi_r * exp(my) * u * v * dx, phi_r * exp(my) * u.dx(0) * v.dx(0) * dx]))
#             dAdmx[:,:,r] = pf.assemble_kron(dAdmx_forms)
#             dAdmy[:,:,r] = pf.assemble_kron(dAdmy_forms)
            
#             dJdmx_forms = list(zip(
#                     [phi_r.dx(0) * mx.dx(0) * dx], 
#                     [1 * dx(domain=self.W.V_mesh)]))
#             dJdmy_forms = list(zip(
#                     [1 * dx(domain=self.W.V_mesh)], 
#                     [phi_r.dx(0) * my.dx(0) * dx]))
#             dJdmx[0,r] = alpha * pf.assemble_kron(dJdmx_forms)
#             dJdmy[0,r] = alpha * pf.assemble_kron(dJdmy_forms)
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
                
#         return (dAdm, dAdmx, dAdmy), (dbdm, dbdmx, dbdmy), (dJdm, dJdmx, dJdmy)
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
        
#         dAdm, dAdmx, dAdmy = dA
#         dbdm, dbdmx, dbdmy = db
#         dJdm, dJdmx, dJdmy = dJ
#         dFdmx = dbdmx - np.tensordot(dAdmx, u, axes=(1,0))
#         gradx = p.dot(dFdmx) + dJdmx
        
#         dFdmy = dbdmy - np.tensordot(dAdmy, u, axes=(1,0))
#         grady = p.dot(dFdmy) + dJdmy
#         return grads, gradx, grady
        return grads

    
# -eps Laplacian(u) + b•grad(u) = f
# parameters: eps (float), b tuple(b1, b2), b1 (array), b2 (array)
# assumes drift b = (b_1(x), b_2(y))
class DriftDiffusion:
    def __init__(self, W, f, bc):
        self.W = W            # product function space
        self.V = W.marginal_function_space
        self.fx, self.fy = f  # forcing function
        self.bc = bc          # boundary conditions
        
    def solve(self, eps, b):
        u, v = TrialFunction(self.V), TestFunction(self.V)
        b1, b2 = b
        
        Ax_forms = [eps * u.dx(0) * v.dx(0) * dx,
                   u * v * dx,
                   b1 * u.dx(0) * v * dx,
                   u * v * dx]
        Ay_forms = [u * v * dx,
                   eps * u.dx(0) * v.dx(0) * dx,
                   u * v * dx,
                   b2 * u.dx(0) * v * dx]
        A_forms = list(zip(Ax_forms, Ay_forms))
        
        bx_forms = [fx * v * dx for fx in self.fx]
        by_forms = [fy * v * dx for fy in self.fy]
        b_forms = list(zip(bx_forms, by_forms))
        
        A, b = pf.assemble_product_system(A_forms, b_forms, self.bc)
        U = np.linalg.solve(A, b)
        self.stiffness = A
        return U
    
    def solve_adjoint(self, eps, b, u, u_d=None):
        p, v = TrialFunction(self.V), TestFunction(self.V)
        mass = assemble(p * v * dx).array()
        u_d = np.zeros_like(u) if u_d is None else u_d
        resid = -(u - u_d)
        b_adj = resid.T.dot(np.kron(mass, mass))

        A_adj, b_adj = self.bc.apply(self.stiffness.T, b_adj)
        p = np.linalg.solve(A_adj, b_adj).T
        return p
    
    # meat and potatoes
    def assemble_partials(self, eps, b, alpha):
        Wdim = self.W.dim
        dAdm = np.zeros((u_dim, v_dim, m_dim))
        dbdm = np.zeros((b_dim, m_dim))
        dJdm = np.zeros((1, m_dim))
        u, v = TrialFunction(self.V), TestFunction(self.V)
        ...
        
    # can probably inherit this 
    def compute_gradient(self, eps, b, alpha):
        # given m solve for u
        u = self.solve(eps, b)
        
        # given m, u solve for p
        p = self.solve_adjoint(eps, b, u)
        
        # given m, u, p compute ∂A/∂m, ∂b/∂m, and ∂J/∂m
        dAdm, dbdm, dJdm = self.assemble_partials(eps, b, alpha)
        
        # compute gradient = p * (∂b/∂m - ∂A/∂m * u) + ∂J/∂m
        dFdm = dbdm - np.tensordot(dAdm, u, axes=(1,0))
        grad = p.dot(dFdm) + dJdm
        return grad
    