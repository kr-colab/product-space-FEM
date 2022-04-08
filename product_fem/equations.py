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
        
    @abstractmethod
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
    
    def solve_adjoint(self, m, u, u_d=None):
        p, v = TrialFunction(self.V), TestFunction(self.V)
        mass = assemble(p * v * dx).array()
        u_d = np.zeros_like(u) if u_d is None else u_d
        resid = -(u - u_d)
        b_adj = resid.T.dot(np.kron(mass, mass))

        A_adj, b_adj = self.bc.apply(self.stiffness.T, b_adj)
        p = np.linalg.solve(A_adj, b_adj).T
        return p
    
    @abstractmethod
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

    
# -eps Laplacian(u) + b•grad(u) = f
# parameters: eps (float), b=concat(b1, b2), b1 (array), b2 (array)
# assumes drift b = (b_1(x), b_2(y))
class DriftDiffusion:
    def __init__(self, W, f, bc, u_d=None, alt_reg=False):
        self.W = W            # product function space
        self.V = W.V          # marginal function space
        self.fx, self.fy = f  # forcing function
        self.bc = bc          # boundary conditions
        self.data = u_d if u_d is not None else np.zeros(W.dim())
        self.alt_reg = alt_reg
        
    # can accept b as tuple of string representations of (b1(x), b2(y))
    # or b as arrays concat(b1, b2)
    def solve(self, eps, b):
        u, v = TrialFunction(self.V), TestFunction(self.V)
        b1, b2 = self.split_b_to_Function(b)
        
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
        self.rhs = b
        return U
    
    def split_b_to_Function(self, b):
        if isinstance(b[0], str):
            b1, b2 = b
        else:
            i = int(len(b) / 2)
            b1, b2 = np.split(b, [i])
        b1 = pf.to_Function(b1, self.V)
        b2 = pf.to_Function(b2, self.V)
        return b1, b2
            
    def b_str_to_array(self, b):
        # b = [b1, b2] as strings
        b1, b2 = self.split_b_to_Function(b)
        return np.concatenate((b1.vector()[:], b2.vector()[:]))

    def loss_functional(self, eps, b, alpha, beta):
        u = self.solve(eps, b)
        u_d = self.data
        b1, b2 = self.split_b_to_Function(b)
        J = 1/2 * self.W.integrate((u - u_d)**2) 
        if self.alt_reg:
            J += alpha/2 * (assemble(b1.dx(0)**2 * dx) + assemble(b2.dx(0)**2 * dx))
        else:
            J += alpha/2 * (assemble(b1**2 * dx) + assemble(b2**2 * dx)) 
        J += beta/2 * eps**2
        return J
        
    def solve_adjoint(self, u):
        p, v = TrialFunction(self.V), TestFunction(self.V)
        mass = assemble(p * v * dx).array()
        u_d = self.data
        resid = -(u - u_d)
        b_adj = resid.T.dot(np.kron(mass, mass))
#         A_adj, b_adj = self.bc.apply(self.stiffness.T, b_adj.T)
        A_adj, b_adj = self.stiffness.T, b_adj
        p = np.linalg.solve(A_adj, b_adj)
        return p
    
    # meat and potatoes: assemble dbdm, dAdm, dJdm
    def assemble_partials(self, eps, b, alpha, beta):
        b1, b2 = self.split_b_to_Function(b)
        
        V_dim = self.V.dim()
        b_dim = 2 * V_dim
        W_dim = self.W.dim()
        dom = self.V.ufl_domain() # for integrating 1 * dx
        
        # here b is independent of m so dbdm = 0
        dbdb = np.zeros((W_dim, b_dim))

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
        
        bdy_dofs = self.bc.get_product_boundary_dofs()
        for r in range(V_dim):
            phi_r = Function(self.V)
            phi_r.vector()[r] = 1.

            # dAdb1 = (phi_r u' v dx) (u v dy)
            # dAdb2 = (u v dx) (phi_r u' v dy)
            dAdb1_forms = list(zip([phi_r * uu.dx(0) * v * dx], [uu * v * dx]))
            dAdb2_forms = list(zip([uu * v * dx], [phi_r * uu.dx(0) * v * dx]))
            dAdb[:,:,r] = pf.assemble_kron(dAdb1_forms)
            dAdb[:,:,r + V_dim] = pf.assemble_kron(dAdb2_forms)
            dAdb[bdy_dofs] = 0

            if self.alt_reg:
                # dJdb1 = (b1' * phi'_r dx) (1 dy)
                # dJdb2 = (1 dx) (b2' * phi'_r dx)
                dJdb1_forms = list(zip([b1.dx(0) * phi_r.dx(0) * dx], [1 * dx(domain=dom)]))
                dJdb2_forms = list(zip([1 * dx(domain=dom)], [b2.dx(0) * phi_r.dx(0) * dx]))
                dJdb[0, r] = alpha * pf.assemble_kron(dJdb1_forms)
                dJdb[0, r + V_dim] = alpha * pf.assemble_kron(dJdb2_forms)
            else:
                # dJdb1 = (b1 * phi_r dx) (1 dy)
                # dJdb2 = (1 dx) (b2 * phi_r dx)
                dJdb1_forms = list(zip([b1 * phi_r * dx], [1 * dx(domain=dom)]))
                dJdb2_forms = list(zip([1 * dx(domain=dom)], [b2 * phi_r * dx]))
                dJdb[0,r] = alpha * pf.assemble_kron(dJdb1_forms)
                dJdb[0,r + V_dim] = alpha * pf.assemble_kron(dJdb2_forms)
            
        return dAdb, dbdb, dJdb
    
    def compute_gradient(self, eps, b, alpha, beta, loss=False):
        # given m solve for u
        u = self.solve(eps, b)
        
        # given m, u solve for p
        p = self.solve_adjoint(u)
        
        # given m, u, p compute ∂A/∂m, ∂b/∂m, and ∂J/∂m
        dAdm, dbdm, dJdm = self.assemble_partials(eps, b, alpha, beta)
        
        # compute gradient = p * (∂b/∂m - ∂A/∂m * u) + ∂J/∂m
        dFdm = dbdm - np.tensordot(dAdm, u, axes=(1,0))
        grad = -p.dot(dFdm) + dJdm
        
        if loss:
            J = self.loss_functional(eps, b, alpha, beta)
            return J, grad
        else:
            return grad
    
    
# sig^2/2 Laplacian(u) + mu•grad(u) = f
# parameters: sig (float), mu (array)
class HittingTimes:
    def __init__(self, W, f, bc, u_d=None, m_true=None):
        self.W = W            # product function space
        self.V = W.V          # marginal function space
        self.fx, self.fy = f  # forcing function
        self.bc = bc          # boundary conditions
        
        # get data directly (u_d) or implicitly (m_true) or null
        if u_d is not None:
            self.data = u_d
        elif m_true is not None:
            self.m_true = m_true
            self.data = self.solve(*m_true)
        else:
            self.data = np.zeros(W.dim())
        
    # for now sig is assumed constant
    def solve(self, mu, sig):
        u, v = TrialFunction(self.V), TestFunction(self.V)
        mu = pf.to_Function(mu, self.V)
        sig = pf.to_Function(sig, self.V)
        
        fxs = [pf.to_Function(fx, self.V) for fx in self.fx]
        fys = [pf.to_Function(fy, self.V) for fy in self.fy]
        
        Ax_forms = [-0.5 * u.dx(0) * Dx(sig**2 * v, 0) * dx,
                    u * v * dx,
                    mu * u.dx(0) * v * dx,
                    u * v * dx]
        Ay_forms = [u * v * dx,
                    -0.5 * u.dx(0) * Dx(sig**2 * v,0) * dx,
                    u * v * dx,
                    mu * u.dx(0) * v * dx]
        A_forms = list(zip(Ax_forms, Ay_forms))
        
        bx_forms = [fx * v * dx for fx in fxs]
        by_forms = [fy * v * dx for fy in fys]
        b_forms = list(zip(bx_forms, by_forms))
        
        A, b = pf.assemble_product_system(A_forms, b_forms, self.bc)
        u_h = np.linalg.solve(A, b)
        self.stiffness = A
        self.rhs = b
        return u_h
    
    def mu_str_to_array(self, mu):
        mu = pf.to_Function(mu, self.V) # string to Function
        mu = mu.vector()[:] # Function to array
        return mu

    def get_reg_params(self, alpha, beta):
        if isinstance(alpha, (list, tuple)):
            alpha1, alpha2 = alpha
        elif isinstance(alpha, float):
            alpha1 = alpha2 = alpha
            
        if isinstance(beta, (list, tuple)):
            beta1, beta2 = beta
        elif isinstance(beta, float):
            beta1 = beta2 = beta
            
        return alpha1, alpha2, beta1, beta2
    
    def loss_functional(self, mu, sig, alpha, beta):
        # square error term
        u = self.solve(mu, sig)
        u_d = self.data
        J = 1/2 * self.W.integrate((u - u_d)**2) 
        
        # regularization terms
        mu = pf.to_Function(mu, self.V)
        sig = pf.to_Function(sig, self.V)
        alpha1, alpha2, beta1, beta2 = self.get_reg_params(alpha, beta)
        J += alpha1 / 2 * assemble(mu.dx(0)**2 * dx) # smoothing mu
        J += alpha2 / 2 * assemble(mu**2 * dx) # sparsing mu
        J += beta1 / 2 * assemble(sig.dx(0)**2 * dx) # smoothing sig
        J += beta2 / 2 * assemble(sig**2 * dx) # sparsing sig
        return J
        
    def solve_adjoint(self, u):
        p, v = TrialFunction(self.V), TestFunction(self.V)
        mass = assemble(p * v * dx).array()
        u_d = self.data
        resid = -(u - u_d)
        b_adj = resid.T.dot(np.kron(mass, mass))
        A_adj, b_adj = self.stiffness.T, b_adj
        p = np.linalg.solve(A_adj, b_adj)
        return p
    
    # meat and potatoes: assemble dbdm, dAdm, dJdm
    def assemble_partials(self, mu, sig, alpha, beta):
        # tensor dimensions
        W_dim = self.W.dim()
        mu_dim = len(mu)
        sig_dim = len(sig)
        dom = self.V.ufl_domain() # for integrating 1 * dx
        
        # parameters
        mu = pf.to_Function(mu, self.V)
        sig = pf.to_Function(sig, self.V)

        # compute gradients of A, b, and J wrt mu, sig
        # here b is independent of mu, sig so dbdm = dbds = 0
        dAdm = np.zeros((W_dim, W_dim, mu_dim))
        dAds = np.zeros((W_dim, W_dim, sig_dim))
        dbdm = np.zeros((W_dim, mu_dim))
        dbds = np.zeros((W_dim, sig_dim))
        dJdm = np.zeros((1, mu_dim))
        dJds = np.zeros((1, sig_dim))
        
        uu, v = TrialFunction(self.V), TestFunction(self.V)
        bdy_dofs = self.bc.get_product_boundary_dofs()
        alpha1, alpha2, beta1, beta2 = self.get_reg_params(alpha, beta)
        for r in range(mu_dim):
            phi_r = Function(self.V)
            phi_r.vector()[r] = 1.

            # dAdm = (phi_r u' v dx) (u v dy) + (u v dx) (phi_r u' v dy)
            dAdm_forms = list(zip([phi_r * uu.dx(0) * v * dx, uu * v * dx], 
                                  [uu * v * dx, phi_r * uu.dx(0) * v * dx]))
            dAdm[:,:,r] = pf.assemble_kron(dAdm_forms)
            dAdm[bdy_dofs] = 0
            
            # dAds = -(u' (v sig phi_r)' dx) (u v dy) - (u v dx) (u' (v sig phi_r)' dy)
            dAds_forms = list(zip([uu.dx(0) * Dx(v * sig * phi_r, 0) * dx, uu * v * dx], 
                                  [uu * v * dx, uu.dx(0) * Dx(v * sig * phi_r, 0) * dx]))
            dAds[:,:,r] = -pf.assemble_kron(dAds_forms)
            dAds[bdy_dofs] = 0

            # dJdm has smoothing and sparsity regularization
            # dJdm = alpha1{(mu' phi'_r dx) (1 dy) + (1 dx) (mu' phi'_r dy)}
            #         + alpha2{(mu phi_r dx) (1 dy) + (1 dx) (mu phi_r dy)}
            
            # mu smoothing term
            x_forms1 = [mu.dx(0) * phi_r.dx(0) * dx, 1 * dx(domain=dom)]
            y_forms1 = [1 * dx(domain=dom), mu.dx(0) * phi_r.dx(0) * dx]
            dJdm_forms1 = list(zip(x_forms1, y_forms1))
            dJdm[0, r] = alpha1 * pf.assemble_kron(dJdm_forms1)
            
            # mu decay term
            x_forms2 = [mu * phi_r * dx, 1 * dx(domain=dom)]
            y_forms2 = [1 * dx(domain=dom), mu * phi_r * dx]
            dJdm_forms2 = list(zip(x_forms2, y_forms2))
            dJdm[0, r] += alpha2 * pf.assemble_kron(dJdm_forms2)
            
            # dJds has smoothing and sparsity regularization
            # dJds = beta1{(sig' phi'_r dx) (1 dy) + (1 dx) (sig' phi'_r dy)}
            #         + beta2{(sig phi_r dx) (1 dy) + (1 dx) (sig phi_r dy)}}
            
            # sigma smoothing term
            x_forms1 = [sig.dx(0) * phi_r.dx(0) * dx, 1 * dx(domain=dom)]
            y_forms1 = [1 * dx(domain=dom), sig.dx(0) * phi_r.dx(0) * dx]
            dJds_forms1 = list(zip(x_forms1, y_forms1))
            dJds[0, r] = beta1 * pf.assemble_kron(dJds_forms1)
            
            # sigma decay terms
            x_forms2 = [sig * phi_r * dx, 1 * dx(domain=dom)]
            y_forms2 = [1 * dx(domain=dom), sig * phi_r * dx]
            dJds_forms2 = list(zip(x_forms2, y_forms2))
            dJds[0, r] += beta2 * pf.assemble_kron(dJds_forms2)

        return (dAdm, dAds), (dbdm, dbds), (dJdm, dJds)
    
    def compute_gradient(self, mu, sig, alpha, beta):
        # given m solve for u
        u = self.solve(mu, sig)
        
        # given m, u solve for p
        p = self.solve_adjoint(u)
        
        # given m, u, p compute ∂A/∂m, ∂b/∂m, and ∂J/∂m
        dA, db, dJ = self.assemble_partials(mu, sig, alpha, beta)
        dAdm, dAds = dA
        dbdm, dbds = db
        dJdm, dJds = dJ
        
        # compute mu gradient = p * (∂b/∂m - ∂A/∂m * u) + ∂J/∂m
        dFdm = dbdm - np.tensordot(dAdm, u, axes=(1,0))
        mu_grad = -p.dot(dFdm) + dJdm
        
        # compute sig gradient = p * (∂b/∂s - ∂A/∂s * u) + ∂J/∂s
        dFds = dbds - np.tensordot(dAds, u, axes=(1,0))
        sig_grad = -p.dot(dFds) + dJds
        
        return mu_grad, sig_grad
        
    def loss_and_grad(self, mu_sig, alpha, beta):
        mu, sig = np.split(mu_sig, [self.V.dim()])
        loss = self.loss_functional(mu, sig, alpha, beta)
        mu_grad, sig_grad = self.compute_gradient(mu, sig, alpha, beta)
        grad = np.concatenate((mu_grad[0], sig_grad[0]))
        return loss, grad