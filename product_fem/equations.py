from product_fem import ProductDirichletBC, ProductForm, Function, Control, to_Function
from fenics import as_matrix, TrialFunction, TestFunction, dx, Dx, inner, grad, div, exp
from .assemblers import Assembler
from .boundary_conditions import default_boundary_conditions
from .function_spaces import ProductFunctionSpace
from .solvers import Solver
from .forms import derivative, depends_on
from .transforms import dense_to_PETSc
import numpy as np
import petsc4py.PETSc as PETSc
from scipy.sparse import csr_matrix


class Equation:
    
    def __init__(self, lhs, rhs, control, bc=None):
        self.lhs = lhs
        self.rhs = rhs
        self.control = control
        
        assert depends_on(lhs, control) or depends_on(rhs, control)
        
        if bc is None:
            W = ProductFunctionSpace(lhs.function_space())
            self.bc = default_boundary_conditions(W)
        else:
            W = bc.function_space()
            self.bc = bc
            
        self.assembler = Assembler()
        self.solver = Solver(W)
        
    def function_space(self):
        return self.bc.product_function_space
    
    def update_control(self, m):
        self.control.update(m)
        
    def assemble_system(self):
        assemble = self.assembler.assemble_product_system
        A, b = assemble(self.lhs, self.rhs, self.bc)
        self.A, self.b = A, b
        return A, b
    
    def solve(self, m=None):
        if m is not None:
            self.update_control(m)
        A, b = self.assemble_system()
        u = self.solver.solve(A, b)
        return u
        
    def derivative_component(self, i, m):
        """Compute the ith component of dF/dm, where F=Au-b.
        dAdm and dbdm"""
        dA_form = derivative(self.lhs, m, m.basis[i])
        db_form = derivative(self.rhs, m, m.basis[i])
        
        dAdm = self.assembler.product_form_to_array(dA_form, out_type='petsc')
        # derivative(form, m) returns 0 when form is independent of m
        if db_form==0: 
            W = self.bc.product_function_space
            dbdm = np.zeros(W.dim())
            if isinstance(dAdm, PETSc.Mat):
                dbdm = dense_to_PETSc(dbdm)
        else:
            dbdm = self.assembler.product_form_to_array(db_form, out_type='petsc')
            
        # rows with boundary dofs are fixed, so dAdm=0 there
        on_boundary = self.bc.get_product_boundary_dofs()
        dAdm.zeroRows(on_boundary, diag=0)
        dbdm.setValues(on_boundary, np.zeros(len(on_boundary)))
#         dAdm = csr_matrix(dAdm.getValuesCSR()[::-1], shape=dAdm.size)
#         dbdm[on_boundary] = 0
        return dAdm, dbdm
        
    def derivative(self, control):
        dAdm, dbdm = [], []
        for m in control:
            for i in range(m.dim()):
                dA_i, db_i = self.derivative_component(i, m)
                dAdm.append(dA_i)
                dbdm.append(db_i)
                
        # could numpy transform (dA/dm)_ijk = dAdm[k] and (db/dm)_ij = dbdm[j]
        return dAdm, dbdm
        
    def set_boundary_conditions(self, bc):
        self.bc = bc
        
        
# sig^2/2 Laplacian(u) + mu•grad(u) = f
# parameters: sig (float), mu (array)
class HittingTimes1D(Equation):
    def __init__(self, W, u_bdy=1., epsilon=1e-2):
        mu, sig = self._init_control(W)
        u, v = TrialFunction(W.V), TestFunction(W.V)
        
        # left hand side forms 
        Ax_forms = [-0.5 * u.dx(0) * Dx(sig**2 * v, 0) * dx,
                    u * v * dx,
                    mu * u.dx(0) * v * dx,
                    u * v * dx]
        Ay_forms = [u * v * dx,
                    -0.5 * u.dx(0) * Dx(sig**2 * v,0) * dx,
                    u * v * dx,
                    mu * u.dx(0) * v * dx]
        
        # right hand side forms 
        bx_forms = [-1. * v * dx]
        by_forms = [1. * v * dx]
        
        # product forms and control
        lhs = ProductForm(Ax_forms, Ay_forms)
        rhs = ProductForm(bx_forms, by_forms)
        control = Control([mu, sig])
        
        # boundary is epsilon nbhd around diagonal x=y
        on_product_boundary = lambda x, y: np.linalg.norm(x - y) <= epsilon
        bc = ProductDirichletBC(W, u_bdy, on_product_boundary)
        
        super().__init__(lhs, rhs, control, bc)
        
    def _init_control(self, W):
        # default control here is mu(x)=0 and sig(x)=0.25
        mu = Function(W.V, name='mu')
        sig = Function(W.V, name='sigma')
        sig.vector()[:] = 0.25
        return mu, sig
        
        
# sig^2/2 Laplacian(u) + mu•grad(u) = -1
# parameters: mu (mean vector), sig (covariance matrix)
class HittingTimes2D(Equation):
    
    def __init__(self, W, u_bdy=1., epsilon=1e-2):
        mu, sig = self._init_control(W)
        # to enforce SPD on sigma we use a log-Cholesky factorization
        L = as_matrix([[exp(sig[0]), sig[2]],[0, exp(sig[1])]])
        sigma = L.T * L
        
        u, v = TrialFunction(W.V), TestFunction(W.V)
        
        # left hand side forms
        Ax_forms = [-0.5 * inner(grad(u), div(sigma * v)) * dx,
                    u * v * dx,
                    inner(mu, grad(u)) * v * dx,
                    u * v * dx]
        Ay_forms = [u * v * dx,
                    -0.5 * inner(grad(u), div(sigma * v)) * dx,
                    u * v * dx,
                    inner(mu, grad(u)) * v * dx]
        
        # right hand side forms
        bx_forms = [1. * v * dx]
        by_forms = [-1. * v * dx]
        
        # product forms and control
        lhs = ProductForm(Ax_forms, Ay_forms)
        rhs = ProductForm(bx_forms, by_forms)
        control = Control([mu, sig])
        
        # boundary is epsilon nbhd around diagonal x=y
        on_product_boundary = lambda x, y: np.linalg.norm(x - y) <= epsilon
        bc = ProductDirichletBC(W, u_bdy, on_product_boundary)
        
        super().__init__(lhs, rhs, control, bc)
    
    def _init_control(self, W):
        mu = Function(W.V, dim=2, name='mu')
        sig = Function(W.V, dim=3, name='sig')
        return mu, sig
        

def HittingTimes(W, u_bdy, epsilon):
    """
    TODO: change this CoalescenceTimes
    
    The equation class for the system of equations (TODO: define $L$ and use this instead)
    .. math::
    
        \frac{\sigma(x)^2}{2} \Delta_x u(x, y) + \mu(x) \cdot \nabla_x u(x, y) &= -1 \\
        \frac{\sigma(y)^2}{2} \Delta_y u(x, y) + \mu(y) \cdot \nabla_y u(x, y) &= -1
    
    if both $x$ and $y$ are in $\Omega$,
    a domain in $\mathbb{R}$ or $\mathbb{R}^2$
    specified by ``W``, and boundary conditions
    .. math::
        \text{ and } u(x, y) = u_b(x, y) \text{ if } |x - y| \le \epsilon
        
    where $u$ is a function on $\Omega \times \Omega$,
    $\mu$ is a function on $\Omega$,
    and $u_b$ is a function on $\Omega \times \Omega$.
    Furthermore, there are reflecting (i.e., Dirichlet)
    boundary conditions on $\delta \Omega \times \Omega \cup \Omega \times \delta \Omega$.
    
    The parameters are {math}`\sigma \in \mathbb{R}_+$ and $\mu : [0,1] \to \mathbb{R}`,
    and so the ``control`` of this equation is of the form ``mu, sigma``.
    The default {math}`\mu` is {math}`\mu(x) = 0`.
    In one dimension, sigma is a single number and defaults to {math}`\sigma = 1/4`.
    In two dimensions, the operator (TODO FIXUP) is
    
    .. math::
    
        \sum_{ij} \Sigma_{ij} \partial_{x_i} \partial_{x_j} u(x, y)
    
    :param ProductFunctionSpace W: The product function space in which $u$ lives.
    :param u_bdy: The values for $u$ along the "diagonal"; either a float or a callable.
    :param float epsilon: The width of the "diagonal" boundary affected by ``u_bdy``.
    """
    gdim = W.V.ufl_domain().geometric_dimension()
    if gdim==1:
        return HittingTimes1D(W, u_bdy, epsilon)
    elif gdim==2:
        return HittingTimes2D(W, u_bdy, epsilon)   
    
    
# -eps Laplacian(u) + b•grad(u) = f
# parameters: eps (float), b=concat(b1, b2), b1 (array), b2 (array)
# assumes drift b = (b_1(x), b_2(y))
class DriftDiffusion(Equation):
    def __init__(self, W, f, bc):
        b, eps = self._init_control(W)
        u, v = TrialFunction(W.V), TestFunction(W.V)
        b1, b2 = b.split()
        
        fxs = [to_Function(fx, W.V) for fx in f[0]]
        fys = [to_Function(fy, W.V) for fy in f[1]]
        
        Ax_forms = [eps * u.dx(0) * v.dx(0) * dx,
                   u * v * dx,
                   b1 * u.dx(0) * v * dx,
                   u * v * dx]
        Ay_forms = [u * v * dx,
                   eps * u.dx(0) * v.dx(0) * dx,
                   u * v * dx,
                   b2 * u.dx(0) * v * dx]
        
        bx_forms = [fx * v * dx for fx in fxs]
        by_forms = [fy * v * dx for fy in fys]

        lhs = ProductForm(Ax_forms, Ay_forms)
        rhs = ProductForm(bx_forms, by_forms)
        control = Control([b, eps])
        
        super().__init__(lhs, rhs, control, bc)
        
    def _init_control(self, W):
        # default control is b=0
        b = Function(W.V, dim=2, name='b')
        eps = Function(W.V, name='eps')
        eps.vector()[:] = 0.25
        return b, eps
    
    
# -grad•(exp(m) grad(u)) = f 
# parameters: m (array)
class ExpDiffusion(Equation):
    
    def __init__(self, W, f, bc=None):
        # forcing function is f(x,y) = f1(x) * f2(y)
        # default control is m(x,y) = m1(x) + m2(y)
        mx, my = self.init_control(W)
        u, v = TrialFunction(W.V), TestFunction(W.V)
        
        # left hand side forms 
        Ax_forms = [exp(mx) * u.dx(0) * v.dx(0) * dx,
                    exp(mx) * u * v * dx]
        Ay_forms = [exp(my) * u * v * dx,
                    exp(my) * u.dx(0) * v.dx(0) * dx]
        
        
        # right hand side forms 
        fx, fy = to_Function(f[0], W.V), to_Function(f[1], W.V)
        bx_forms = [fx * v * dx]
        by_forms = [fy * v * dx]
        
        self.control = Control([mx, my])
        lhs = ProductForm(Ax_forms, Ay_forms)
        rhs = ProductForm(bx_forms, by_forms)
        
        if bc is None:
            bc = ProductDirichletBC(W, 0, 'on_boundary')
        super().__init__(lhs, rhs, bc)
        
    def init_control(self, W):
        m = to_Function('cos(x[0])', W.V)
        mx = Function(W.V)
        mx.assign(m)
        
        m = to_Function('sin(x[0])', W.V)
        my = Function(W.V)
        my.assign(m)
        
        return mx, my
    
    
class Poisson(Equation):
    
    def __init__(self, W, bc=None):
        u, v = TrialFunction(W.V), TestFunction(W.V)
        
        Ax_forms = [u.dx(0) * v.dx(0) * dx, u * v * dx]
        Ay_forms = [u * v * dx, u.dx(0) * v.dx(0) * dx]
        lhs = ProductForm(Ax_forms, Ay_forms)
        
        fx, fy = f
        rhs = ProductForm(fx * v * dx, fy * v * dx)
        control = Control([])
        super().__init__(lhs, rhs, bc)