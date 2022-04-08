import numpy as np
import scipy.sparse as sps
from fenics import *


# CONVERTERS
# from strings
def string_to_Function(string, V, proj=False):
    if proj:
        return project(Expression(string, element=V.ufl_element()), V)
    else:
        return interpolate(Expression(string, element=V.ufl_element()), V)
    
def string_to_array(string, V, proj=False):
    func = string_to_Function(string, V, proj)
    return func.vector()[:]
    
# from python functions
def pyfunc_to_Function(pyfunc, V):
    pyfunc_array = pyfunc_to_array(pyfunc, V)
    return array_to_Function(pyfunc_array, V)

def pyfunc_to_array(pyfunc, V):
    dof_coords = V.tabulate_dof_coordinates()
    return np.array([pyfunc(*x) for x in dof_coords])

# from dolfin Functions
def Function_to_array(func):
    return func.vector()[:]

# from numpy arrays
def array_to_Function(array, V):
    f = Function(V)
    f.vector()[:] = array.copy()
    return f

# from ufl forms
def form_to_array(form):
    array = assemble(form)
    rank = len(form.arguments())
    if rank==0:
        return array.real
    elif rank==1:
        return array[:]
    elif rank==2:
        return array.array()
    
# to dolfin Functions
def to_Function(func, V):
    if isinstance(func, str):
        return string_to_Function(func, V)
    elif isinstance(func, np.ndarray):
        return array_to_Function(func, V)
    elif callable(func):
        return pyfunc_to_Function(func, V)
    
# to numpy arrays
def to_array(func, V):
    if isinstance(func, str):
        return string_to_array(func, V)
    elif isinstance(func, Function):
        return Function_to_array(func)
    elif callable(func):
        return pyfunc_to_array(func, V)
    
# ASSEMBLERS
# assemble sum_i kron(Ax_i, Ay_i)
def assemble_kron(forms):
    # forms = [(Ax_1, Ay_1), ..., (Ax_n, Ay_n)]
    krons = []
    for Ax_forms, Ay_forms in forms:
        Ax_arr = form_to_array(Ax_forms)
        Ay_arr = form_to_array(Ay_forms)
        krons.append(np.kron(Ax_arr, Ay_arr))
    A = sum(krons)
    return A
        
def assemble_product_system(A_forms, b_forms, bc=None):
    # assembles linear system Au=b
    A, b = assemble_kron(A_forms), assemble_kron(b_forms)
    if bc:
        A, b = bc.apply(A, b)
    return A, b

# product boundary
def default_product_boundary(W, x, y):
    # rule which determines if (x,y) is on the product boundary
    # default product boundary is bdy(M)xM cup Mxbdy(M)
    marginal_bc = DirichletBC(W.marginal_function_space(), 0, 'on_boundary')
    marginal_bc_dofs = marginal_bc.get_boundary_values().keys()
    
    # loop over marginal boundary dofs i
    # determine if x or y near marginal boundary
    for i_bdy in marginal_bc_dofs:
        # this probably doesn't work with 2d mesh
        x_bdy = near(x, W.dofmap.marginal_dof_coords[i_bdy])
        y_bdy = near(y, W.dofmap.marginal_dof_coords[i_bdy])
        on_bdy = x_bdy or y_bdy
        if on_bdy: break
            
    return on_bdy
            
    
class ProductDirichletBC:
    def __init__(self, W, u_bdy, on_product_boundary='default'):
        # on_product_boundary: (x,y) -> True if (x,y) on product boundary, else False
        # u_bdy is a map (x,y) -> u(x,y) given that on_bound(x,y)==True
        # W is ProductFunctionSpace, can use .dofmap to help with on_bound
        if on_product_boundary in ['on_boundary', 'default']:
            on_product_boundary = lambda x,y: default_product_boundary(W, x, y)
        if isinstance(u_bdy, (int, float)):
            u_bv = float(u_bdy)
            u_bdy = lambda x,y: u_bv
            
        self.product_function_space = W
        self.marginal_function_space = W.marginal_function_space()
        self.boundary_values = u_bdy
        self.on_boundary = on_product_boundary
            
    def get_marginal_boundary_dofs(self):
        bc = DirichletBC(self.marginal_function_space, 0, 'on_boundary')
        return bc.get_boundary_values().keys()

    def get_product_boundary_dofs(self):
        # dofs ij where either i or j in marginal bdy dofs
        marginal_bdy_dofs = self.get_marginal_boundary_dofs()
        dof_coords = self.product_function_space.dofmap._dofs_to_coords # ij->(x_i,y_j)
        product_bdy_dofs = []
        for ij, xy in dof_coords.items():
            if self.on_boundary(*xy):
                product_bdy_dofs.append(ij)
        return product_bdy_dofs

    def get_product_boundary_coords(self):
        prod_bdy_dofs = self.get_product_boundary_dofs()
        dof_to_coords = self.product_function_space.dofmap._dofs_to_coords # ij->(x_i,y_j)
        product_bdy_coords = [dof_to_coords[ij] for ij in prod_bdy_dofs]
        return product_bdy_coords
            
    def apply(self, A, b):
        # applies desired bdy conds to system AU=b
        # for bdy dof ij, replace A[ij] with e[ij]
        # replace b[ij] with u_bdy(x_i, y_j)
        e = np.eye(len(A))
        prod_bdy_dofs = self.get_product_boundary_dofs() # ij on boundary
        prod_bdy_coords = self.get_product_boundary_coords() # (x_i, y_j) on boundary
        bvs = [self.boundary_values(*xy) for xy in prod_bdy_coords]
        for k, ij in enumerate(prod_bdy_dofs):
            A[ij] = e[ij]
            b[ij] = bvs[k]
        return A, b
    
    
class ProductFunctionSpace:
    def __init__(self, V):
        # V is fenics.FunctionSpace
        self.V = V
        self.V_mesh = V.mesh()
        self.dofmap = ProductDofMap(V)
        self.mass = self._compute_mass()
        
    def dofs(self):
        # need to do bijections that can be
        # restricted to the boundary
        # product_dofs <-kron-> marginal_dofs
        return self.dofmap._dofs_to_product_dofs
    
    def tabulate_dof_coordinates(self):
        # marginal_dofs <-dof_coords-> marginal_coords
        # product_dofs <--> marginal_coords 
        return list(self.dofmap._product_dofs_to_coords.values())
    
    def _compute_mass(self):
        v = TestFunction(self.V)
        mass = assemble(v * dx)[:]
        mass = np.kron(mass, mass)
        return mass
    
    def integrate(self, f):
        # integrates f(x,y) over product space
        return np.dot(f, self.mass)
        
    def marginal_function_space(self):
        return self.V
    
    def marginal_mesh(self):
        return self.V_mesh
    
    def dim(self):
        return self.V.dim()**2


class ProductDofMap:
    # main usage is to obtain bijections between
    # dofs ij <-> product dofs (i,j) (defined by Kronecker product)
    # dofs ij <-> product coordinates (x_i, y_j)
    def __init__(self, function_space):
        # marginal dofs and coordinates
        dofmap = function_space.dofmap()
        dofs = dofmap.dofs(function_space.mesh(), 0)
        dof_coords = function_space.tabulate_dof_coordinates()
        
        # product space dofs ij and coordinates (x_i, y_j)
        self.dofs = [ij for ij in range(len(dofs)**2)] # sets stiffness sparsity pattern
        self.product_dofs = [(i,j) for i in dofs for j in dofs] 
        self.product_dof_coords = [(x.item(),y.item()) for x in dof_coords for y in dof_coords]
        
        # dictionaries for dof/coordinate mapping
        self._dofs_to_product_dofs = dict(zip(self.dofs, self.product_dofs)) # ij -> (i,j)
        self._product_dofs_to_dofs = dict(zip(self.product_dofs, self.dofs)) # (i,j) -> ij 
        self._dofs_to_coords = dict(zip(self.dofs, self.product_dof_coords)) # ij -> (x_i, y_j)
        self._product_dofs_to_coords = dict(zip(self.product_dofs, self.product_dof_coords)) # (i,j)->(x_i,y_j)
        
        # save marginal space dofs and coordinates
        self.marginal_function_space = function_space
        self.marginal_dofmap = dofmap
        self.marginal_dofs = dofs 
        self.marginal_dof_coords = dof_coords

    
# TODO: inherit from np ndarray? 
class ProductFunction:
    def __init__(self, W):
        # initializes product space function 
        # f(x,y) = sum_ij f_ij phi_i(x)phi_j(y)
        # where f_ij = f(x_i, y_j)
        # by default f_ij=0 for all ij
        self.W = W
        n_dofs = len(W.dofmap.dofs)
        self.array = np.zeros(n_dofs)
        
    def assign(self, f):
        # assigns values in array f to product function f(x,y)
        # i.e. f contains f_ij
        f_array = self.array
        dof_to_coords = self.W.dofmap._dofs_to_coords
        for dof, xy in dof_to_coords.items():
            f_array[dof] = f(*xy) # when f is a python function of (x,y)
        self.array = f_array
        
    def plot(self, marginal_slice):
        ...