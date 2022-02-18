import numpy as np
import scipy.sparse as sps
from fenics import *

## TODO
### write tests
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
            
def form_to_array(form):
    rank = len(form.arguments())
    arr = assemble(form)
    if rank==0:
        array = arr.real
    elif rank==1:
        array = arr[:]
    elif rank==2:
        array = arr.array()
    return array

# assemble sum_i kron(Ax_i, Ay_i)
def assemble_kron(forms):
    # forms = [(Ax_1, Ay_1), ..., (Ax_n, Ay_n)]
    krons = []
    for BC in forms:
        Ax_forms, Ay_forms = BC
        B_i = form_to_array(Ax_forms)
        C_i = form_to_array(Ay_forms)
        krons.append(np.kron(B_i, C_i))
    A = sum(krons)
        
    return A
        
def assemble_product(A_forms, b_forms):
    # assembles linear system AU=b where
    # A = sum_i kron(B_i,C_i)
    # b = sum_i kron(c_i,d_i)

    # LHS & RHS assembly
    A = assemble_kron(A_forms)
    b = assemble_kron(b_forms)

    return A, b

def assemble_product_system(A_forms, b_forms, bc=None):
    # assemble forms
    A, b = assemble_product(A_forms, b_forms)
    if bc is not None:
        A, b = bc.apply(A, b)
    return A, b
    

class ProductDofMap:
    # main usage is to obtain bijections between
    # dofs ij <-> product dofs (i,j) (defined by Kronecker product)
    # dofs ij <-> product coordinates (x_i, y_j)
    def __init__(self, function_space):
        # marginal dofs and coordinates
        dofmap = function_space.dofmap()
        dofs = dofmap.dofs()
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
        
        
class ProductFunctionSpace:
    def __init__(self, V):
        # V is fenics.FunctionSpace
        self.V = V
        self.V_mesh = V.ufl_domain()
        self.dofmap = ProductDofMap(V)
        self.mass = self._compute_mass()
        self.dim = V.dim()**2
        
    def dofs(self):
        # need to do bijections that can be
        # restricted to the boundary
        # product_dofs <-kron-> marginal_dofs
        return self.dofmap._dofs_to_product_dofs
    
    def tabulate_dof_coordinates(self):
        # marginal_dofs <-dof_coords-> marginal_coords
        # product_dofs <--> marginal_coords 
        # ^^factors through the previous 2 bijections
        return self.dofmap._dofs_to_coords
    
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
        
        
class ProductDirichletBC:
    def __init__(self, W, u_bdy, on_product_boundary='default'):
        # on_product_boundary: (x,y) -> True if (x,y) on product boundary, else False
        # u_bdy is a map (x,y) -> u(x,y) given that on_bound(x,y)==True
        # W is ProductFunctionSpace, can use .dofmap to help with on_bound
        if on_product_boundary in ['on_boundary', 'default']:
            on_product_boundary = default_product_boundary
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
        dofs = self.product_function_space.dofmap._dofs_to_product_dofs # ij->(i,j)
        product_bdy_dofs = []
        for ij, ij_pair in dofs.items():
            i, j = ij_pair
            if i in marginal_bdy_dofs or j in marginal_bdy_dofs:
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
        bvs = [self.boundary_values(xy[0], xy[1]) for xy in prod_bdy_coords]
        for k, ij in enumerate(prod_bdy_dofs):
            A[ij] = e[ij]
            b[ij] = bvs[k]
        return A, b
    
    