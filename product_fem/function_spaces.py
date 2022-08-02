from fenics import assemble, Function, TestFunction, TrialFunction, dx, vertex_to_dof_map, dof_to_vertex_map
from itertools import product
import numpy as np
from .functions import ProductFunction, ProductBasisFunction
from .transforms import to_Function, callable_to_ProductFunction

    
class ProductFunctionSpace:
    def __init__(self, V):
        # V is fenics.FunctionSpace
        self.V = V
        self.V_mesh = V.mesh()
        self.dofmap = ProductDofMap(V)
        self.mass = self._compute_mass()
        
    def dofs(self):
        return self.dofmap.product_dofs
#         n = self.V.dim()
#         return [(i,j) for i, j in product(range(n), range(n))]
    
    def tabulate_dof_coordinates(self):
        # marginal_dofs <-dof_coords-> marginal_coords
        # product_dofs <--> marginal_coords 
        V_dofs = self.V.tabulate_dof_coordinates()
        W_dofs = [xy for xy in product(V_dofs, V_dofs)]
        return np.array(W_dofs).squeeze()
    
    def dim(self):
        return self.V.dim()**2
    
    def _marginal_basis_i(self, i):
        phi_i = Function(self.V, name=f'phi_{i}')
        phi_i.vector()[i] = 1.
        return phi_i
    
    def _compute_mass(self):
        x, y = TestFunction(self.V), TestFunction(self.V)
        mass = assemble(x * dx)[:]
        return np.outer(mass, mass)
    
    def product_mass(self):
        x, y = TestFunction(self.V), TrialFunction(self.V)
        pmass = assemble(x * y * dx).array()
        return pmass
    
    def integrate(self, f):
        # integrates f(x,y) over product space
        # int f dxdy = sum int f_ij phi_i phi_j dxdy 
        # int f dxdy = inner(f, M)
        # where M_ij = int phi_i phi_j dxdy
        if isinstance(f, np.ndarray):
            return np.dot(f, self.mass.flatten())
        elif isinstance(f, ProductFunction):
            return np.dot(f.array.flatten(), self.mass.flatten())
        
    def marginal_function_space(self):
        return self.V
    
    def marginal_mesh(self):
        return self.V_mesh
    
    def marginal_basis(self):
        return [self._marginal_basis_i(i) for i in range(self.V.dim())]
    
    def _basis_ij(self, i, j):
        name = f'phi_{i},{j}'
        return ProductBasisFunction(self, i, j, name=name)
    
    def basis(self):
        return [self._basis_ij(i, j) for i, j in self.dofs()]
    
    
# maybe don't need this 
class ProductDofMap:
    # main usage is to obtain bijections between
    # dofs ij <-> product dofs (i,j) (defined by Kronecker product)
    # dofs ij <-> product coordinates (x_i, y_j)
    def __init__(self, function_space):
        # marginal dofs and coordinates
        dofmap = function_space.dofmap()
        dofs = dofmap.dofs(function_space.mesh(), 0)
        dof_coords = function_space.tabulate_dof_coordinates()
        domain_dim = function_space.ufl_domain().geometric_dimension()
        v2d = vertex_to_dof_map(function_space)
        
        # product space dofs ij and coordinates (x_i, y_j)
        self.dofs = [ij for ij in range(len(dofs)**2)] 
        self.product_dofs = [(v2d[i],v2d[j]) for i in dofs for j in dofs] 
        if domain_dim==1:
            xy = [(x.item(),y.item()) for x in dof_coords for y in dof_coords]
        elif domain_dim==2:
            xy = [(x,y) for x in dof_coords for y in dof_coords]
        self.product_dof_coords = xy
        
        # dictionaries for dof/coordinate mapping
        self._dofs_to_product_dofs = dict(zip(self.dofs, self.product_dofs)) # ij -> (i,j)
        self._product_dofs_to_dofs = dict(zip(self.product_dofs, self.dofs)) # (i,j) -> ij
        self._product_dofs_to_dofs = dict(zip(self.product_dofs, self.dofs)) # (i,j) -> ij 
        self._dofs_to_coords = dict(zip(self.dofs, self.product_dof_coords)) # ij -> (x_i, y_j)
        self._product_dofs_to_coords = dict(zip(self.product_dofs, self.product_dof_coords)) # (i,j)->(x_i,y_j)
        
        # save marginal space dofs and coordinates
        self.marginal_function_space = function_space
        self.marginal_dofmap = dofmap
        self.marginal_dofs = dofs 
        self.marginal_dof_coords = dof_coords

    