from fenics import DirichletBC, near
import numpy as np
import scipy.sparse as sps
import petsc4py.PETSc as PETSc
import time

def near2d(x, y, tol=3e-10):
    return np.linalg.norm(x-y) < tol

def default_product_boundary_1d(W, x, y):
    # rule which determines if (x,y) is on the product boundary
    # default product boundary is bdy(M)xM cup Mxbdy(M)
    marginal_bc = DirichletBC(W.marginal_function_space(), 0, 'on_boundary')
    marginal_bc_dofs = marginal_bc.get_boundary_values().keys()
    
    # loop over marginal boundary dofs i
    # determine if x or y near marginal boundary
    for i_bdy in marginal_bc_dofs:
        x_bdy = near(x, W.dofmap.marginal_dof_coords[i_bdy])
        y_bdy = near(y, W.dofmap.marginal_dof_coords[i_bdy])
        on_bdy = x_bdy or y_bdy
        if on_bdy: break
            
    return on_bdy
            
def default_product_boundary_2d(W, x, y):
    # rule which determines if (x,y) is on the product boundary
    # default product boundary is bdy(M)xM cup Mxbdy(M)
    marginal_bc = DirichletBC(W.marginal_function_space(), 0, 'on_boundary')
    marginal_bc_dofs = marginal_bc.get_boundary_values().keys()
    
    # loop over marginal boundary dofs i
    # determine if x or y near marginal boundary
    for i_bdy in marginal_bc_dofs:
        bdy_point = W.dofmap.marginal_dof_coords[i_bdy]
        x_bdy = near2d(x, bdy_point)
        y_bdy = near2d(y, bdy_point)
        on_bdy = x_bdy or y_bdy
        if on_bdy: break
            
    return on_bdy

def default_product_boundary(W, x, y):
    geo_dim = W.V.ufl_domain().geometric_dimension()
    if geo_dim==1:
        return default_product_boundary_1d(W, x, y)
    elif geo_dim==2:
        return default_product_boundary_2d(W, x, y)

def default_boundary_conditions(W):
    return ProductDirichletBC(W, 0., 'default')
    
    
class ProductDirichletBC:
    
    def __init__(self, W, u_bdy, on_product_boundary='default'):
        """Arguments:
        W: ProductFunctionSpace,
        u_bdy: Returns u(x,y) assuming (x,y) is on boundary
        on_product_boundary: True if (x,y) on product boundary, else False
        """
        if on_product_boundary in ['on_boundary', 'default']:
            on_product_boundary = lambda x,y: default_product_boundary(W, x, y)
        if isinstance(u_bdy, (int, float)):
            u_bv = float(u_bdy)
            u_bdy = lambda x,y: u_bv
            
        self.product_function_space = W
        self.marginal_function_space = W.marginal_function_space()
        self.boundary_values = u_bdy
        self.on_boundary = on_product_boundary
        self._product_boundary_dofs = None
        self._product_boundary_coords = None
        
    def function_space(self):
        return self.product_function_space
    
    def get_marginal_boundary_dofs(self):
        bc = DirichletBC(self.marginal_function_space, 0, 'on_boundary')
        return bc.get_boundary_values().keys()

    def _get_product_boundary_dofs(self):
        # dofs ij where either i or j in marginal bdy dofs
        marginal_bdy_dofs = self.get_marginal_boundary_dofs()
        dof_coords = self.product_function_space.tabulate_dof_coordinates()
        product_bdy_dofs = []
        for ij, xy in enumerate(dof_coords):
            if self.on_boundary(*xy):
                product_bdy_dofs.append(ij)
        return product_bdy_dofs

    def _get_product_boundary_coords(self):
        prod_bdy_dofs = self.get_product_boundary_dofs()
        dof_coords = self.product_function_space.tabulate_dof_coordinates()
        product_bdy_coords = [dof_coords[ij] for ij in prod_bdy_dofs]
        return product_bdy_coords
    
    def get_product_boundary_dofs(self):
        if self._product_boundary_dofs is None:
            self._product_boundary_dofs = self._get_product_boundary_dofs()
        
        return self._product_boundary_dofs
    
    def get_product_boundary_coords(self):
        if self._product_boundary_coords is None:
            self._product_boundary_coords = self._get_product_boundary_coords()
        
        return self._product_boundary_coords
        
    def dense_apply(self, A, b):
        # applies desired bdy conds to system AU=b
        # for bdy dof ij, replace A[ij] with e[ij]
        # replace b[ij] with u_bdy(x_i, y_j)
        e = np.eye(A.shape[0])
        prod_bdy_dofs = self.get_product_boundary_dofs() 
        prod_bdy_coords = self.get_product_boundary_coords() 
        bvs = [self.boundary_values(*xy) for xy in prod_bdy_coords]
        for k, ij in enumerate(prod_bdy_dofs):
            A[ij] = e[ij]
            b[ij] = bvs[k]
        return A, b
    
    def sparse_apply(self, A, b):
        # applies desired bdy conds to system AU=b
        # for bdy dof ij, replace A[ij] with e[ij]
        # replace b[ij] with u_bdy(x_i, y_j)
        e = sps.eye(A.shape[0], format='lil')
        A = A.tolil()
        prod_bdy_dofs = self.get_product_boundary_dofs() 
        prod_bdy_coords = self.get_product_boundary_coords() 
        bvs = [self.boundary_values(*xy) for xy in prod_bdy_coords]
        for k, ij in enumerate(prod_bdy_dofs):
            A[ij] = e[ij]
            b[ij] = bvs[k]
        return A.tocsr(), b
    
    def petsc_apply(self, A, b):
        """Apply bc when A and b are PETSc Mat and Vec objects"""
        assert isinstance(A, PETSc.Mat)
        assert isinstance(b, PETSc.Vec)
        prod_bdy_dofs = self.get_product_boundary_dofs()
        prod_bdy_coords = self.get_product_boundary_coords() 
        bvs = [self.boundary_values(*xy) for xy in prod_bdy_coords]
        
        A.zeroRows(prod_bdy_dofs, diag=1)
        b.setValues(prod_bdy_dofs, bvs)
        return A, b
        
    def apply(self, A, b):        
        if sps.issparse(A):
            result = self.sparse_apply(A, b)
        elif isinstance(A, np.ndarray):
            result = self.dense_apply(A, b)
        elif isinstance(A, PETSc.Mat):
            result = self.petsc_apply(A, b)
        
        return result