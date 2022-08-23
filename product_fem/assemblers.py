from fenics import assemble, as_backend_type
import numpy as np
import scipy.sparse as sps
import petsc4py.PETSc as PETSc
from .transforms import PETSc_kron, PETSc_to_sparse


class Assembler:
    """Assembler acts on ProductForm objects
    by mapping them to dense, sparse, or PETSc arrays"""
    
    ## UFL FORM ASSEMBLY
    def _to_dense_vector(self, linear_form):
        return assemble(linear_form)[:]
    
    def _to_PETSc_vector(self, linear_form):
        vector = assemble(linear_form)
        vector = as_backend_type(vector).vec()
        return vector
    
    def _to_dense_matrix(self, bilinear_form):
        # only use for comparing dense to sparse
        return assemble(bilinear_form).array()
    
    def _to_sparse_matrix(self, bilinear_form, zero_rows=None):
        matrix = self._to_PETSc_matrix(bilinear_form)
        matrix = PETSc_to_sparse(matrix)
        return matrix 
    
    def _to_PETSc_matrix(self, bilinear_form):
        matrix = assemble(bilinear_form)
        matrix = as_backend_type(matrix).mat()
        return matrix
    
    ################################################################
    # rank 0 ufl form to scalar
    def to_scalar(self, linear_functional):
        return assemble(linear_functional).real
    
    # rank 1 ufl form to vector
    def to_vector(self, linear_form, out_type='petsc'):
        assert out_type in ['dense', 'sparse', 'petsc']
        if out_type in ['dense', 'sparse']:
            return self._to_dense_vector(linear_form)
        elif out_type=='petsc':
            return self._to_PETSc_vector(linear_form)
    
    # rank 2 ufl form to matrix
    def to_matrix(self, bilinear_form, out_type='petsc'):
        assert out_type in ['dense', 'sparse', 'petsc']
        if out_type=='dense':
            return self._to_dense_matrix(bilinear_form)
        elif out_type=='sparse':
            return self._to_sparse_matrix(bilinear_form)
        elif out_type=='petsc':
            return self._to_PETSc_matrix(bilinear_form)
    
    def form_to_array(self, form, out_type='petsc'):
        rank = len(form.arguments())
        if rank==0:
            return self.to_scalar(form)
        elif rank==1:
            return self.to_vector(form, out_type)
        elif rank==2:
            return self.to_matrix(form, out_type)
    
    def product_form_to_array(self, product_form, out_type='petsc'):
        assert out_type in ['dense', 'sparse', 'petsc']
        
        # which kronecker product to use
        kron_fns = {'dense': np.kron, 
                    'sparse': lambda x,y: sps.kron(x, y, 'csr'), 
                    'petsc': PETSc_kron}
        
        # rank 1 forms will not be sparse
        if product_form.rank()==1:
            kron_fns['sparse'] = np.kron
            
        products = []
        for x_form, y_form in product_form:
            x = self.form_to_array(x_form, out_type)
            y = self.form_to_array(y_form, out_type)
            products.append( kron_fns[out_type](x, y) )
        return sum(products)
    
    ## LINEAR SYSTEM ASSEMBLY
    def assemble_product_system(self, lhs, rhs, bc=None, out_type='petsc'):
        A = self.product_form_to_array(lhs, out_type)
        b = self.product_form_to_array(rhs, out_type)
        
        if bc:
            A, b = bc.apply(A, b)
        return A, b
    