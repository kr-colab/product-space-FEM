from fenics import assemble, as_backend_type
import numpy as np
import scipy.sparse as sps
import petsc4py.PETSc as PETSc


# maybe have FormAssembler and LossAssembler?
class Assembler:
    """Assembler acts on ProductForm objects
    by mapping them to dense or sparse arrays"""
    
    def to_dense_matrix(self, bilinear_form):
        # only use for comparing dense to sparse
        return assemble(bilinear_form).array()
    
    def to_PETSc_matrix(self, bilinear_form):
        matrix = assemble(bilinear_form)
        matrix = as_backend_type(matrix).mat()
        return matrix
    
    def to_sparse_matrix(self, bilinear_form, zero_rows=None):
        matrix = self.to_PETSc_matrix(bilinear_form)
        matrix = sps.csr_matrix(matrix.getValuesCSR()[::-1], shape=matrix.size)
        return matrix 
    
    def to_matrix(self, bilinear_form):
        return self.to_sparse_matrix(bilinear_form)
        
    def to_vector(self, linear_form):
        return assemble(linear_form)[:]
    
    def to_scalar(self, linear_functional):
        return assemble(linear_functional).real
    
    def form_to_array(self, form):
        rank = len(form.arguments())
        if rank==0:
            return self.to_scalar(form)
        elif rank==1:
            return self.to_vector(form)
        elif rank==2:
            return self.to_matrix(form)
                
    def product_form_to_PETSc(self, product_form):
        krons = []
        for i in range(len(product_form)):
            x_form, y_form = product_form[i]
            x = self.form_to_array(x_form)
            y = self.form_to_array(y_form)
            krons.append(sps.kron(x, y))
        M = sum(krons)
        csr = M.indptr, M.indices, M.data
        
        # assign to PETSc matrix
        M = PETSc.Mat().createAIJ(size=M.shape, csr=csr)
        return M
        
    def product_form_to_array(self, product_form, kron):
        krons = []
        for i in range(len(product_form)):
            x_form, y_form = product_form[i]
            x = self.form_to_array(x_form)
            y = self.form_to_array(y_form)
            krons.append(kron(x, y))
        return sum(krons)
    
    def assemble_lhs(self, lhs):
        return self.product_form_to_array(lhs, sps.kron)
        
    def assemble_rhs(self, rhs):
        return self.product_form_to_array(rhs, np.kron)
    
    def assemble_product_system(self, lhs, rhs, bc=None):
        A = self.assemble_lhs(lhs)
        b = self.assemble_rhs(rhs)
        if bc:
            A, b = bc.apply(A, b)
        return A, b
    