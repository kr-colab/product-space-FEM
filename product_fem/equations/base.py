from product_fem.assemblers import Assembler
from product_fem.boundary_conditions import default_boundary_conditions
from product_fem.function_spaces import ProductFunctionSpace
from product_fem.solvers import Solver
from product_fem.forms import derivative, depends_on
import numpy as np
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
        if m is None:
            m = self.control
        else:
            self.update_control(m)
        A, b = self.assemble_system()
        u = self.solver.solve(A, b)
        return u
        
    def derivative_component(self, i, m):
        """Compute the ith component of dF/dm, where F=Au-b.
        dAdm and dbdm"""
        dA_form = derivative(self.lhs, m, m.basis[i])
        db_form = derivative(self.rhs, m, m.basis[i])
        
        dAdm = self.assembler.product_form_to_PETSc(dA_form)
        dbdm = self.assembler.assemble_rhs(db_form)
        
        # derivative(form, m) returns 0 when form is independent of m
        if dbdm==0: 
            W = self.bc.product_function_space
            dbdm = np.zeros(W.dim())
        
        on_boundary = self.bc.get_product_boundary_dofs()
        dAdm.zeroRows(on_boundary, diag=0)
        dAdm = csr_matrix(dAdm.getValuesCSR()[::-1], shape=dAdm.size)
        dbdm[on_boundary] = 0
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