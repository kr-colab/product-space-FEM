import numpy as np
import scipy.sparse as sps
import petsc4py.PETSc as PETSc
from .transforms import to_Function, dense_to_PETSc


class Solver:
    """Solver acts on linear systems Ax=b by inverting A."""
    
    def __init__(self, W):
        self.W = W
        
    def dense_solve(self, A, b):
        u = np.linalg.solve(A, b)
        return u
    
    def sparse_solve(self, A, b):
        u = sps.linalg.spsolve(A, b)
        return u
    
    def petsc_solve(self, A, b):
        if not isinstance(b, PETSc.Vec):
            b = dense_to_PETSc(b)
            
        # krylov solver for Au = b
        ksp = PETSc.KSP().create()
        ksp.setOperators(A)
        
        u = A.createVecRight()
        ksp.solve(b, u)
        return u
    
    def solve(self, A, b):
        if isinstance(A, np.ndarray):
            u = self.dense_solve(A, b)
        elif isinstance(A, sps.csr_matrix):
            u = self.sparse_solve(A, b)
        elif isinstance(A, PETSc.Mat):
            u = self.petsc_solve(A, b)[:]
        
        solution = to_Function(u, self.W)
        return solution