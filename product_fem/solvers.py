import numpy as np
import scipy.sparse as sps
import petsc4py.PETSc as PETSc
from .transforms import to_Function


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
        # krylov solver for b=Au
        ksp = PETSc.KSP().create()
        ksp.setOperators(A)
        
        u = A.createVecRight()
        ksp.solve(b, u)
        return u
    
    def solve(self, A, b):
        if isinstance(A, np.ndarray):
            u = self.dense_solve(A, b)
        elif sps.issparse(A):
            u = self.sparse_solve(A, b)
        elif isinstance(A, PETSc.Mat):
            u = self.petsc_solve(A, b)[:]
            
        return to_Function(u, self.W)