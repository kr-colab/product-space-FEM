import numpy as np
import scipy.sparse as sps
import petsc4py.PETSc as PETSc
from .transforms import to_Function, dense_to_PETSc, PETSc_to_sparse

ksp_error = {3: 'CONVERGED_ATOL',
             9: 'CONVERGED_ATOL_NORMAL',
             6: 'CONVERGED_CG_CONSTRAINED',
             5: 'CONVERGED_CG_NEG_CURVE',
             8: 'CONVERGED_HAPPY_BREAKDOWN',
             0: 'CONVERGED_ITERATING',
             4: 'CONVERGED_ITS',
             2: 'CONVERGED_RTOL',
             1: 'CONVERGED_RTOL_NORMAL',
             7: 'CONVERGED_STEP_LENGTH',
             -5: 'DIVERGED_BREAKDOWN',
             -6: 'DIVERGED_BREAKDOWN_BICG',
             -4: 'DIVERGED_DTOL',
             -10: 'DIVERGED_INDEFINITE_MAT',
             -8: 'DIVERGED_INDEFINITE_PC',
             -3: 'DIVERGED_MAX_IT',
             -9: 'DIVERGED_NANORINF',
             -7: 'DIVERGED_NONSYMMETRIC',
             -2: 'DIVERGED_NULL',
             -11: 'DIVERGED_PCSETUP_FAILED'}

class Solver:
    """Solver acts on linear systems Ax=b by inverting A."""
    
    def __init__(self, W):
        self.W = W
        self.ksp = PETSc.KSP().create()
        
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
#         ksp.setType('ibcgs')
        self.ksp.setOperators(A)
        u = A.createVecRight()
        self.ksp.solve(b, u)
        
        if not self.ksp.converged: 
            reason = ksp_error[self.ksp.getConvergedReason()]
            if reason=='DIVERGED_MAX_IT':
                maxits = self.ksp.getIterationNumber()
                resnorm = self.ksp.getResidualNorm()
                print(f'Krylov solver reached max {maxits} iterations')
                print(f'Residual norm is {resnorm}')
            else:
                raise Exception(f'Krylov solver did not converge, reason: {reason}')
        
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