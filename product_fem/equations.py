import numpy as np
import product_fem as pf
from fenics import *


class Poisson:
    def __init__(self, W, f, bc):
        self.W = W
        self.fx, self.fy = f
        self.bc = bc
        
    def solve(self):
        V = self.W._marginal_function_space
        u, v = TrialFunction(V), TestFunction(V)
        
        B_forms = [u.dx(0) * v.dx(0) * dx, u * v * dx]
        C_forms = [u * v * dx, u.dx(0) * v.dx(0) * dx]
        A_forms = list(zip(B_form, C_form))
        
        c_forms = [fx * v * dx for fx in self.fx]
        d_forms = [fy * v * dx for fy in self.fy]
        b_forms = list(zip(c_forms, d_forms))
        
        A, b = pf.assemble_product_system(A_forms, b_forms, self.bc)
        U = np.linalg.solve(A, b)
        return U
    

class DriftDiffusion:
    # assumes drift b = (b_1(x), b_2(y))
    def __init__(self, W, eps, b, f, bc):
        self.W = W            # product function space
        self.eps = eps        # diffusion coef
        self.b = b            # drift vector
        self.fx, self.fy = f  # forcing function
        self.bc = bc          # boundary conditions
        
    def solve(self):
        V = self.W._marginal_function_space
        u, v = TrialFunction(V), TestFunction(V)
        eps = self.eps
        b1, b2 = self.b
        
        B_forms = [eps * u.dx(0) * v.dx(0) * dx,
                   u * v * dx,
                   b1 * u.dx(0) * v * dx,
                   u * v * dx]
        C_forms = [u * v * dx,
                   eps * u.dx(0) * v.dx(0) * dx,
                   u * v * dx,
                   b2 * u.dx(0) * v * dx]
        A_forms = list(zip(B_forms, C_forms))
        
        c_forms = [fx * v * dx for fx in self.fx]
        d_forms = [fy * v * dx for fy in self.fy]
        b_forms = list(zip(c_forms, d_forms))
        
        A, b = pf.assemble_product_system(A_forms, b_forms, self.bc)
        U = np.linalg.solve(A, b)
        return U