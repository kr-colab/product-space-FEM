import numpy as np
import scipy.sparse as sps
from fenics import *
import product_fem as prod

# This module is designed to calculate the gradient of a loss
# functional J wrt PDE parameter(s) m. This enables us to optimize
# J(u, m) under the PDE-constraint F(u, m) = 0.
# 
# We assume the forward system Au=b has already been assembled
# and possibly solved using the product_fem module.
#
# In this case v, the adjoint function associated with u,
# solves the system A^T v^T = dJdu^T, where vanishing boundary
# conditions are applied to v.
#
# Once the adjoint system is solved, the desired gradient is
#   nabla J = -v * dFdm + dJdm
#
# To compute dFdm we can either use AD or, seeing as F = Au - b
# it follows that dFdm = dAdm u - dbdm and we should be able
# to write down these partial derivatives from the integral
# definitions of A_ij and b_i
# Note dAdm is a tensor with shape (rows(A), cols(A), dim(m)),
# the contraction dAdm u is taken over the middle index,
# and dbdm is a matrix with dim(b) rows and dim(m) columns


class LossFunctional:
    def __init__(self, kind, u, m, data, prod_func_space):
        # kind (str) one of 'L2', 'L2+reg'
        # Here, data is a set of observations modeled with F(u, m) = 0
        self.kind = kind
        self.u = u
        self.m = m
        self.data = data
        self.product_function_space = prod_func_space
        
    def du(self):
        # Let _u denote an arbitrary perturbation of u, so that dJdu(_u) denotes
        # the directional derivative of J wrt u in the direction of _u
        # If J := 0.5 L2(u - data) then dJdu(_u) = (u - data) _u dx
        # Numerically, the ijth entry of dJdu is dJdu(phi_ij)
        # for finite element basis function phi_ij of u
        # Note dJdu is a row vector with dim=dim(u)
        
        # want to compute r^T K where r_ij = (u-u_d)(x_i, y_j)
        # and K_{ij,kl} = int(phi_i phi_k) dx * int(phi_j phi_l) dy
        r = self.u - self.data
        r = np.expand_dims(r, axis=1) # make column vector
        
        V = self.product_function_space._marginal_function_space
        _u, _v = TrialFunction(V), TestFunction(V)
        K_form = _u * _v * dx
        K = assemble(K_form).array()
        K = np.kron(K, K)
        dJdu = np.dot(r.T, K)
        return dJdu
    
    def dm(self):
        # Let _m denote an arbitrary perturbation of m, so that dJdm(_m) denotes
        # the directional derivative of J wrt m in the direction of _m
        # If J := 0.5 L2(u - data) then dJdm = 0
        # If J := L2(u - data) + alpha/2 L2(m) then dJdm(_m) = alpha m _m dx
        # Numerically, the ith entry of dJdm is dJdm(phi_i)
        # for finite element basis function phi_i of m
        # Note dJdm is a row vector with dim=dim(m)
        
        # Need to write dJdm as kronecker product kron(A, B) + kron(C, D)
        pass


class AdjointSystem:
    def __init__(self, product_stiffness, loss_functional):
        self.functional = loss_functional
        self.stiffness = product_stiffness
    
    def solve(self):
        # TODO
        #  apply vanishing bc to v
        A = self.stiffness
        return np.linalg.solve(A.T, self.functional.du().T)
    
    def compute_gradient(self):
        # Once the adjoint system is solved, the desired gradient is
        #   nabla J = -v * dFdm + dJdm
        v = self.solve()
        dJdm = self.functional.dm()
        dFdm = dAdm * u - dbdm
        
        
        
        