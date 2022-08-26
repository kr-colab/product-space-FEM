from fenics import project, interpolate, Expression, Function
import numpy as np
import scipy.sparse as sps
import product_fem as pf
import petsc4py.PETSc as PETSc


# DERIVATIVES
def derivative(form, control, direction):
    pass

# SPATIAL TRANSFORMS
def tensordot(X, Y, **kwargs):
    return np.tensordot(X, Y, **kwargs)

# translate long/lats to displacement from point p
def translate(xy, p):
    if p.ndim==1:
        p = p.reshape(-1, 1)
    assert len(xy)==len(p)
    return xy - p

# rotate coordinates xy to new x-axis v
def rotate(xy, v):
    if isinstance(v, list):
        v = np.array(v).reshape(-1, 1)
    v /= np.linalg.norm(v)
    cos, sin = v.flatten()
    R = np.array([[cos, sin], [-sin, cos]])
    return R.dot(xy)

# given point (x,y) the orthogonal projection onto v is 
# (vv^T)/(v^Tv) (x,y)
def proj(xy, v):
    if isinstance(v, list):
        v = np.array(v).reshape(-1, 1)
    if v.ndim==1:
        v = v.reshape(-1, 1)
#         v = v[:, np.newaxis]
    P = v.dot(v.T) / v.T.dot(v)
    assert len(P.T)==len(xy)
    return P.dot(xy)

def rescale(xy):
    return xy / np.max(xy)

# CONVERTERS
# scipy sparse to PETSc
def dense_to_PETSc(vector):
    return PETSc.Vec().createWithArray(vector)

def sparse_to_PETSc(matrix):
    # assumes sparse matrix is CSR
    assert isinstance(matrix, sps.csr_matrix)
    csr_encoding = matrix.indptr, matrix.indices, matrix.data
    M = PETSc.Mat().createAIJ(size=matrix.shape, csr=csr_encoding)
    return M

def PETSc_to_sparse(matrix):
    M = sps.csr_matrix(matrix.getValuesCSR()[::-1], shape=matrix.size)
    return M

def PETSc_matrix_kron(A, B):
    assert isinstance(A, PETSc.Mat)
    assert isinstance(B, PETSc.Mat)
    
    A, B = PETSc_to_sparse(A), PETSc_to_sparse(B)
    product_matrix = sps.kron(A, B, 'csr')
    return sparse_to_PETSc(product_matrix)

def PETSc_vector_kron(A, B):
    assert isinstance(A, PETSc.Vec)
    assert isinstance(B, PETSc.Vec)
    
    A, B = A[:], B[:]
    product_vector = np.kron(A, B)
    return dense_to_PETSc(product_vector)
    
def PETSc_kron(A, B):
    # inputs must be PETSc.Mat or PETSc.Vec
    if isinstance(A, PETSc.Mat):
        return PETSc_matrix_kron(A, B)
    elif isinstance(A, PETSc.Vec):
        return PETSc_vector_kron(A, B)

# from strings
def string_to_Function(string, V, proj=True):
    func = pf.Function(V)
    if proj:
        f = project(Expression(string, element=V.ufl_element()), V)
    else:
        f = interpolate(Expression(string, element=V.ufl_element()), V)
    func.assign(f)
    return func

def string_to_array(string, V, proj=True):
    func = string_to_Function(string, V, proj)
    array = Function_to_array(func)
    return array
    
# from python functions
def callable_to_array(func, V):
    dof_coords = V.tabulate_dof_coordinates()
    return np.array([func(*x) for x in dof_coords])

def callable_to_Function(func, V):
    array = callable_to_array(func, V)
    func = array_to_Function(array, V)
    return func

def callable_to_ProductFunction(func, V):
    array = callable_to_array(func, V)
    func = array_to_ProductFunction(array, V)
    return func
    
# from dolfin Functions
def Function_to_array(func):
    array = func.vector()[:]
    return array

def Function_to_Function(func):
    V = func.function_space()
    f = pf.Function(V)
    f.assign(func)
    return f

# from numpy arrays
def array_to_Function(array, V):
    dim = 1
    if len(array.shape) == 2:
        dim = array.shape[1]
    f = pf.Function(V, dim=dim)
    f.vector()[:] = array.copy().flatten()
    return f

def array_to_ProductFunction(array, W):
    f = pf.ProductFunction(W)
    f.assign(array)
    return f

# from ufl forms
def form_to_array(form):
    array = assemble(form)
    rank = len(form.arguments())
    if rank==0:
        return array.real
    elif rank==1:
        return array[:]
    elif rank==2:
        return array.array()
    
# to product_fem Functions
def to_Function(func, V):
    
    # from dolfin Function
    if isinstance(func, Function):
        return Function_to_Function(func)
    
    # from strings
    if isinstance(func, str):
        return string_to_Function(func, V)
    
    # from array
    elif isinstance(func, np.ndarray):
        if isinstance(V, pf.ProductFunctionSpace):
            return array_to_ProductFunction(func, V)
        else:
            return array_to_Function(func, V)
    
    # from callable
    elif callable(func):
        if isinstance(V, pf.ProductFunctionSpace):
            return callable_to_ProductFunction(func, V)
        else:
            return callable_to_Function(func, V)
    
# to numpy arrays
def to_array(func, V):
    
    # from strings
    if isinstance(func, str):
        return string_to_array(func, V)
    
    # from dolfin or product_fem Function
    elif isinstance(func, (pf.Function, Function)):
        return Function_to_array(func)
    
    # from callable
    elif callable(func):
        return callable_to_array(func, V)
    
