from fenics import project, interpolate, Expression, Function, FunctionSpace, VectorFunctionSpace, cells
import numpy as np
import scipy.sparse as sps
import product_fem as pf
import petsc4py.PETSc as PETSc


# SPATIAL TRANSFORMS
# translate long/lats to displacement from point p
def translate(xy, p):
    if p.ndim==1:
        p = p.reshape(-1, 1)
    assert len(xy)==len(p)
    return xy - p

# rotate coordinates xy to new x-axis v
def rotate(xy, v):
    if isinstance(v, list):
        v = np.array(v)
    assert np.linalg.norm(v) > np.finfo(v.dtype).eps
    v /= np.linalg.norm(v)
    cos, sin = v.flatten()
    R = np.array([[cos, -sin], [sin, cos]])
    return xy.dot(R.T)

def stretch(xy, s):
    # rescale x and y axes by factor of s[0] and s[1], resp.
    return xy * s
    
# given point (x,y) the orthogonal projection onto v is 
# (vv^T)/(v^Tv) (x,y)
def proj(xy, v):
    if isinstance(v, list):
        v = np.array(v).reshape(-1, 1)
    if v.ndim==1:
        v = v.reshape(-1, 1)
    P = v.dot(v.T) / v.T.dot(v)
    assert len(P.T)==len(xy)
    return P.dot(xy)

def rescale(xy):
    return xy / np.max(xy)

# CONVERTERS
# scipy sparse to PETSc
def dense_to_PETSc(vector):
    petsc_vec = PETSc.Vec().createWithArray(vector)
    return petsc_vec

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
        kron = PETSc_matrix_kron(A, B)
    elif isinstance(A, PETSc.Vec):
        kron = PETSc_vector_kron(A, B)
    
    return kron

# from strings
def string_to_Function(string, V, proj=True):
    if proj:
        f = project(Expression(string, element=V.ufl_element()), V)
    else:
        f = interpolate(Expression(string, element=V.ufl_element()), V)
    return f

def string_to_array(string, V, proj=True):
    func = string_to_Function(string, V, proj)
    array = Function_to_array(func)
    return array
    
# from python functions
def callable_to_array(func, V):
    dim = V.dolfin_element().value_dimension(0)
    dof_coords = V.tabulate_dof_coordinates()[::dim]
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

# from numpy arrays
def array_to_Function(array, V):
    f = Function(V)
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
    elif isinstance(func, Function):
        return Function_to_array(func)
    
    # from callable
    elif callable(func):
        return callable_to_array(func, V)
    
def function_space_basis(V):
    basis = []
    for i in range(V.dim()):
        f = Function(V)
        f.vector()[i] = 1.
        basis.append(f)
    return basis

def vectorized_fn(V, dim, name):
    mesh = V.mesh()
    family = V.ufl_element().family()
    degree = V.ufl_element().degree()
    VV = VectorFunctionSpace(mesh, family, degree, dim)
    return Function(VV, name=name)

def sig_to_cov(sig):
    """
    Converts the unconstrained parametrization of the covariance matrix
    to a dim=3 array where cov[i] is the covariance matrix at the ith dof
    
    At each point, the unconstrained parametrization sig has 3 values
    and the covariance matrix K = L^t L where 
    
        L = [ exp(sig[0])    sig[2]  ] 
            [     0       exp(sig[1])]
        
    This factorization is unique so long as the diagonal of L is positive,
    which explains the use of exp here. Explicitly, the covariance matrix is
    
        K = [   exp(2 * sig[0])        exp(sig[0]) * sig[2]   ]
            [ exp(sig[0]) * sig[2]   exp(2 sig[1]) + sig[2]^2 ]
    """
    # get array if sig is Function
    if isinstance(sig, Function):
        assert len(sig)==3
        sig = sig.vector()[:]
    
    # split sig into components
    assert len(sig) % 3 == 0
    s0 = sig[::3]
    s1 = sig[1::3]
    s2 = sig[2::3]
    
    # L at each dof, moveaxis so L[i] is 2x2 covariance matrix at ith dof
    Ls = np.array( [[np.exp(s0), s2], [np.zeros(len(s0)), np.exp(s1)]] )
    Ls = np.moveaxis(Ls, source=[0,1,2], destination=[1,2,0])
        
    # vectorized calculation of L^t L at each dof
    covs = np.einsum('ijl, ilk -> ijk', Ls.transpose(0,2,1), Ls)
    return covs

def cov_eigenvals(cov):
    """Assume cov is 2x2 spd matrix.
    The eigenvalues of a symmetric 2x2 matrix [[a, b], [b, c]] are
        lambda = 1/2 (a+c +- sqrt((a-c)^2 + 4b^2))
    """
    a, b, c = cov[0,0], cov[1,0], cov[1,1]
    ac = a + c
    root = np.sqrt((a - c)**2 + 4 * b**2)
    lambda_1 = 1/2 * (ac + root)
    lambda_2 = 1/2 * (ac - root)
    return lambda_1, lambda_2

def local_peclets(mu, sig):
    """For a mesh cell C and a point x in C the Peclet number is
        Pe_C(x) = (|mu(x)| h_C) / (2 a(x))
    where a(x) is the smallest eigenvalue of the covariance matrix at x
    and h_C is the scale parameter for the cell.
    We compute local Peclet numbers by looping through the mesh cells,
    letting x be the cell midpoint.
    """
    mesh = mu.function_space().mesh()
    pecs = []
    for cell in cells(mesh):
        x = cell.midpoint().array()[:-1]
        mu_x = np.linalg.norm(mu(x))
        sig_x = sig(x)
        cov_x = sig_to_cov(sig_x).squeeze()
        _, lambda_2 = cov_eigenvals(cov_x)
        h = cell.h()
        
        peclet = 0.5 * mu_x * h / lambda_2
        pecs.append((x, peclet))
    
    return pecs