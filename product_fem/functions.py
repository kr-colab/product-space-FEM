from .transforms import to_array, to_Function, function_space_basis
from fenics import Function, FunctionSpace, VectorFunctionSpace, plot
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable


class Control:
    """List of Functions that act as control variables in constraint"""

    def __init__(self, control):
        if not isinstance(control, list):
            control = [control]
        self.functions = control
        self.function_spaces = [m.function_space() for m in control]
        self.dims = [V.dim() for V in self.function_spaces]
        self.names = [m.name() for m in control]
        self.ids = [m.id() for m in control]
        self._bases = None

    def __mul__(self, other):
        result = [c * other for c in self.functions]
        return Control(result)

    def __rmul__(self, other):
        result = [other * c for c in self.functions]
        return Control(result)

    def __add__(self, other):
        result = [self[i] + other[i] for i in range(len(self))]
        return Control(result)

    def __len__(self):
        return len(self.function_spaces)

    def __getitem__(self, item):
        return self.functions[item]

    def _get_ids(self):
        return [m.id() for m in self]

    def _update_from_array(self, array):
        assert len(array)==self.dim()
        arrays = np.split(array, np.cumsum(self.dims))[:-1]
        assert len(arrays)==len(self)
        for i in range(len(self)):
            self[i].vector()[:] = arrays[i].copy()
            assert self[i].id()==self.ids[i]

    def _update_from_control(self, control):
        assert len(control)==len(self)
        for i in range(len(self)):
            self[i].assign(control[i])
            assert self[i].id()==self.ids[i]

    def _update_from_list(self, x):
        assert len(x) == len(self)
        arrays = []
        for xx in x:
            if isinstance(xx, Function):
                xx = xx.vector()[:]
            arrays.append(xx)
        array = np.concatenate(tuple(arrays))
        return self._update_from_array(array)
    
    def argwhere(self, m):
        for i, mi in enumerate(self):
            if mi==m: return i
        
    def get_basis(self, m):
        return self.bases[self.argwhere(m)]
    
    def _get_bases(self):
        bases = []
        for V in self.function_spaces:
            bases.append(function_space_basis(V))
        return bases
    
    @property
    def bases(self):
        if self._bases is None:
            self._bases = self._get_bases()
        return self._bases
            
    def dim(self):
        return sum(self.dims)

    def update(self, m_new):
        if isinstance(m_new, np.ndarray):
            self._update_from_array(m_new)
        elif isinstance(m_new, Control):
            self._update_from_control(m_new)
        elif isinstance(m_new, list):
            self._update_from_list(m_new)

    def split(self):
        return self.control

    def split_arrays(self):
        return [to_array(m, m.function_space()) for m in self]

    def array(self):
        return np.concatenate(tuple(self.split_arrays()))

    # untested
    def plot(self):
        fig, ax = plt.subplots(len(self))
        for i, m in enumerate(self):
            ax[i] = m.plot()
        plt.show()


class ProductFunction:

    def __init__(self, W, **kwargs):
        # initializes product space function
        # f(x,y) = sum_ij f_ij phi_i(x)phi_j(y)
        # by default f_ij=0 for all ij
        self.W = W
        self.array = np.zeros(W.dim())
        self._name = kwargs.get('name', '')

    def __len__(self):
        return len(self.array)

    def __mul__(self, other):
        result = ProductFunction(self.W)

        # scalar mult
        if isinstance(other, (int, float)):
            result.assign(self.array * other)

        # product function mult
        elif isinstance(other, ProductFunction):
            result.assign(self.array * other.array)

        return result

    def __rmul__(self, other):
        return self.__mul__(other)

    def __add__(self, other):
        result = ProductFunction(self.W)
        result.assign(self.array + other.array)
        return result

    def __sub__(self, other):
        result = ProductFunction(self.W)
        result.assign(self.array - other.array)
        return result

    def __pow__(self, other):
        result = ProductFunction(self.W)
        result.assign(self.array**other)
        return result

    def __call__(self, x, y):
        g = self.get_slice(x)
        return g(y)

    def dim(self):
        return self.W.dim()

    def name(self):
        return self._name

    def function_space(self):
        return self.W

    def assign_from_callable(self, f):
        # assigns values in array f to product function f(x,y)
        # i.e. f contains f_ij
        assert callable(f)
        f_array = np.zeros_like(self.array)
        dof_coords = self.W.tabulate_dof_coordinates()
        for i, xy in enumerate(dof_coords):
            f_array[i] = f(*xy) # when f is a python function of (x,y)
        self.array = f_array

    def assign_from_array(self, f):
        self.array = f

    def assign(self, f):
        if callable(f):
            return self.assign_from_callable(f)
        else:
            return self.assign_from_array(f)

    def dot(self, other):
        return self.array.dot(other)

    def as_matrix(self):
        """Since u(x,y) = sum_ij U_ij phi_i(x)phi_j(y)
        we can represent a ProductFunction as the matrix U."""
        return self.array.reshape(self.W.V.dim(), self.W.V.dim())

    def get_slice(self, x):
        """Given slice x, return u_x(y) := u(x,y) as a function of y.
        This sliced function is now a Function with function space V.
        In V the slice is expanded as u_x(y) = sum_j U_j phi_j(y)
        where U_j = sum_i u_ij phi_i(x)"""
        basis = self.W.marginal_basis()
        basis_x = np.array([phi(x) for phi in basis])
        U = self.as_matrix()
        u_x = basis_x.dot(U)
        return to_Function(u_x, self.W.V)

    def plot(self, x):
        fig, ax = plt.subplots(dpi=150)

        u_x = self.get_slice(x)
        p = plot(u_x)
        ax.scatter(x[0], x[1],
                   marker='*', c='orange', s=15**2,
                   edgecolors='white', alpha=0.6)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        fig.colorbar(p, cax=cax, orientation='vertical')
        return fig


# NOTE: it would be more efficient to store just the marginal bases 
#       and perform a kronecker product when needed
class ProductBasisFunction(ProductFunction):

    def __init__(self, W, i, j, **kwargs):
        super().__init__(W, **kwargs)
        ij = W.dofmap._product_dofs_to_dofs[(i, j)]
        self.array[ij] = 1.
        self.phi_ij = W._marginal_basis_i(i), W._marginal_basis_i(j)

    def __call__(self, x, y):
        phi_i, phi_j = self.phi_ij
        return phi_i(x) * phi_j(y)



class SpatialData:
    """Contains spatial sampling locations and measurements"""

    def __init__(self, data, points, W):
        self.data = data
        self.points = points
        self.W = W
        self.eval_matrix = self._assemble_eval_matrix()
        
    def _assemble_eval_matrix(self):
        """Given n sample points, we wish to evaluate an arbitrary
        ProductFunction u at the n(n+1)/2 unordered pairs of points.
        The evaluation matrix E is defined so that
            u(x_i, y_i) = (EU)_i     i = 1, ..., n(n+1)/2
        where u(x,y) = sum_i U_i phi_i(x,y)
        """
        n = len(self.points)
        N = int(n * (n + 1) / 2)
        basis = self.W.marginal_basis()
        
        idx, E = 0, np.zeros((N, self.W.dim()))
        for i, x in enumerate(self.points):
            px = [phi(x) for phi in basis]
            for y in self.points[i:]:
                py = [phi(y) for phi in basis]
                E[idx] = np.kron(px, py)
                idx += 1
        return E