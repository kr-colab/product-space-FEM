from .transforms import to_array, to_Function
from fenics import Function, FunctionSpace, VectorFunctionSpace, plot
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable


FenicsFunction = Function


class Function(FenicsFunction):
    """This class will be the base class for product_fem functions.
    To keep things in the family, we return a Function from various
    algebraic operations."""

    def __init__(self, V, dim=1, *args, **kwargs):
        if V.dolfin_element().value_dimension(0)!=dim:
            mesh = V.mesh()
            family = V.ufl_element().family()
            degree = V.ufl_element().degree()
            V = VectorFunctionSpace(mesh, family, degree, dim)

        self._basis = None
        super().__init__(V, *args, **kwargs)

    def __array__(self):
        return self.vector()[:]

    def __array_wrap__(self, array):
        return to_Function(array.copy(), self.function_space())

    def __mul__(self, other):
        try:
            if isinstance(other, (int, float, np.float64)):
                other_vector = other
            else:
                other_vector = other.vector()[:]
            out = self.copy()
            out.vector()[:] *= other_vector
        except AttributeError:
            out = super().__mul__(other)
        return out

    def __rmul__(self, other):
        return self.__mul__(other)

    def __add__(self, other):
        out = self.copy()
        out.vector()[:] += other.vector()[:]
        return out

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        out = self.copy()
        out.vector()[:] -= other.vector()[:]
        return out

    def __rsub__(self, other):
        return -self.__sub__(other)

    def copy(self):
        V = self.function_space()
        return to_Function(self.array().copy(), V)

    def array(self):
        return to_array(self, self.function_space())

    def value_dim(self):
        return self.value_dimension(0)

    def plot(self):
        if self.value_dim() < 3:
            return plot(self)
        else:
            raise NotImplementedError

    def dim(self):
        return self.function_space().dim()

    def _basis_i(self, i):
        V = self.function_space()
        dim = V.dolfin_element().value_dimension(0)
        name = f'{self.name()}_{i}'
        phi = Function(V, dim, name=name)
        phi.vector()[i] = 1.
        return phi

    @property
    def basis(self):
        if self._basis is None:
            self._basis = [self._basis_i(i) for i in range(self.dim())]
        return self._basis


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
        elif issubclass(other, ProductFunction):
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
        fig, ax = plt.subplots(figsize=(6,6))

        u_x = self.get_slice(x)
        p = plot(u_x)
        ax.scatter(x[0], x[1],
                   marker='*', c='orange', s=15**2,
                   edgecolors='white', alpha=0.6)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        fig.colorbar(p, cax=cax, orientation='vertical')
        plt.show()


class ProductBasisFunction(ProductFunction):

    def __init__(self, W, i, j, **kwargs):
        super().__init__(W, **kwargs)
        ij = W.dofmap._product_dofs_to_dofs[(i, j)]
        self.array[ij] = 1.
        self.phi_ij = W._marginal_basis_i(i), W._marginal_basis_i(j)

    def __call__(self, x, y):
        phi_i, phi_j = self.phi_ij
        return phi_i(x) * phi_j(y)


class Control:
    """List of Functions that act as control variables in constraint"""

    def __init__(self, control):
        if not isinstance(control, list):
            control = [control]
        self._control = control
        self.names = [m.name() for m in control]
        self.function_spaces = [m.function_space() for m in control]
        self.dims = [V.dim() for V in self.function_spaces]
        self.ids = [m.id() for m in control]

    def __mul__(self, other):
        result = [c * other for c in self._control]
        return Control(result)

    def __rmul__(self, other):
        result = [other * c for c in self._control]
        return Control(result)

    def __add__(self, other):
        result = [self[i] + other[i] for i in range(len(self))]
        return Control(result)

    def __len__(self):
        return len(self.names)

    def __getitem__(self, item):
        return self._control[item]

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
                xx = xx.array()
            arrays.append(xx)
        array = np.concatenate(tuple(arrays))
        return self._update_from_array(array)

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


class SpatialData:
    """Contains spatial sampling locations and measurements"""

    def __init__(self, data):
        self.data = data

    def as_product_function(self):
        """Returns the ProductFunction representation of spatial data"""
