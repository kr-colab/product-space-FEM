import product_fem as pf
import fenics as fx
import numpy as np
import pytest

def example_funs(include_fenics=True):
    mesh = fx.UnitSquareMesh(7, 8)
    V = fx.FunctionSpace(mesh, "CG", 1)
    yield pf.to_Function("1.0", V)
    yield pf.to_Function(lambda x, y: x + y, V)
    # TODO: some d=2
    # yield pf.to_Function(lambda x, y: [x ** 2, x + y], V)
    # x = np.linspace(0, 10, V.dim() * 2).reshape((V.dim(), 2))
    # yield pf.to_Function(x, V)
    if include_fenics:
        f = fx.Function(V)
        yield f
        f.vector()[:] = np.arange(len(f.vector()[:]))
        yield f

def example_pairs():
    for f in example_funs():
        for g in example_funs():
            if isinstance(f, fx.Function) or isinstance(g, fx.Function):
                fdim = f.function_space().dim()
                gdim = g.function_space().dim()
                if fdim == gdim:
                    yield (f, g)


class TestToFunctions:

    def test_function_from_callable(self):
        mesh = fx.UnitSquareMesh(7, 6)
        V = fx.VectorFunctionSpace(mesh, "CG", 1, 2)
        # this is linear, so should be exact
        def ff(x, y):
            return [x - 0.5, x + y]
        
        ff_array = pf.transforms.callable_to_array(ff, V)
        f = pf.to_Function(ff, V)
        for x in np.linspace(0, 1, 11):
            for y in np.linspace(0, 1, 8):
                assert np.allclose(f(x, y), ff(x, y))


# class TestFunctionArithmetic:

#     @pytest.mark.parametrize("fg", example_pairs())
#     def test_plus(self, fg):
#         f, g = fg
#         fpg = f + g
#         assert isinstance(fpg, fx.Function)
#         assert np.all(fpg.vector()[:] == f.vector()[:] + g.vector()[:])

#     @pytest.mark.parametrize("fg", example_pairs())
#     def test_times(self, fg):
#         f, g = fg
#         fpg = f * g
#         assert isinstance(fpg, fx.Function)
#         assert np.allclose(fpg.vector()[:], f.vector()[:] * g.vector()[:])

#     @pytest.mark.parametrize("fg", example_pairs())
#     def test_sub(self, fg):
#         f, g = fg
#         fmg = f - g
#         assert isinstance(fmg, fx.Function)
#         assert np.all(fmg.vector()[:] == f.vector()[:] - g.vector()[:])

#     @pytest.mark.parametrize("f", example_funs(include_fenics=False))
#     def test_scalar_mult(self, f):
#         for a in [0.0, -1.0, np.float64(0.0), np.float64(1/3)]:
#             for af in (a * f, f * a):
#                 assert isinstance(af, fx.Function)
#                 assert np.all(af.vector()[:] == f.vector()[:] * a)

#     @pytest.mark.parametrize("f", example_funs(include_fenics=False))
#     def test_copy(self, f):
#         assert f == f
#         f_copy = f.copy()
#         assert f != f_copy
#         for x, y in [(0.0, 0.0), (0.1, 0.0), (1/3, 0.5), (1.0, 0.0)]:
#             assert np.all(f(x, y) == f_copy(x, y))
#         assert np.all(f.vector()[:] == f_copy.vector()[:])
#         # test copy is deep
#         fx = f.vector()[0]
#         f.vector()[0] = fx + 1
#         assert f.vector()[0] == fx + 1
#         assert f_copy.vector()[0]  == fx
