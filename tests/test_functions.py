import product_fem as pf
import fenics as fx
import numpy as np
import pytest

def example_funs():
    mesh = fx.UnitSquareMesh(7, 8)
    V = fx.FunctionSpace(mesh, "CG", 1)
    yield pf.to_Function("1.0", V)
    yield pf.to_Function(lambda x, y: x + y, V)
    f = fx.Function(V)
    yield f
    f.vector()[:] = np.arange(len(f.vector()[:]))
    yield f

def example_pairs():
    for f in example_funs():
        for g in example_funs():
            if isinstance(f, pf.Function) or isinstance(g, pf.Function):
                yield (f, g)

class TestFunctions:

    @pytest.mark.parametrize("fg", example_pairs())
    def test_plus(self, fg):
        f, g = fg
        fpg = f + g
        assert isinstance(fpg, pf.Function)
        assert np.all(fpg.vector()[:] == f.vector()[:] + g.vector()[:])

    @pytest.mark.parametrize("fg", example_pairs())
    def test_times(self, fg):
        f, g = fg
        fpg = f * g
        assert isinstance(fpg, pf.Function)
        assert np.allclose(fpg.vector()[:], f.vector()[:] * g.vector()[:])

    @pytest.mark.parametrize("fg", example_pairs())
    def test_sub(self, fg):
        f, g = fg
        fmg = f - g
        assert isinstance(fmg, pf.Function)
        assert np.all(fmg.vector()[:] == f.vector()[:] - g.vector()[:])

    @pytest.mark.parametrize("f", example_funs())
    def test_scalar_mult(self, f):
        for a in [0.0, -1.0]:
            for af in (a * f, f * a):
                assert isinstance(af, pf.Function)
                assert np.all(af.vector()[:] == f.vector()[:] * a)

    @pytest.mark.parametrize("f", example_funs())
    def test_copy(self, f):
        assert f == f
        f_copy = f.copy()
        assert f != f_copy
        for x, y in [(0.0, 0.0), (0.1, 0.0), (1/3, 0.5), (1.0, 0.0)]:
            assert f(x, y) == f_copy(x, y)
        assert np.all(f.vector()[:] == f_copy.vector()[:])
        # test copy is deep
        f.vector()[0] += 1
        assert f.vector()[0] == f_copy.vector()[0] + 1
