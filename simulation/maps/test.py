import numpy as np

import map_utils


class TestConversion:

    def test_floats_to_rgb(self):
        for min, max in [(0, 1), (-1, 1), (-5, -2)]:
            x = np.linspace(min, max, 256)
            n = map_utils.floats_to_rgb(x, min=min, max=max)
            assert n.dtype is np.dtype("uint8")
            xx = map_utils.rgb_to_floats(n, min=min, max=max)
            assert np.allclose(x, xx)

    def test_rgb_to_floats(self):
        rng = np.random.default_rng(123)
        n = rng.integers(low=0, high=255, size=1000)
        for min, max in [(0, 1), (-1, 1), (-5, -2)]:
            x = map_utils.rgb_to_floats(n, min=min, max=max)
            assert x.dtype is np.dtype("float")
            nn = map_utils.floats_to_rgb(x, min=min, max=max)
            assert np.all(n == nn)

    def f1(self, x, y):
        return np.cos(x/3 + y/5)

    def test_xyz_to_array(self):
        nr, nc = 19, 12
        xvals = np.linspace(2, 30, nc)
        yvals = np.linspace(-4, 10, nr)
        x = np.empty(nr * nc)
        y = np.empty(nr * nc)
        z = np.empty(nr * nc)
        k = 0
        for i in range(nr):
            for j in range(nc):
                y[k] = yvals[i]
                x[k] = xvals[j]
                z[k] = self.f1(x[k], y[k])
                k += 1

        xx, yy, zz = map_utils.xyz_to_array(x, y, z)
        assert np.allclose(xvals, xx)
        assert np.allclose(yvals, yy)
        assert zz.shape[0] == nr
        assert zz.shape[1] == nc
        for i in range(nr):
            for j in range(nc):
                assert zz[i, j] == self.f1(xx[j], yy[i])

        rng = np.random.default_rng(123)
        pi = rng.permutation(nr * nc)
        x = x[pi]
        y = y[pi]
        z = z[pi]
        xx, yy, zz = map_utils.xyz_to_array(x, y, z)
        assert np.allclose(xvals, xx)
        assert np.allclose(yvals, yy)
        assert zz.shape[0] == nr
        assert zz.shape[1] == nc
        for i in range(nr):
            for j in range(nc):
                assert zz[i, j] == self.f1(xx[j], yy[i])

    def test_xyz_to_function(self):
        nr, nc = 19, 12
        xvals = np.linspace(2, 30, nr)
        yvals = np.linspace(-4, 10, nc)
        x = np.empty(nr * nc)
        y = np.empty(nr * nc)
        z = np.empty(nr * nc)
        k = 0
        for i in range(nr):
            for j in range(nc):
                x[k] = xvals[i]
                y[k] = yvals[j]
                z[k] = self.f1(x[k], y[k])
                k += 1

        ff = map_utils.xyz_to_function(x, y, z)
        for i in range(nr):
            for j in range(nc):
                xy = (xvals[i], yvals[j])
                assert np.isclose(ff(xy), self.f1(*xy))
