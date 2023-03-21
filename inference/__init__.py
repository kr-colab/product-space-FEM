import itertools
import warnings
import numpy as np
import pandas as pd
from scipy import optimize

class SpatialDivergenceData:

    def __init__(self, spatial_data, genetic_data):
        """
        `spatial_data`: should be a data frame with columns `name`, `x`, and `y`.
        `genetic_data`: should be a data frame with columns `name1`, `name2`,
            and `divergence` (which when name1==name2 is heterozygosity).
        """
        for n in ('name', 'x', 'y'):
            assert n in spatial_data.columns, f"{n} not in spatial data"
        self.spatial_data = spatial_data.set_index('name')
        for n in ('name1', 'name2', 'divergence'):
            assert n in genetic_data.columns, f"{n} not in genetic data"
        genetic_data['pair'] = [
                (a, b) if a < b else (b, a)
                for a, b in zip(genetic_data['name1'], genetic_data['name2'])
        ]
        self.genetic_data = genetic_data.set_index('pair')
        self.scaling = {
            'x' : {
                'shift' : 0.0,
                'scale' : 1.0
            },
            'y' : {
                'shift' : 0.0,
                'scale' : 1.0
            },
            'divergence' : {
                'scale' : 1.0
            },
        }

    def _normalise(self, x, x0, x1, shift=True):
        # beware! acts on x in place.
        b = np.min(x) - x0 if shift else 0
        a = (np.max(x) - np.min(x)) / (x1 - x0)
        x -= b
        x /= a
        return a, b

    def normalise(self, min_xy=0, max_xy=1, max_div=1):
        x_range = np.max(self.spatial_data['x']) - np.min(self.spatial_data['x'])
        y_range = np.max(self.spatial_data['y']) - np.min(self.spatial_data['y'])
        min_x = min_y = min_xy
        max_x = max_xy * min(1, x_range / y_range)
        max_y = max_xy * min(1, y_range / x_range)
        a, b = self._normalise(self.spatial_data.loc[:, 'x'], min_x, max_x)
        self.scaling['x']['shift'] = b
        self.scaling['x']['scale'] = a
        a, b = self._normalise(self.spatial_data.loc[:, 'y'], min_y, max_y)
        self.scaling['y']['shift'] = b
        self.scaling['y']['scale'] = a
        a, _ = self._normalise(self.genetic_data.loc[:, 'divergence'], 0, max_div, shift=False)
        self.scaling['divergence']['scale'] = a

    def normalize(self, *args, **kwargs):
        self.normalise(*args, **kwargs)

    def pair_xy(self):
        """
        returns the xy locations for the pairs of locations in genetic_data
        """
        xy0 = self.spatial_data.loc[self.genetic_data['name1'], ('x', 'y')].to_numpy()
        xy1 = self.spatial_data.loc[self.genetic_data['name2'], ('x', 'y')].to_numpy()
        return xy0, xy1

    def distances(self):
        """
        Returns the vector of pairwise geographic distances corresponding to the genetic data.
        """
        xy0, xy1 = self.pair_xy()
        return np.hypot(xy1[:,0] - xy0[:,0], xy1[:,1] - xy0[:,1])

    def distance_matrix(self):
        sd = self.spatial_data
        gd = self.genetic_data
        n = self.spatial_data.shape[0]
        out = np.full((n, n), -1, dtype='float')
        dists = self.distances()
        for k, d in enumerate(dists):
            a = gd['name1'][k]
            b = gd['name2'][k]
            i = np.where(sd.index == a)[0][0]
            j = np.where(sd.index == b)[0][0]
            out[i, j] = out[j, i] = d
        return out

    def boundary_fn(self, eps0, eps1):
        """
        Returns the function which gives a Gaussian kernel density estimate
        for divergence between two points, used on the boundary (but works anywhere,
        although maybe not well).

        The weight for two points with midpoint xy on div(xy0, xy1) is proportional to
        exp(- |xy0 - xy1|^2/(2 eps0^2) - |xy - xy0|^2/(2 eps1^2) - |xy - xy0|^2/(2 eps1^2)) ,
        i.e., the total length of the triangle but with the leg between xy0 and xy1
        scaled diferently.
        """
        xy0, xy1 = self.pair_xy()
        d01_sq = ( (xy1[:,0] - xy0[:,0])**2 + (xy1[:,1] - xy0[:,1])**2 ) / (eps0**2)

        def boundary(x, y):
            xy = np.add(x, y) / 2
            dists_sq = (
                    d01_sq
                    + ( (xy0[:,0] - xy[0])**2 + (xy0[:,1] - xy[1])**2
                       + (xy1[:,0] - xy[0])**2 + (xy1[:,1] - xy[1])**2 ) / (eps1**2)
                )
            dists_sq -= np.mean(dists_sq)
            kernl = np.exp(-dists_sq/2)
            return np.sum(self.genetic_data['divergence'] * kernl) / np.sum(kernl)

        return boundary

    def _subset_locations(self, names, both=True):
        if both:
            sd = self.spatial_data.loc[names, :].reset_index(names="name")
            ut = np.logical_and(
                    np.isin(self.genetic_data['name1'], names),
                    np.isin(self.genetic_data['name2'], names)
            )
        else:
            sd = self.spatial_data.loc[:, :].reset_index(names="name")
            ut = np.logical_or(
                    np.isin(self.genetic_data['name1'], names),
                    np.isin(self.genetic_data['name2'], names)
            )
        gd = self.genetic_data.iloc[ut, :].reset_index(drop=True)
        return sd, gd

    def split(self, k, include_between=False, seed=None):
        """
        An iterator over even, non-overlapping splits of the locations into k folds,
        returning train, test SpatialDivergenceData objects,
        where `train` has only data for the 'training' locations,
        and `test` has data between 'test' locations; if
        `include_between` is True then `test` also includes data between
        the 'testing' and 'training' locations.
        """
        n = self.spatial_data.shape[0]
        rng = np.random.default_rng(seed=seed)
        kvec = np.array((list(range(k)) * int(np.ceil(n/k)))[:n])
        rng.shuffle(kvec)
        for j in range(k):
            train_names = self.spatial_data.index[kvec != j]
            train_sd, train_gd = self._subset_locations(train_names, both=True)
            train = SpatialDivergenceData(train_sd, train_gd)
            test_names = self.spatial_data.index[kvec == j]
            test_sd, test_gd = self._subset_locations(test_names, both=not include_between)
            test = SpatialDivergenceData(test_sd, test_gd)
            yield train, test

    def _boundary_xval_mad(self, eps0, eps1, k, small, seed=None):
        """
        Return the list of MADs across folds for the boundary function across
        pairs in the test data of distance less than or equal to `small`.
        """
        mads = [0.0 for _ in range(k)]
        for j, (train, test) in enumerate(self.split(k, seed=seed)):
            b = train.boundary_fn(eps0, eps1)
            xy0, xy1 = test.pair_xy()
            dists = np.hypot(xy0[:,0] - xy1[:,0], xy0[:,1] - xy1[:,1])
            diffs = [
                b(x, y) - div
                for div, x, y, d in zip(test.genetic_data['divergence'], xy0, xy1, dists)
                if d <= small
            ]
            mads[j] = np.median(np.abs(diffs))
        return mads

    def choose_epsilons(self, k=20, small=None, big=None, method='random', seed=None, num_guesses=400, **kwargs):
        """
        Do crossvalidation to find good epsilons for the boundary function,
        by random hyperparamter search!
        """
        dists = self.distance_matrix()
        if small is None:
            small = np.median([np.min(x[x>0]) for x in dists]) / 2
        if big is None:
            big = np.median([np.quantile(x[x>0], 0.2) for x in dists])
        def f(eps):
            return np.mean(self._boundary_xval_mad(eps0=eps[0], eps1=eps[1], k=k, small=small/4))
        eps0 = small
        eps1 = (small + big)/2
        start_f = f((eps0, eps1))
        print(f"Choosing epsilons: \n"
              f"    starting at eps0={eps0}, eps1={eps1} \n"
              f"    with mean MAD = {start_f}")
        if method == 'random':
            rng = np.random.default_rng(seed=seed)
            end_f = start_f
            for _ in range(num_guesses):
                e0 = rng.uniform(small/5, small*20)
                e1 = rng.uniform(small/5, 2*big)
                new_f = f((e0, e1))
                if new_f < end_f:
                    end_f = new_f
                    eps0, eps1 = e0, e1
        elif method == 'trust-constr':
            opt = optimize.minimize(f, x0=(eps0, eps1), method="trust-constr",
                                    bounds=[(0, 2*big), (0, 2*big)])
            eps0, eps1 = opt['x']
            end_f = f((eps0, eps1))
            if not opt['success']:
                warnings.warn(f"Optimization error: {opt['message']}")
        else:
            raise ValueError(f"method {method} unknown")
        print(f"Complete: \n"
              f"    ending at eps0={eps0}, eps1={eps1} \n"
              f"    with mean MAD = {end_f}")
        return {'eps0': eps0, 'eps1': eps1, 'start': start_f, 'end': end_f}
        

