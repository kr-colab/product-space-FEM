import numpy as np
import pandas as pd

import inference

class TestSpatialDivergenceData:

    def get_example(self, n=None, seed=None):
        if n is None:
            spatial_data = pd.DataFrame({
                'name' : ['one', 'two', 'three'],
                'x': [0.2, 3, -2],
                'y': [1, 0, 11],
            })
            genetic_data = pd.DataFrame({
                'name1' : ['one', 'one', 'one', 'two', 'two', 'three'],
                'name2' : ['one', 'two', 'three', 'two', 'three', 'three'],
                'divergence': [0.5, 0.0, 8, 0.75, 1.25, 0.05],
            })
        else:
            rng = np.random.default_rng(seed=seed)
            names = [f"a{k}" for k in np.arange(n)]
            spatial_data = pd.DataFrame({
                'name' : names,
                'x' : rng.uniform(0, 4, n),
                'y' : rng.uniform(-2, 10, n),
            })

            def g(x, y):
                d = np.hypot(x[0] - y[0], x[1] - y[1])
                return np.exp( (x[0]+y[0])/2 - d/3 + rng.normal(loc=0, scale=0.1, size=1))

            name_pairs = [
                    (names[i], names[j])
                    for i in range(len(names)) for j in range(len(names)) if i <= j
            ]
            div = [
                g((spatial_data['x'][i], spatial_data['y'][i]),
                  (spatial_data['x'][j], spatial_data['y'][j]))
                for i in range(len(names)) for j in range(len(names)) if i <= j
            ]
            genetic_data = pd.DataFrame({
                'name1' : [a for a, _ in name_pairs],
                'name2' : [b for _, b in name_pairs],
                'divergence' : div,
            })
        return spatial_data, genetic_data

    def test_init(self):
        sdd = inference.SpatialDivergenceData(*self.get_example())
        sd = sdd.spatial_data
        gd = sdd.genetic_data
        assert sd.shape[0] == 3
        assert gd.shape[0] == 6
        assert np.all(sd.index == ['one', 'two', 'three'])
        for a in sd.index:
            for b in sd.index:
                assert (a, b) in gd.index or (b, a) in gd.index

    def test_normalise(self):
        sd, gd = self.get_example()
        sdd = inference.SpatialDivergenceData(sd, gd)
        sdd.normalise()
        new_sd = sdd.spatial_data
        new_gd = sdd.genetic_data
        assert np.all(new_sd['x'] >= 0)
        assert np.all(new_sd['x'] <= 1)
        assert np.all(new_sd['y'] >= 0)
        assert np.all(new_sd['y'] <= 1)
        assert np.all(new_gd['divergence'] >= 0)
        assert np.all(new_gd['divergence'] <= 1)
        assert np.allclose(
                sd['x'],
                sdd.scaling['x']['shift'] + sdd.scaling['x']['scale'] * new_sd['x']
        )
        assert np.allclose(
                sd['y'],
                sdd.scaling['y']['shift'] + sdd.scaling['y']['scale'] * new_sd['y']
        )
        assert np.allclose(
                gd['divergence'],
                sdd.scaling['divergence']['scale'] * new_gd['divergence']
        )

    def test_boundary_fn_runs(self):
        for multiplier in (1.0, 0.0):
            sd, gd = self.get_example()
            gd['divergence'] *= multiplier
            sdd = inference.SpatialDivergenceData(sd, gd)
            f = sdd.boundary_fn(eps0=0.5, eps1=10)
            mindxy, maxdxy = np.min(sdd.genetic_data['divergence']), np.max(sdd.genetic_data['divergence']),
            for x in [(0, 1.), (-2, 0.4), (6, 0.)]:
                for y in [(0, 1.), (-2, 0.4), (6, 0.)]:
                    fxy = f(x, y)
                    assert fxy >= mindxy and fxy <= maxdxy
    
    def test_distances(self):
        sdd = inference.SpatialDivergenceData(*self.get_example())
        sd = sdd.spatial_data
        gd = sdd.genetic_data
        x = sd['x']
        y = sd['y']
        d = sdd.distances()
        D = sdd.distance_matrix()
        assert len(d) == gd.shape[0]
        for i in range(gd.shape[0]):
            j1 = np.where(sd.index == gd['name1'][i])[0][0]
            j2 = np.where(sd.index == gd['name2'][i])[0][0]
            d12 = np.sqrt( (x[j1] - x[j2])**2 + (y[j1] - y[j2])**2 )
            assert np.isclose(d[i], d12)
            assert d[i] == D[j1, j2]
            assert d[i] == D[j2, j1]

    def validate_data(self, sdd):
        sd = sdd.spatial_data
        gd = sdd.genetic_data
        # check all pairwise comparisons are in gd
        assert gd.shape[0] == sd.shape[0] * (sd.shape[0] + 1) / 2
        for n in sd.index:
            for m in sd.index:
                a = np.logical_or(
                        np.logical_and( gd['name1'] == n, gd['name2'] == m),
                        np.logical_and( gd['name2'] == n, gd['name1'] == m),
                )
                assert np.sum(a) == 1

    def test_split(self):
        for k in (3, 5, 10):
            sdd = inference.SpatialDivergenceData(*self.get_example(n=20))
            sd, gd = sdd.spatial_data, sdd.genetic_data
            self.validate_data(sdd)
            is_test = np.zeros(sd.shape[0])
            for train, test in sdd.split(k=5):
                is_test[np.isin(sd.index, test.spatial_data.index)] += 1
                self.validate_data(train)
                self.validate_data(test)
                assert train.spatial_data.shape[0] + test.spatial_data.shape[0] == sd.shape[0]
                for n in sd.index:
                    num = int(n in train.spatial_data.index) + int(n in test.spatial_data.index)
                    if num != 1:
                        print(n, train.spatial_data.index)
                    assert 1 == int(n in train.spatial_data.index) + int(n in test.spatial_data.index)
            assert np.all(is_test == 1)

    def test_choose_epsilons(self):
        sdd = inference.SpatialDivergenceData(*self.get_example(n=30, seed=123))
        opt = sdd.choose_epsilons(k=10, num_guesses=50)
        assert opt['eps0'] > 0
        assert opt['eps1'] > 0
        assert opt['end'] < opt['start']
