import numpy as np

class DiscretizedFunction:
    def __init__(self, bounds, length_scales, init_sum_f=0., init_sum_f2=1., pseudo_counts=1):
        self.bins = []
        for b_idx, b in enumerate(bounds):
            l = length_scales[b_idx]
            num_bins = np.maximum(int((b[1]-b[0]) / l), 10)
#             print('num bins', num_bins, l)
            assert num_bins >= 10, f'Error = {b}, {l}'
            self.bins.append(np.linspace(b[0],b[1],num_bins))


        self.sum_f = init_sum_f * np.ones(tuple([b.size-1 for b in self.bins]))
        self.sum_f2 = init_sum_f2 * np.ones(self.sum_f.shape)
        self.counts = pseudo_counts * np.ones(self.sum_f.shape)

    def update(self, x, f):
        assert x.shape[0] == 1
        idx = self._get_idx(x[0,:])
        self.counts.flat[idx] += 1
        self.sum_f.flat[idx] += f
        self.sum_f2.flat[idx] += f**2

    def predict(self, x):
        assert x.shape[0] == 1
        idx = self._get_idx(x[0,:])
        n = self.counts.flat[idx]
        mu = self.sum_f.flat[idx] / n
        var = (self.sum_f2.flat[idx] / n) - (mu * mu)
        return mu, np.sqrt(var)

    def _get_idx(self, x):
        v = np.squeeze(
                [np.maximum(0, np.digitize(x[b_idx], b, right=True)-1) for b_idx, b in enumerate(self.bins)]
                )
        return np.ravel_multi_index(v,self.sum_f.shape)

    def sample(self, acq_func):

        mean_arr = np.divide(self.sum_f, self.counts)
        var_arr = np.divide(self.sum_f2, self.counts) - np.multiply(mean_arr, mean_arr)
        acq_vals = acq_func(mean_arr, var_arr)

        try:
            max_val = np.max(acq_vals)
        except ValueError as ve:
            raise RuntimeError(f'acq_func returned unusable values: {ve}') from ve
        idxs = np.where(acq_vals == max_val)
        choice_idx = np.random.choice(np.arange(len(idxs[0])))
        xs = np.squeeze([np.mean(self.bins[dim_idx][l[choice_idx]:l[choice_idx]+2]) for dim_idx, l in enumerate(idxs)])
#         print('sample', xs, max_val)
        return xs, max_val
