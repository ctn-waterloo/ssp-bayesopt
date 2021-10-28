import numpy as np

# BLR code written under the influence of 
# https://maxhalford.github.io/blog/bayesian-linear-regression/

class BayesianLinearRegression:
    def __init__(self, size_in, size_out=1, alpha=0.01, beta=None):
        self.input_dim = size_in
        self.output_dim = size_out
        self.S_inv = alpha * np.eye(size_in)
        self.m = np.zeros((size_in,1))
        self.beta = beta
        self.S = np.linalg.pinv(self.S_inv)
    ### end __init__

    def update(self, phis:np.ndarray, ts:np.ndarray):
        assert phis.shape[1] == self.input_dim, f'Expected input shape ({ts.shape[0]}, {self.input_dim}), got {phis.shape}'
        assert len(ts.shape) > 1 and ts.shape[1] == self.output_dim, f'Expected output shape ({phis.shape[0]}, 1), got {ts.shape}'

        if self.beta is None:
            self.beta = 1. / np.var(ts)
        S_inv = self.S_inv + self.beta * np.dot(phis.T, phis)
        S = np.linalg.pinv(S_inv)
#         x = self.beta * np.dot(self.S, np.dot(phis.T, ts))
#         self.m += x
        x = self.beta * np.dot(phis.T, ts)
        assert x.shape == (self.input_dim, 1), f'Mean update should be shape {self.input_dim, 1} was {x.shape}'

        
        self.m = S @ (self.S_inv @ self.m + x)

        self.S_inv = S_inv
        self.S = S

        assert self.m.shape[0] == self.input_dim and self.m.shape[1] == 1

    def predict(self, phi):
#         var = (1. / self.beta) + np.dot(phi, np.dot(self.S, phi.T))
#         return np.dot(self.m.T, phi.T), np.diag(var)
        var = (1. / self.beta) + np.einsum('ij,ij->i', phi, np.dot(phi, self.S.T))
        return np.dot(self.m.T, phi.T), var
