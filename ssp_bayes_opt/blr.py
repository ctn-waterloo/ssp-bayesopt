import numpy as np

# BLR code written under the influence of 
# https://maxhalford.github.io/blog/bayesian-linear-regression/

class BayesianLinearRegression:
    def __init__(self, size_in, alpha=0.01, beta=None):
        self.S_inv = alpha * np.eye(size_in)
        self.m = np.zeros((size_in,1))
        self.beta = beta
        self.S = np.linalg.pinv(self.S_inv)
    ### end __init__

    def update(self, phis, ts):
        if self.beta is None:
            self.beta = 1. / np.var(ts)
        S_inv = self.S_inv + self.beta * np.dot(phis.T, phis)
        S = np.linalg.pinv(S_inv)
#         x = self.beta * np.dot(self.S, np.dot(phis.T, ts))
#         self.m += x
        x = self.beta * np.dot(phis.T, ts)

        
        self.m = S @ (self.S_inv @ self.m + x)

        self.S_inv = S_inv
        self.S = S

    def predict(self, phi):
#         var = (1. / self.beta) + np.dot(phi, np.dot(self.S, phi.T))
#         return np.dot(self.m.T, phi.T), np.diag(var)
        var = (1. / self.beta) + np.einsum('ij,ij->i', phi, np.dot(phi, self.S.T))
        return np.dot(self.m.T, phi.T), var
