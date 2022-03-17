import numpy as np

# BLR code written under the influence of 
# https://maxhalford.github.io/blog/bayesian-linear-regression/

class BayesianLinearRegression:
    def __init__(self, size_in, size_out=1, alpha=0.01, beta=None):
        '''
        Create the Bayesian Linear Regression object.

        Parameters:
        -----------

        size_in : int
            The dimensionality of the input data

        size_out : int
            The dimensionality of the data that is being predicted.
            Currently has to be 1, implementing multi-regression TBD.

        alpha : float
            Value for initial prior distribution. Default value 0.01

        beta : float
            Estimate of 1/variance of the observations.  If none, will
            be computed from the observations provided in the "fit" 
            method.
        '''

        self.input_dim = size_in
        self.output_dim = size_out
        self.S_inv = alpha * np.eye(size_in)
        self.S = np.linalg.pinv(self.S_inv)

        self.m = np.zeros((size_in,1))
        self.beta = beta
        self.rank_one_updates = True
    ### end __init__

    def fit(self, phis:np.ndarray, ts:np.ndarray):
        '''
        Fit the Bayeisan Linear Regression.

        Performs initial estimate of the hyperparameter beta
        Covariance matrix is inverted using pseudoinverse as opposed
        to sequence of rank 1 updates.

        Parameters:
        -----------
        phis : np.ndarray
            The (num_datapoints, num_features) array to be regressed against 
            in some unknown feature representation

        ts : np.ndarray
            The (num_datapoints, 1) array of target values to model.
        '''
        assert phis.shape[1] == self.input_dim, f'Expected input shape ({ts.shape[0]}, {self.input_dim}), got {phis.shape}'
        assert len(ts.shape) > 1 and ts.shape[1] == self.output_dim, f'Expected output shape ({phis.shape[0]}, 1), got {ts.shape}'

        assert self.beta is None, 'Expected beta to be undefined. To update a distribution call the "update" method'

        self.beta = 1. / np.var(ts)
        S_inv = self.S_inv + self.beta * np.dot(phis.T, phis)
        S = np.linalg.pinv(S_inv)

        x = self.beta * np.dot(phis.T, ts)
        assert x.shape == (self.input_dim, 1), f'Mean update should be shape {self.input_dim, 1} was {x.shape}'
        
        self.m = S @ (self.S_inv @ self.m + x)
        self.S_inv = S_inv
        self.S = S

        pass

    def update(self, phis:np.ndarray, ts:np.ndarray):
        '''
        Compute one-step update of bayesian linear regression
        '''
        assert phis.shape[1] == self.input_dim, f'Expected input shape ({ts.shape[0]}, {self.input_dim}), got {phis.shape}'
        assert len(ts.shape) > 1 and ts.shape[1] == self.output_dim, f'Expected output shape ({phis.shape[0]}, 1), got {ts.shape}'

        if self.beta is None:
            self.beta = 1. / np.var(ts)
        S_inv = self.S_inv + self.beta * np.dot(phis.T, phis)
        
        S = np.copy(self.S)
        if self.rank_one_updates:
            for i in range(phis.shape[0]):
                phi = np.atleast_2d(phis[i,:])  * np.sqrt(self.beta)
                scale = (1 + phi @ S @ phi.T)
                S = S - (S @ phi.T @ phi @ S) / scale
        else:
            S = np.linalg.pinv(S_inv)
        x = self.beta * np.dot(phis.T, ts)
        assert x.shape == (self.input_dim, 1), f'Mean update should be shape {self.input_dim, 1} was {x.shape}'
        
        self.m = S @ (self.S_inv @ self.m + x)
        self.S_inv = S_inv
        self.S = S

        assert self.m.shape[0] == self.input_dim and self.m.shape[1] == 1

    def predict(self, phi):
        var = (1. / self.beta) + np.einsum('ij,ij->i', phi, np.dot(phi, self.S.T))
        return np.dot(self.m.T, phi.T), var

    def sample(self):
        phi_init = None
        try:
            phi_init = np.atleast_2d(np.random.multivariate_normal(self.m.flatten(), 
                                                                   self.S).reshape(-1,1)
                                    )
        except np.linalg.LinAlgError as e:
            print(e)
            phi_init = (-self.S_inv @ self.m).reshape(-1,1)
        assert phi_init.ndim == 2
        assert phi_init.shape[1]
        return phi_init
    ### end sample
### end class BayesianLinearRegression
