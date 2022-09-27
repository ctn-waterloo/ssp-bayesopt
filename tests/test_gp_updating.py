import numpy as np
import matplotlib.pyplot as plt


X = np.linspace(start=0, stop=10, num=1_000).reshape(-1,1)
y = np.squeeze(X * np.sin(X))

plt.plot(X, y,label=r'$f(x) = x \sin(x)$', linestyle='dotted')
plt.legend()
plt.xlabel('$x$')
plt.ylabel('$f(x)$')
_ = plt.title('True Generative Process')

rng = np.random.RandomState(1)
training_indices = rng.choice(np.arange(y.size), size=6, replace=False)
X_train, y_train = X[training_indices], y[training_indices]

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern

kernel = Matern(nu=2.5)#, length_scale_bounds='fixed')

gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=9)
gp.fit(X_train, y_train)


kernel2 = Matern(nu=2.5,
                 length_scale=np.exp(gp.kernel_.theta),
                 length_scale_bounds='fixed')
gp2 = GaussianProcessRegressor(kernel=kernel2, n_restarts_optimizer=9)
gp2.fit(X, y)
print(gp2.kernel_)
print(dir(gp.kernel_))
print(np.exp(gp.kernel_.theta))
print(gp.kernel_.get_params())

