import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from ssp_bayes_opt import sspspace
import tensorflow as tf
import sklearn
from tensorflow import keras
from tensorflow.keras import layers, regularizers
import time

domain_dim=2
ssp_space = sspspace.HexagonalSSPSpace(domain_dim,
                        ssp_dim=217, 
                        n_rotates=6, 
                        n_scales=6, 
                        scale_min=2*np.pi/np.sqrt(6) - 0.5,
                        scale_max=2*np.pi/np.sqrt(6) + 0.5,
                        domain_bounds= np.tile([-1,1],(domain_dim,1)),
                        length_scale=0.1
                        )
start = time.time()
history = ssp_space.train_decoder_net(n_training_pts=200000,n_hidden_units = 8,
                      learning_rate=1e-3,n_epochs = 20)
end = time.time()
train_time = (end - start)


loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, 20 + 1)

plt.figure()
plt.plot(epochs, loss, 'r', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()

n_test = 25
test_ssps, test_points = ssp_space.get_sample_pts_and_ssps(num_points_per_dim=n_test,
                                                              method='grid')

#predict_pts = model.predict(test_ssps) 
# Can call decode with full matrix for method='network' (& its faster), but not for method='network-optim'
predict_pts = ssp_space.decode(test_ssps, method='network')
errors = np.sqrt(np.sum((predict_pts -test_points)**2, axis=1))

xs = np.linspace(-1,1,n_test)
ys = xs
X,Y = np.meshgrid(xs,ys)
plt.figure()
plt.pcolormesh(X,Y,errors.reshape((n_test,n_test)))
plt.colorbar()


direct_time = np.zeros(n_test)
netoptim_time = np.zeros(n_test)
net_time = np.zeros(n_test)
for i in range(n_test):
    start = time.time()
    ssp_space.decode(test_ssps, method='direct-optim');
    end = time.time()
    direct_time[i] = (end - start)
    
    start = time.time()
    ssp_space.decode(test_ssps, method='network-optim');
    end = time.time()
    netoptim_time[i] = (end - start)
    
    start = time.time()
    ssp_space.decode(test_ssps, method='network');
    end = time.time()
    net_time[i] = (end - start)

for t in [direct_time, netoptim_time, net_time]:
    print(np.mean(t))
    print(np.max(np.abs(np.mean(t)- t)))