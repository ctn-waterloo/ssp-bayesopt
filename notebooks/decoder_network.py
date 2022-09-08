import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from ssp_bayes_opt import sspspace
import tensorflow as tf
import sklearn
from tensorflow import keras
from tensorflow.keras import layers, regularizers

domain_dim=2
ssp_space = sspspace.HexagonalSSPSpace(domain_dim,
                        ssp_dim=151, 
                        n_rotates=5, 
                        n_scales=5, 
                        scale_min=2*np.pi/np.sqrt(6) - 0.5,
                        scale_max=2*np.pi/np.sqrt(6) + 0.5,
                        domain_bounds= np.tile([-1,1],(domain_dim,1)),
                        length_scale=0.1
                        )

history = ssp_space.train_decoder_net(n_training_pts=200000,n_hidden_units = 12,
                      learning_rate=1e-3,n_epochs = 20)

# def sampling_fun(n,d,seed=0.5):
#     def phi(d): 
#         x=2.0000 
#         for i in range(10): 
#           x = pow(1+x,1/(d+1)) 
#         return x
#     g = phi(d) 
#     alpha = np.zeros(d) 
#     for j in range(d): 
#       alpha[j] = pow(1/g,j+1) %1 
#     z = np.zeros((n, d)) 
#     for i in range(n):
#         z[i] = seed + alpha*(i+1)
#     z = z %1
#     return z

# sample_points = 2*(sampling_fun(200000, 2) -0.5)
# sample_ssps = ssp_space.encode(sample_points)
# # sample_ssps, sample_points = ssp_space.get_sample_pts_and_ssps(num_points_per_dim=400,
# #                                                               method='grid')

# model = keras.Sequential([
#      layers.Dense(ssp_space.ssp_dim, activation="relu", name="layer1"),# layers.Dropout(.1),
#      layers.Dense(12, activation="relu", name="layer2"), # kernel_regularizer=regularizers.L1L2(l1=1e-5, l2=1e-4)),
#      layers.Dense(ssp_space.domain_dim, name="output"),
#     ])

# model.compile(
#     optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
#     loss='mean_squared_error')


# shuffled_ssps, shuffled_pts = sklearn.utils.shuffle(sample_ssps, sample_points)
# history = model.fit(shuffled_ssps, shuffled_pts,
#     epochs=20,verbose=1, validation_split = 0.2)

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, 20 + 1)

plt.figure()
plt.plot(epochs, loss, 'r', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()

n_test = 100
test_ssps, test_points = ssp_space.get_sample_pts_and_ssps(num_points_per_dim=n_test,
                                                              method='grid')

#predict_pts = model.predict(test_ssps) 
# Can call decode with full matrix for method='network' (& its faster), but not for method='network-optim'
predict_pts = ssp_space.decode(test_ssps, method='network-optim')
errors = np.sqrt(np.sum((predict_pts -test_points)**2, axis=1))

xs = np.linspace(-1,1,n_test)
ys = xs
X,Y = np.meshgrid(xs,ys)
plt.figure()
plt.pcolormesh(X,Y,errors.reshape((n_test,n_test)))
plt.colorbar()