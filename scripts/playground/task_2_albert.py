# -*- coding: utf-8 -*-
"""spectral_mlgeom.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1BKgYZ1QfcbBDhwnTA1PAnyschEUJrWQk
"""

from google.colab import drive
drive.mount('/content/drive')

!pip install sympy matplotlib ipykernel

"""# MLGeometry guide

This introduction demonstrates how to use MLGeometry to:
1. Generate a hypersurface.
2. Build a bihomogeneous neural network.
3. Use the model to compute numerical Calabi-Yau metrics with the embedding method.
4. Plot $\eta$ on a rational curve.

## Configure imports

Import tensorflow_probability to use the L-BFGS optimizer:
"""

import sympy as sp
import tensorflow as tf
import tensorflow.python.keras.backend as K
import tensorflow_probability as tfp

cd /content/drive/MyDrive/CalabiYau/MLGeometry

#import MLGeometry as
import os
print(os.getcwd())
from MLGeometry import spectralNN as bnn
from MLGeometry import hypersurface
from MLGeometry import cicyhypersurface
from MLGeometry import bihomoNN
from MLGeometry import lbfgs
from MLGeometry import loss
from MLGeometry import tf_dataset
from MLGeometry import complex_math

import MLGeometry as mlg

"""Import the libraries to plot the $\eta$ on the rational curve (see the last section):"""

import math
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

"""## Set a random seed (optional)
Some random seed might be bad for numerical calulations. If there are any errors during the training, you may want to try a different seed.
"""

np.random.seed(42)
tf.random.set_seed(42)

"""## Define a hypersurface
First define a set of coordinates and a function as sympy symbols:
"""

z0, z1, z2, z3, z4 = sp.symbols('z0, z1, z2, z3, z4')
Z = [z0,z1,z2,z3,z4]
f = z0**5 + z1**5 + z2**5 + z3**5 + z4**5 + 0.5*z0*z1*z2*z3*z4

"""Then define a hypersurface as a collection of points which solve the equation f = 0, using the `Hypersurface` class in the `mlg.hypersurface` module. The parameter n_pairs is the number of random pairs of points used to form the random lines in $\mathbf{CP}^{N+1}$. Then we take the intersections of those random lines and the hypersurface. By Bezout's theorem, each line intersects the hypersurface in precisely d points where d is the number of homogeneous coordinates. So the total number of points is d * n_pairs."""

n_pairs = 250
HS_train = mlg.hypersurface.Hypersurface(Z, f, n_pairs)
HS_test = mlg.hypersurface.Hypersurface(Z, f, n_pairs)

"""The Hypersurface class will take care of the patchwork automatically. Let's use the `list_patches` function to check the number of points on each patch:"""

HS_train.list_patches()

HS_test.list_patches()

"""You can also invoke this method on one of the patches to check the distribution on the subpatches:"""

HS_train.patches[0].list_patches()

"""The Hypersurface class contains some symbolic and numerical methods as well, which will be introduced elsewhere.

## Training with Tensorflow
The following steps are similar to a regular Tensorflow training process.
### Generate datasets
The `mlg.tf_dataset.generate_dataset` function converts a hypersurface to a Tensorflow Dataset, which has four componets: the points on the hypersurface, the volume form $\small \Omega \wedge \bar\Omega$, the mass reweighting the points distribution and the restriction which restricts the Kähler metric to a subpatch. The restriction contains an extra linear transformation so that points on different affine patches can all be processed in one call. It is also possible to generate a dataset only on one affine patch.
"""

train_set = mlg.tf_dataset.generate_dataset(HS_train)
test_set = mlg.tf_dataset.generate_dataset(HS_test)

"""Shuffle and batch the datasets:"""

train_set = train_set.shuffle(HS_train.n_points).batch(25)
test_set = test_set.shuffle(HS_test.n_points).batch(25)

"""Let's look at what is inside a dataset:"""

points, Omega_Omegabar, mass, restriction, FS_metric = next(iter(train_set))
print(points.shape)

print(restriction.shape)

"""### Build a bihomogeneous neural network

The `mlg.bihomoNN` module provides the necessary layers (e.g. `Bihomogeneous` and `Dense` ) to construct the Kähler potential with a bihomogeneous neural network. Here is an example of a two-hidden-layer network (k = 4) with 70 and 100 hidden units:
"""

class Kahler_potential(tf.keras.Model):
    def __init__(self):
        super(Kahler_potential, self).__init__()
        # The first layer transforms the complex points to the bihomogeneous form.
        # The number of the outputs is d^2, where d is the number of coordinates.
        self.spectral = bnn.Spectral(d=5)
        self.layer1 = bnn.SquareDense(50, 70, activation=tf.math.softplus)
        self.layer2 = bnn.SquareDense(70, 100, activation=tf.math.softplus)
        self.layer3 = bnn.SquareDense(100, 1, activation=tf.math.softplus)

    def call(self, inputs):
        x = self.spectral(inputs)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x

model = Kahler_potential()

"""Define the Kähler metric $g_{i \bar j} = (g_{FS})_{i \bar j} + \partial_i\bar\partial_{\bar j} K$ and the volume form $d\mu_g = \det g_{i \bar j}$:"""

@tf.function
def volume_form(points, Omega_Omegabar, mass, restriction, FS_metric):

    kahler_metric =  mlg.complex_math.complex_hessian(tf.math.real(model(points)), points)
    #note - not actually Kahler metric, since we haven't added FS yet
    volume_form = FS_metric + tf.matmul(restriction, tf.matmul(kahler_metric, restriction, adjoint_b=True))
    volume_form = tf.math.real(tf.linalg.det(volume_form))

    # Calculate the normalization constant to make the overall integration as 1
    # It is a batchwise calculation but we expect it to converge to a constant eventually
    weights = mass / tf.reduce_sum(mass)
    factor = tf.reduce_sum(weights * volume_form / Omega_Omegabar)

    return volume_form / factor

"""### Train the model with Adam and L-BFGS
#### Adam
Setup the keras optmizer as `Adam` and the loss function as one of weighted loss in the `mlg.loss` module. Some available functions are `weighted_MAPE`, `weighted_MSE`, `max_error` and `MAPE_plus_max_error`. They are weighted with the mass formula since the points on the hypersurface are distributed according to the Fubini-Study measure while the measure used in the integration is determined by the volume form $\small \Omega \wedge \bar\Omega$.
"""

optimizer = tf.keras.optimizers.Adam()
loss_func = mlg.loss.weighted_MAPE

"""Loop over the batches and train the network:"""

max_epochs = 300
epoch = 0
while epoch < max_epochs:
    epoch = epoch + 1
    for step, (points, Omega_Omegabar, mass, restriction, FS_metric) in enumerate(train_set):
        with tf.GradientTape() as tape:
            det_omega = volume_form(points, Omega_Omegabar, mass, restriction, FS_metric)
            loss = loss_func(Omega_Omegabar, det_omega, mass)
            grads = tape.gradient(loss, model.trainable_weights)
        optimizer.apply_gradients(zip(grads, model.trainable_weights))
    if epoch % 50 == 0:
        print("epoch %d: loss = %.5f" % (epoch, loss))

"""Let's check the loss of the test dataset. First define a function to calculate the total loss over the whole dataset:"""

def cal_total_loss(dataset, loss_function):
    total_loss = tf.constant(0, dtype=tf.float32)
    total_mass = tf.constant(0, dtype=tf.float32)

    for step, (points, Omega_Omegabar, mass, restriction) in enumerate(dataset):
        det_omega = volume_form(points, Omega_Omegabar, mass, restriction)
        mass_sum = tf.reduce_sum(mass)
        total_loss += loss_function(Omega_Omegabar, det_omega, mass) * mass_sum
        total_mass += mass_sum
    total_loss = total_loss / total_mass

    return total_loss.numpy()

"""Check the results of MAPE and MSE:"""

sigma_test = cal_total_loss(test_set, mlg.loss.weighted_MAPE)
E_test = cal_total_loss(test_set, mlg.loss.weighted_MSE)
print("sigma_test = %.5f" % sigma_test)
print("E_test = %.5f" % E_test)

"""You can also check the error of the Monte Carlo integration, estimated by:

$$\delta \sigma = \frac{1}{\sqrt{N_p}} {\left( \int_X (|\eta - 1_X| - \sigma)^2 d\mu_{\Omega}\right)}^{1/2},$$

where $N_p$ is the number of points on the hypersurface and $\sigma$ is the `weighted_MAPE` loss, and

$$\eta = \frac{\det \omega}{\small \Omega \wedge \bar \Omega}$$
"""

def delta_sigma_square_test(y_true, y_pred, mass):
    weights = mass / K.sum(mass)
    return K.sum((K.abs(y_true - y_pred) / y_true - sigma_test)**2 * weights)

delta_sigma = cal_total_loss(test_set, delta_sigma_square_test)
print("delta_sigma = %.5f" % delta_sigma)

"""#### Save and Load
The trained network can be saved by:
"""

model.save('trained_model/70_100_1')

"""And loaded by the `load_model` method:"""

model = tf.keras.models.load_model('trained_model/70_100_1', compile=False)

"""#### L-BFGS
As elaborated in our paper, when the network getting more complicated, L-BFGS converges faster than Adam near the minima. It is recommanded to use it after pretraining with Adam. However, L-BFGS is not in the standard Tensorflow library so the training process is slightly different: (Only ~20 iterations are shown here. In a real case you may want to set the `max_epochs` to ~1000)
"""

# The displayed max_epochs will be three to four times this value since iter + 1 everytime the function
# is invoked, which also happens during the evaluation of the function itself and its gradient
max_epochs = 5

# Setup the function to be optimized by L-BFGS

train_func = mlg.lbfgs.function_factory(model, loss_func, train_set)

# Setup the inital values and train
init_params = tf.dynamic_stitch(train_func.idx, model.trainable_variables)
results = tfp.optimizer.lbfgs_minimize(value_and_gradients_function=train_func,
                                        initial_position=init_params,
                                        max_iterations=max_epochs)
# Update the model after the last loop
train_func.assign_new_model_parameters(results.position)

"""Note that the definition of the volume form is already in the `mlg.lbfgs` module. Also note that the standard L-BFGS does not support multi-batch training. You can still batch the dataset in case the GPU is out of memory, but the parameters are only updated after a whole epoch.

You can also check the test dataset:
"""

sigma_test = cal_total_loss(test_set, mlg.loss.weighted_MAPE)
E_test = cal_total_loss(test_set, mlg.loss.weighted_MSE)
print("sigma_test = %.5f" % sigma_test)
print("E_test = %.5f" % E_test)

"""#### Print out the metrics
After all of the trainings are done, the final results for the metrics can be printed out explicitly, using the previously generated data points and restriction matrices:
"""

@tf.function
def get_cy_metric(points, restriction):

    cy_metric = mlg.complex_math.complex_hessian(tf.math.real(model(points)), points)
    cy_metric = tf.matmul(restriction, tf.matmul(cy_metric, restriction, adjoint_b=True))

    return cy_metric

cy_metric = get_cy_metric(points, restriction)
print(points[5].numpy())
print(cy_metric[5].numpy())

"""### $\eta$ on the rational curve

Now let's retrict our model to a subspace and check the local behavior of $\eta$. With the quintic 3-fold f = 0, we can choose the embedding

$$(z_0, -z_0, z_1, 0, -z_1),$$

and the local coordinate system defined by $t = z_1 / z_0$. Using shperical coordinates $(\theta, \phi)$, it can be embedded into $\mathbb{R}^3$ by:

$$z_0 = \sin \theta \cos \phi, \qquad z_1= \sin \theta \sin \phi + i \cos \phi$$

So first sample the points on the rational curve:
"""

theta, phi = np.linspace(0.001,np.pi+0.001, 400), np.linspace(0.001, 2*np.pi+0.001, 400)
eps = 0.0001 + 0.0001j

R = []
points_list = []
for j in phi:
    for i in theta:
        t = complex(math.sin(i)*math.sin(j), math.cos(i)) / (math.sin(i)*math.cos(j))
        if np.absolute(t) <= 1:
            # The Bihomogeneous layer will remove the zero entries automatically.
            # So here we add a small number eps to avoid being removed
            points_list.append([1+eps, -1+eps, t+eps, 0+eps, -t+eps])
        else:
            # Use the symmetry:
            points_list.append([1+eps, -1+eps, 1/t+eps, 0+eps, -1/t+eps])

"""Use this set of points to generate the rational curve with norm_coordinate = z0 and max_grad_coordinate = z1:"""

rc = mlg.hypersurface.Hypersurface(Z, f, points=points_list, norm_coordinate=0, max_grad_coordinate=0)
rc_dataset = mlg.tf_dataset.generate_dataset(rc).batch(rc.n_points)

"""Calculate $\eta$:"""

points, Omega_Omegabar, mass, restriction = next(iter(rc_dataset))
det_omega = volume_form(points, Omega_Omegabar, mass, restriction)
eta = (det_omega / Omega_Omegabar).numpy()

"""Convert to Cartesian coordinates:"""

R = eta.reshape(400, 400)
THETA, PHI = np.meshgrid(theta, phi)
X = R * np.sin(THETA) * np.cos(PHI)
Y = R * np.sin(THETA) * np.sin(PHI)
ZZ = R * np.cos(THETA)

"""Plot the figure:"""

fig = plt.figure()
ax = fig.add_subplot(1,1,1, projection='3d')
ax.set_zlim3d(-1.0, 1.0)
plot = ax.plot_surface(
    X, Y, ZZ, rstride=1, cstride=1, cmap=plt.cm.YlGnBu_r,
    linewidth=0, antialiased=False)

"""$\eta$ is expected to approach the constant function 1 as k increases."""