import sympy as sp
import tensorflow as tf
import tensorflow.python.keras.backend as K
import tensorflow_probability as tfp

import MLGeometry
from src.MLGeometry import bihomoNN as bnn
import math
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)
tf.random.set_seed(42)

z0, z1, z2, z3, z4 = sp.symbols('z0, z1, z2, z3, z4')
Z = [z0, z1, z2, z3, z4]
f = z0 ** 5 + z1 ** 5 + z2 ** 5 + z3 ** 5 + z4 ** 5 + 0.5 * z0 * z1 * z2 * z3 * z4

n_pairs = 1000
HS_train = MLGeometry.hypersurface.Hypersurface(Z, f, n_pairs)
HS_test = MLGeometry.hypersurface.Hypersurface(Z, f, n_pairs)
HS_train.list_patches()

train_set = MLGeometry.tf_dataset.generate_dataset(HS_train)
test_set = MLGeometry.tf_dataset.generate_dataset(HS_test)
train_set = train_set.shuffle(HS_train.n_points).batch(500)
test_set = test_set.shuffle(HS_test.n_points).batch(500)
points, Omega_Omegabar, mass, restriction = next(iter(train_set))
print(points)


class Kahler_potential(tf.keras.Model):
    def __init__(self):
        super(Kahler_potential, self).__init__()
        # The first layer transforms the complex points to the bihomogeneous form.
        # The number of the outputs is d^2, where d is the number of coordinates.
        self.bihomogeneous = bnn.Bihomogeneous(d=5)
        self.layer1 = bnn.SquareDense(5 ** 2, 70, activation=tf.square)
        self.layer2 = bnn.SquareDense(70, 100, activation=tf.square)
        self.layer3 = bnn.SquareDense(100, 1)

    def call(self, inputs):
        x = self.bihomogeneous(inputs)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = tf.math.log(x)
        return x

model = Kahler_potential()

@tf.function
def volume_form(points, Omega_Omegabar, mass, restriction):
    kahler_metric = MLGeometry.complex_math.complex_hessian(tf.math.real(model(points)), points)
    volume_form = tf.matmul(restriction, tf.matmul(kahler_metric, restriction, adjoint_b=True))
    volume_form = tf.math.real(tf.linalg.det(volume_form))

    # Calculate the normalization constant to make the overall integration as 1
    # It is a batchwise calculation but we expect it to converge to a constant eventually
    weights = mass / tf.reduce_sum(mass)
    factor = tf.reduce_sum(weights * volume_form / Omega_Omegabar)

    return volume_form / factor

optimizer = tf.keras.optimizers.Adam()
loss_func = MLGeometry.loss.weighted_MAPE

max_epochs = 500
epoch = 0
while epoch < max_epochs:
    epoch = epoch + 1
    for step, (points, Omega_Omegabar, mass, restriction) in enumerate(train_set):
        with tf.GradientTape() as tape:
            det_omega = volume_form(points, Omega_Omegabar, mass, restriction)
            loss = loss_func(Omega_Omegabar, det_omega, mass)
            grads = tape.gradient(loss, model.trainable_weights)
        optimizer.apply_gradients(zip(grads, model.trainable_weights))
    if epoch % 50 == 0:
        print("epoch %d: loss = %.5f" % (epoch, loss))