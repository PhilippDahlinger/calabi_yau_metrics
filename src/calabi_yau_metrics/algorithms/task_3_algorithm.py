import MLGeometry
from calabi_yau_metrics.algorithms.abstract_algorithm import AbstractAlgorithm
from calabi_yau_metrics.architectures.bihomogenous_nn import BihomogenousNN
import tensorflow as tf

from calabi_yau_metrics.architectures.spectral_nn import SpectralNN
from calabi_yau_metrics.architectures.u1_equivariant_nn import u1_model_relu


class Task3Algorithm(AbstractAlgorithm):

    def __init__(self, config, env):
        super().__init__(config, env)

    def get_model(self):
        if self.config.network_structure == "guide":
            n_units = [16, 16, 16]
            m_units = [128, 128, 128, 1]
            g_steps = 8
        else:
            raise ValueError("Invalid network structure")

        return u1_model_relu(n_units, m_units, g_steps=g_steps)

    def get_omega_mass(self, batch):
        points, Omega_Omegabar, mass, restriction, FS_metric = batch
        return Omega_Omegabar, mass

    @tf.function
    def volume_form(self, batch):
        points, Omega_Omegabar, mass, restriction, FS_metric = batch
        kahler_metric = MLGeometry.complex_math.complex_hessian(tf.math.real(self.model(points)), points)
        # note - not actually Kahler metric, since we haven't added FS yet
        volume_form = FS_metric + tf.matmul(restriction, tf.matmul(kahler_metric, restriction, adjoint_b=True))
        volume_form = tf.math.real(tf.linalg.det(volume_form))

        # Calculate the normalization constant to make the overall integration as 1
        # It is a batchwise calculation but we expect it to converge to a constant eventually
        weights = mass / tf.reduce_sum(mass)
        factor = tf.reduce_sum(weights * volume_form / Omega_Omegabar)

        return volume_form / factor

