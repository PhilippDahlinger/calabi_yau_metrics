import MLGeometry
from calabi_yau_metrics.algorithms.abstract_algorithm import AbstractAlgorithm
from calabi_yau_metrics.architectures.bihomogenous_nn import BihomogenousNN
import tensorflow as tf

from calabi_yau_metrics.architectures.spectral_nn import SpectralNN


class Task2Algorithm(AbstractAlgorithm):

    def __init__(self, config, env):
        super().__init__(config, env)

    def get_model(self):
        if self.config.network_structure == "guide":
            layers_config = [
                {"input_dim": 50, "output_dim": 70},
                {"input_dim": 70, "output_dim": 100},
                {"input_dim": 100, "output_dim": 1},
            ]
        elif self.config.network_structure == "3_layers":
            layers_config = [
                {"input_dim": 50, "output_dim": 30},
                {"input_dim": 30, "output_dim": 70},
                {"input_dim": 70, "output_dim": 100},
                {"input_dim": 100, "output_dim": 1},
            ]
        elif self.config.network_structure == "4_layers":
            layers_config = [
                {"input_dim": 50, "output_dim": 64},
                {"input_dim": 64, "output_dim": 128},
                {"input_dim": 128, "output_dim": 256},
                {"input_dim": 256, "output_dim": 512},
                {"input_dim": 512, "output_dim": 1},
            ]
        elif self.config.network_structure == "5_layers":
            layers_config = [
                {"input_dim": 50, "output_dim": 30},
                {"input_dim": 30, "output_dim": 30},
                {"input_dim": 30, "output_dim": 30},
                {"input_dim": 30, "output_dim": 70},
                {"input_dim": 70, "output_dim": 100},
                {"input_dim": 100, "output_dim": 1},
            ]
        elif self.config.network_structure == "thick":
            layers_config = [
                {"input_dim": 50, "output_dim": 70},
                {"input_dim": 70, "output_dim": 150},
                {"input_dim": 150, "output_dim": 250},
                {"input_dim": 250, "output_dim": 1},
            ]
        elif self.config.network_structure == "thicker":
            layers_config = [
                {"input_dim": 50, "output_dim": 64},
                {"input_dim": 64, "output_dim": 256},
                {"input_dim": 256, "output_dim": 1024},
                {"input_dim": 1024, "output_dim": 1},
            ]
        elif self.config.network_structure == "thickerx2":
            layers_config = [
                {"input_dim": 50, "output_dim": 64},
                {"input_dim": 64, "output_dim": 512},
                {"input_dim": 512, "output_dim": 2048},
                {"input_dim": 2048, "output_dim": 1},
            ]
        else:
            raise ValueError("Invalid network structure")
        # batchnorm and dropout
        for layer in layers_config:
            layer["batchnorm"] = self.config.batchnorm
            layer["dropout_rate"] = self.config.dropout_rate

        return SpectralNN(layers_config)

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

