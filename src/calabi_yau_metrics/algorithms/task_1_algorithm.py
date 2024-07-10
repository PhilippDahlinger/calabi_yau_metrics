import MLGeometry
from calabi_yau_metrics.architectures.bihomogenous_nn import BihomogenousNN
import tensorflow as tf


class Task1Algorithm:
    def __init__(self, config, env):
        self.config = config
        self.env = env
        if self.config.network_structure == "guide":
            layers_config = [
                {"input_dim": 5 ** 2, "output_dim": 70},
                {"input_dim": 70, "output_dim": 100},
                {"input_dim": 100, "output_dim": 1},
            ]
        elif self.config.network_structure == "3_layers":
            layers_config = [
                {"input_dim": 5 ** 2, "output_dim": 30},
                {"input_dim": 30, "output_dim": 70},
                {"input_dim": 70, "output_dim": 100},
                {"input_dim": 100, "output_dim": 1},
            ]
        elif self.config.network_structure == "4_layers":
            layers_config = [
                {"input_dim": 5 ** 2, "output_dim": 30},
                {"input_dim": 30, "output_dim": 50},
                {"input_dim": 50, "output_dim": 70},
                {"input_dim": 70, "output_dim": 100},
                {"input_dim": 100, "output_dim": 1},
            ]
        elif self.config.network_structure == "5_layers":
            layers_config = [
                {"input_dim": 5 ** 2, "output_dim": 30},
                {"input_dim": 30, "output_dim": 50},
                {"input_dim": 50, "output_dim": 70},
                {"input_dim": 70, "output_dim": 100},
                {"input_dim": 100, "output_dim": 150},
                {"input_dim": 150, "output_dim": 1},
            ]
        elif self.config.network_structure == "thick":
            layers_config = [
                {"input_dim": 5 ** 2, "output_dim": 70},
                {"input_dim": 70, "output_dim": 150},
                {"input_dim": 150, "output_dim": 250},
                {"input_dim": 250, "output_dim": 1},
            ]
        else:
            raise ValueError("Invalid network structure")
        # batchnorm and dropout
        for layer in layers_config:
            layer["batchnorm"] = self.config.batchnorm
            layer["dropout_rate"] = self.config.dropout_rate

        self.model = BihomogenousNN(layers_config)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=config.learning_rate)
        self.training_mode = False

    @tf.function
    def volume_form(self, points, Omega_Omegabar, mass, restriction):
        kahler_metric = MLGeometry.complex_math.complex_hessian(
            tf.math.real(self.model(points, training=self.training_mode)), points)
        volume_form = tf.matmul(restriction, tf.matmul(kahler_metric, restriction, adjoint_b=True))
        volume_form = tf.math.real(tf.linalg.det(volume_form))

        # Calculate the normalization constant to make the overall integration as 1
        # It is a batchwise calculation but we expect it to converge to a constant eventually
        weights = mass / tf.reduce_sum(mass)
        factor = tf.reduce_sum(weights * volume_form / Omega_Omegabar)

        return volume_form / factor

    def single_train_step(self, points, Omega_Omegabar, mass, restriction):
        self.training_mode = True
        with tf.GradientTape() as tape:
            det_omega = self.volume_form(points, Omega_Omegabar, mass, restriction)
            loss = MLGeometry.loss.weighted_MAPE(Omega_Omegabar, det_omega, mass)
            grads = tape.gradient(loss, self.model.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))
        return loss

    def calc_total_loss(self, dataset, loss_function):
        # this is used for testing
        self.training_mode = False
        total_loss = tf.constant(0, dtype=tf.float32)
        total_mass = tf.constant(0, dtype=tf.float32)

        for step, (points, Omega_Omegabar, mass, restriction) in enumerate(dataset):
            det_omega = self.volume_form(points, Omega_Omegabar, mass, restriction)
            mass_sum = tf.reduce_sum(mass)
            total_loss += loss_function(Omega_Omegabar, det_omega, mass) * mass_sum
            total_mass += mass_sum
        total_loss = total_loss / total_mass

        return total_loss.numpy()
