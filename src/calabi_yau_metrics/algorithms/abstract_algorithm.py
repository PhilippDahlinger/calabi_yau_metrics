from abc import abstractmethod

import MLGeometry
from calabi_yau_metrics.architectures.bihomogenous_nn import BihomogenousNN
import tensorflow as tf


class AbstractAlgorithm:
    def __init__(self, config, env):
        self.config = config
        self.env = env
        self.model = self.get_model()
        if config.learning_rate_decay_steps is None:
            lr_schedule = config.learning_rate
        else:
            lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
                config.learning_rate,
                decay_steps=len(self.env.train_set) * config.learning_rate_decay_steps,
                decay_rate=0.96,
                staircase=True)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
        self.training_mode = False

    @abstractmethod
    def get_model(self):
        raise NotImplementedError()

    @tf.function
    def volume_form(self, batch):
        raise NotImplementedError()

    @abstractmethod
    def get_omega_mass(self, batch):
        raise NotImplementedError()

    def single_train_step(self, batch):
        self.training_mode = True
        with tf.GradientTape() as tape:
            det_omega = self.volume_form(batch)
            Omega_Omegabar, mass = self.get_omega_mass(batch)
            loss = MLGeometry.loss.weighted_MAPE(Omega_Omegabar, det_omega, mass)
            grads = tape.gradient(loss, self.model.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))
        return loss

    def calc_total_loss(self, dataset, loss_function):
        # this is used for testing
        self.training_mode = False
        total_loss = tf.constant(0, dtype=tf.float32)
        total_mass = tf.constant(0, dtype=tf.float32)

        for step, batch in enumerate(dataset):
            det_omega = self.volume_form(batch)
            Omega_Omegabar, mass = self.get_omega_mass(batch)

            mass_sum = tf.reduce_sum(mass)
            total_loss += loss_function(Omega_Omegabar, det_omega, mass) * mass_sum
            total_mass += mass_sum
        total_loss = total_loss / total_mass

        return total_loss.numpy()
