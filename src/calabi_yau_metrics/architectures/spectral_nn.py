import tensorflow as tf
from MLGeometry import bihomoNN as bnn


class SpectralNN(tf.keras.Model):
    def __init__(self, layers_config):
        super(SpectralNN, self).__init__()
        # The first layer transforms the complex points to the bihomogeneous form.
        # The number of the outputs is d^2, where d is the number of coordinates.
        self.spectral = bnn.Spectral(d=5)
        self.my_layers = []
        for layer_config in layers_config[:-1]:
            self.my_layers.append(
                bnn.SquareDense(layer_config["input_dim"], layer_config["output_dim"], activation=tf.math.softplus,
                                batchnorm=layer_config.get("batchnorm", False))
            )
            if layer_config.get("dropout_rate", 0) > 0:
                self.my_layers.append(tf.keras.layers.Dropout(layer_config["dropout_rate"]))
        self.output_layer = bnn.SquareDense(layers_config[-1]["input_dim"], layers_config[-1]["output_dim"],
                                            activation=None, batchnorm=False)

    def call(self, inputs, training=False):
        x = self.spectral(inputs)
        for layer in self.my_layers:
            x = layer(x, training=training)
        x = self.output_layer(x, training=training)
        return x
