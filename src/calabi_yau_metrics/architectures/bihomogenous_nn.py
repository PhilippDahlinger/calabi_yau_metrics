import tensorflow as tf
from MLGeometry import bihomoNN as bnn


class BihomogenousNN(tf.keras.Model):
    def __init__(self):
        super(BihomogenousNN, self).__init__()
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