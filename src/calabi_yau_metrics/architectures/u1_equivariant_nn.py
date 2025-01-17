from MLGeometry.u1_equivariant_layer import U1EquivariantLayer
import tensorflow as tf

kINPUT_DIM = 10


class u1_model_relu(tf.keras.Model):

    def __init__(self, n_units, m_units, g_steps=8):
        super().__init__()
        assert len(n_units) > 0
        assert len(m_units) > 0

        inner_layers = tf.keras.Sequential([
            tf.keras.layers.Dense(
                n_units[i],
                input_shape=((kINPUT_DIM,) if i == 0 else (n_units[i - 1],)),
                activation=('relu' if i < len(n_units) - 1 else None),
                name=f'inner_dense_{i}'
            )
            for i in range(len(n_units))
        ])

        self.u1_layer = U1EquivariantLayer(inner_layers, g_steps=g_steps)

        self.outer_layers = tf.keras.Sequential([
            tf.keras.layers.Dense(
                m_units[i],
                activation=('relu' if i < len(m_units) - 1 else None),
                name=f'outer_dense_{i}'
            )
            for i in range(len(m_units))
        ])

    def call(self, inputs):

        x = self.u1_layer(inputs)
        x = self.outer_layers(x)
        return x
