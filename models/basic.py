import tensorflow as tf


class FullConnected(tf.keras.Model):
    def __init__(self, units_latent, units_output, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._dense_latent = tf.keras.layers.Dense(units=units_latent)
        self._dense_output = tf.keras.layers.Dense(units=units_output, activation=tf.keras.activations.softmax)

    def call(self, tensor_input, **kwargs):
        out = self._dense_latent(tensor_input)
        out = self._dense_output(out)
        return out
