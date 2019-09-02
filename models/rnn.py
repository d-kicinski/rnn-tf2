from typing import TypeVar

import tensorflow as tf
import tensorflow.compat.v1 as tfc
from tensorflow import keras
from tensorflow.compat.v1.lite.experimental.nn import TFLiteLSTMCell
from tensorflow.compat.v1.lite.experimental.nn import dynamic_rnn as lite_dynamic_rnn
from tensorflow.compat.v1.nn import static_rnn, dynamic_rnn
from tensorflow.compat.v1.nn.rnn_cell import LSTMCell


class RNN:
    def output(self, features: tfc.placeholder, sequence_length: bool = None):
        raise NotImplementedError()


RNN.Class = TypeVar("Class", bound=RNN)


class StaticLSTM(RNN):
    def __init__(self, latent_units: int):
        self._cell = LSTMCell(latent_units)

    def output(self, features: tfc.placeholder, sequence_length: bool = None):
        input_lstm = tf.unstack(features, axis=1)
        output, state = static_rnn(self._cell, input_lstm, dtype=tf.float32, sequence_length=sequence_length)
        output = tf.stack(output, axis=1)

        return output


class DynamicLSTM(RNN):
    def __init__(self, latent_units: int):
        self._cell = LSTMCell(latent_units)

    def output(self, features: tfc.placeholder, sequence_length: bool = None):
        output, _ = dynamic_rnn(self._cell, features, dtype=tf.float32, sequence_length=sequence_length)

        return output


class LiteDynamicLSTM(RNN):
    def __init__(self, latent_units: int):
        self._cell = TFLiteLSTMCell(latent_units)

    def output(self, features: tfc.placeholder, sequence_length: bool = None):
        input_lstm = tf.transpose(features, [1, 0, 2])
        output, _ = lite_dynamic_rnn(self._cell, input_lstm, dtype=tf.float32, sequence_length=sequence_length)
        output = tf.transpose(output, [1, 0, 2])

        return output


class KerasLSTM(RNN):
    def __init__(self, latent_units: int, training: bool = False, *args, **kwargs):
        self._training = training
        cell = keras.layers.LSTMCell(latent_units)
        self._rnn = keras.layers.RNN(cell, unroll=self.unroll, return_sequences=True)

    @property
    def unroll(self):
        raise NotImplementedError()

    def output(self, features, sequence_length=None):
        output = self._rnn(features, training=self._training)

        return output


class KerasStaticLSTM(KerasLSTM):
    def __init__(self, latent_units: int, training: bool = False):
        super().__init__(latent_units, training)

    @property
    def unroll(self):
        return True


class KerasDynamicLSTM(KerasLSTM):
    def __init__(self, latent_units: int, training: bool = False, *args, **kwargs):
        super().__init__(latent_units, training, *args, **kwargs)

    @property
    def unroll(self):
        return False
