import tensorflow as tf
from tensorflow.compat.v1.lite.experimental.nn import TFLiteLSTMCell
from tensorflow.compat.v1.lite.experimental.nn import dynamic_rnn as lite_dynamic_rnn
from tensorflow.compat.v1.nn import static_rnn, dynamic_rnn
from tensorflow.compat.v1.nn.rnn_cell import LSTMCell

from typing import TypeVar


class RNN:
    def output(self, features, sequence_length=None):
        raise NotImplementedError()


RNN.Class = TypeVar("Class", bound=RNN)


class StaticLSTM(RNN):
    def __init__(self, latent_units):
        self._cell = LSTMCell(latent_units)

    def output(self, features, sequence_length=None):
        input_lstm = tf.unstack(features, axis=1)
        output, state = static_rnn(self._cell, input_lstm, dtype=tf.float32, sequence_length=sequence_length)
        output = tf.stack(output, axis=1)

        return output


class DynamicLSTM(RNN):
    def __init__(self, latent_units):
        self._cell = LSTMCell(latent_units)

    def output(self, features, sequence_length=None):
        output, _ = dynamic_rnn(self._cell, features, dtype=tf.float32, sequence_length=sequence_length)

        return output


class LiteDynamicLSTM(RNN):
    def __init__(self, latent_units):
        self._cell = TFLiteLSTMCell(latent_units)

    def output(self, features, sequence_length=None):
        input_lstm = tf.transpose(features, [1, 0, 2])
        output, _ = lite_dynamic_rnn(self._cell, input_lstm, dtype=tf.float32, sequence_length=sequence_length)
        output = tf.transpose(output, [1, 0, 2])

        return output


class CustomLSTM(RNN):
    def __init__(self):
        pass

    def output(self, features, sequence_length=None):
        raise NotImplementedError
