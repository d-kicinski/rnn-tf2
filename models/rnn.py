import tensorflow as tf
from tensorflow.compat.v1.lite.experimental.nn import TFLiteLSTMCell
from tensorflow.compat.v1.lite.experimental.nn import dynamic_rnn as lite_dynamic_rnn
from tensorflow.compat.v1.nn import static_rnn
from tensorflow.compat.v1.nn.rnn_cell import LSTMCell


# from tensorflow.compat.v1.lite.experimental.nn import TFLiteLSTMCell, dynamic_rnn


# class LiteKerasLSTM(tf.keras.Model):
#     def __init__(self, latent_units, vocab_size, embeddings_dim, class_size, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#
#         self.embedding = tf.keras.layers.Embedding(vocab_size, embeddings_dim)
#         self.dense_output = tf.keras.layers.Dense(class_size, activation=tf.keras.activations.softmax)
#         self.cell = TFLiteLSTMCell(num_units=latent_units)
#
#     def call(self, indices, **kwargs):
#         encodings = self.embedding(indices)
#         outputs, state = dynamic_rnn(self.cell, encodings, sequence_length=None)
#         probabilities = self.dense_output(outputs)
#
#         return probabilities


# class KerasLSTM(tf.keras.Model):
#     def __init__(self, latent_units, vocab_size, embeddings_dim, class_size, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#
#         self.embedding = tf.keras.layers.Embedding(vocab_size, embeddings_dim)
#         self.bilstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(latent_units, return_sequences=True))
#         self.dense_class = tf.keras.layers.TimeDistributed(
#             tf.keras.layers.Dense(class_size, activation=tf.keras.activations.softmax))
#
#     def call(self, indices, **kwargs):
#         x = self.embedding(indices)
#         x = self.bilstm(x)
#         x = self.dense_class(x)
#         return x


class StaticLSTM:
    def __init__(self, latent_units):
        self._cell = LSTMCell(latent_units)

    def output(self, features, sequence_length=None):
        input_lstm = tf.unstack(features, axis=1)
        output, state = static_rnn(self._cell, input_lstm, dtype=tf.float32, sequence_length=sequence_length)
        output = tf.stack(output, axis=1)

        return output


class DynamicLSTM:
    def __init__(self, latent_units):
        self._cell = TFLiteLSTMCell(latent_units)

    def output(self, features, sequence_length=None):
        input_lstm = tf.transpose(features, [1, 0, 2])
        output, _ = lite_dynamic_rnn(self._cell, input_lstm, dtype=tf.float32, sequence_length=sequence_length)
        output = tf.transpose(output, [1, 0, 2])

        return output


class CustomLSTM:
    def __init__(self, latent_units):
        pass

    def __call__(self, features):
        pass
