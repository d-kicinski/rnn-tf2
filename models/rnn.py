import tensorflow as tf


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


class LiteLSTM:
    # CONFIG = tf.ConfigProto(device_count={"GPU": 0})

    def __init__(self, latent_units, vocab_size, embeddings_dim, class_size, *args, **kwargs):
        self._latent_units = latent_units
        self._vocab_size = vocab_size
        self._embeddings_dim = embeddings_dim
        self._class_size = class_size

    def __call__(self, indices, *args, **kwargs):
        # Weights and biases for output softmax layer.
        # input image placeholder

        lstm_inputs = tf.transpose(indices, [1, 0, 2])
        cell_fw = tf.lite.experimental.nn.TFLiteLSTMCell(self._latent_units)
        cell_bw = tf.lite.experimental.nn.TFLiteLSTMCell(self._latent_units)

        outputs_fw, _ = tf.lite.experimental.nn.dynamic_rnn(cell_fw, lstm_inputs)

        return outputs_fw
