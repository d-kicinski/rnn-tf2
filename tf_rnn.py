import os
from itertools import chain
from time import time

os.environ['TF_ENABLE_CONTROL_FLOW_V2'] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import numpy as np
import tensorflow as tf
import tensorflow.compat.v1 as tfc

from dataclasses import dataclass
from typing import *

from models import rnn


@dataclass
class Shapes:
    batch = 10
    sequence = 30
    features = 50
    latent = 100


@dataclass
class Constants:
    epochs = 100
    max_sequence_length = 500


class TrackTime:
    def __init__(self, name):
        self._name = name
        self._time_begin = None
        self._time_end = None

    def __enter__(self):
        self._time_begin = time()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._time_end = time()
        print(f"{self._name} TIME: {self._time_end - self._time_begin:.6} [s]")


@dataclass
class Data:
    sequence_lengths: List
    features: List

    @staticmethod
    def dynamic():
        """Not padded data with different sequence lengths"""
        sequence_lengths = []
        features = []
        for seq_len in chain(range(100, 501, 100), range(500, 99, -100)):
            sequence_lengths.append([np.array([seq_len for _ in range(Shapes.batch)])
                                     for _ in range(Constants.epochs)])

            features.append([np.random.rand(Shapes.batch,
                                            seq_len,
                                            Shapes.features) for _ in range(Constants.epochs)])
        return Data(sequence_lengths, features)

    @staticmethod
    def static():
        """Padded data with different sequence lengths"""
        features = []
        sequence_lengths = []

        for seq_len in chain(range(100, 501, 100), range(500, 99, -100)):
            sequence_lengths.append([np.array([seq_len for _ in range(Shapes.batch)])
                                     for _ in range(Constants.epochs)])

            features.append([np.random.rand(Shapes.batch,
                                            Constants.max_sequence_length,
                                            Shapes.features) for _ in range(Constants.epochs)])
        return Data(sequence_lengths, features)


def run_dynamic_rnn(data: Data):
    with tfc.variable_scope("input"):
        features = tf.placeholder(shape=(Shapes.batch, None, Shapes.features), dtype=tf.float32)

    with tfc.variable_scope("dynamic_lstm"):
        lstm_dynamic = rnn.DynamicLSTM(latent_units=Shapes.latent).output(features)

    with tfc.Session() as sess:
        sess.run(tfc.global_variables_initializer())

        with TrackTime("[lstm_dynamic]\t TOTAL"):
            for seq_len, data in zip(data.sequence_lengths, data.features):
                with TrackTime(f"[lstm_dynamic]\t batch: {Shapes.batch}\t sequence_length: {seq_len}\t"):
                    for batch in data:
                        sess.run(lstm_dynamic, feed_dict={features: batch})


def run_static_rnn(data: Data, use_sequence_length_info=False):
    with tfc.variable_scope("input"):
        features = tfc.placeholder(shape=(Shapes.batch, Constants.max_sequence_length, Shapes.features),
                                   dtype=tf.float32, name="features")

        sequence_length = tfc.placeholder(shape=(Shapes.batch,), dtype=tf.float32, name="sequence_length")

    with tfc.variable_scope("model", reuse=tfc.AUTO_REUSE):
        lstm_static = rnn.StaticLSTM(latent_units=Shapes.latent) \
            .output(features, sequence_length=sequence_length if use_sequence_length_info else None)

    with tfc.Session() as sess:
        sess.run(tfc.global_variables_initializer())

        with TrackTime("[lstm_dynamic]\t TOTAL"):
            for seq_len, data in zip(data.sequence_lengths, data.features):
                with TrackTime(f"[lstm_static]\t batch: {Shapes.batch}\t sequence_length: {seq_len[0][0]}\t"):
                    for seq_len_b, data_b in zip(seq_len, data):
                        sess.run(lstm_static, feed_dict={features: data_b,
                                                         sequence_length: seq_len_b})


if __name__ == '__main__':
    # run_static_rnn()
    data_dynamic = Data.dynamic()
    # run_dynamic_rnn(data_dynamic)

    data_static = Data.static()
    run_static_rnn(data_static, use_sequence_length_info=True)
    run_static_rnn(data_static)
