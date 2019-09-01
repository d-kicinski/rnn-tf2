from itertools import chain
from time import time

import os
from typing import *

os.environ['TF_ENABLE_CONTROL_FLOW_V2'] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import numpy as np
import tensorflow as tf
import tensorflow.compat.v1 as tfc

from dataclasses import dataclass

from models.rnn import RNN, StaticLSTM, DynamicLSTM, LiteDynamicLSTM


@dataclass
class Shapes:
    batch: int = 10
    sequence: int = 30
    features: int = 50
    latent: int = 100


@dataclass
class Constants:
    epochs: int = 100
    max_sequence_length: int = 1000


class TrackTime:
    def __init__(self, name: str):
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
    sequence_lengths: List[np.array]
    features: List[np.array]

    @staticmethod
    def _benchmark_range():
        seq_range = [10, 50, 200, 500, 1000]
        return chain(seq_range, reversed(seq_range))

    @staticmethod
    def dynamic():
        """Not padded data with different sequence lengths"""
        sequence_lengths = []
        features = []
        for seq_len in Data._benchmark_range():
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

        for seq_len in Data._benchmark_range():
            sequence_lengths.append([np.array([seq_len for _ in range(Shapes.batch)])
                                     for _ in range(Constants.epochs)])

            features.append([np.random.rand(Shapes.batch,
                                            Constants.max_sequence_length,
                                            Shapes.features) for _ in range(Constants.epochs)])
        return Data(sequence_lengths, features)


def run_rnn(model_cls: Type[RNN.Class], data: Data, use_sequence_length_info: bool = False):
    seq_len_dim = Constants.max_sequence_length if model_cls is StaticLSTM else None
    with tfc.variable_scope("input"):
        features = tfc.placeholder(shape=(Shapes.batch, seq_len_dim, Shapes.features),
                                   dtype=tf.float32, name="features")

        sequence_length = tfc.placeholder(shape=(Shapes.batch,), dtype=tf.float32, name="sequence_length")

    with tfc.variable_scope(model_cls.__name__ + str(use_sequence_length_info), reuse=tfc.AUTO_REUSE):
        lstm = model_cls(latent_units=Shapes.latent) \
            .output(features, sequence_length=sequence_length if use_sequence_length_info else None)

    with tfc.Session() as sess:
        sess.run(tfc.global_variables_initializer())
        print(f"Running {model_cls.__name__}")
        with TrackTime("TOTAL"):
            for seq_len, data in zip(data.sequence_lengths, data.features):
                with TrackTime(f"batch_size: {Shapes.batch}\t sequence_length: {seq_len[0][0]}\t "
                               f"use_sequence_length_info={use_sequence_length_info}\t "):
                    for seq_len_b, data_b in zip(seq_len, data):
                        sess.run(lstm, feed_dict={features: data_b,
                                                  sequence_length: seq_len_b})


if __name__ == '__main__':
    data_dynamic = Data.dynamic()
    run_rnn(DynamicLSTM, data_dynamic)
    run_rnn(DynamicLSTM, data_dynamic, use_sequence_length_info=True)

    run_rnn(LiteDynamicLSTM, data_dynamic)
    run_rnn(LiteDynamicLSTM, data_dynamic, use_sequence_length_info=True)

    data_static = Data.static()
    run_rnn(StaticLSTM, data_static)
    run_rnn(StaticLSTM, data_static, use_sequence_length_info=True)
