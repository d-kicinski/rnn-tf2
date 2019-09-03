from itertools import chain
from time import time
from typing import *

import numpy as np
import tensorflow as tf
import tensorflow.compat.v1 as tfc
from dataclasses import dataclass

from models.rnn import RNN, StaticLSTM


@dataclass
class BenchmarkSequences:
    long: Tuple[int] = (10, 50, 200, 500, 1000)
    short: Tuple[int] = (10, 50, 200)

@dataclass
class Shapes:
    batch: int = 32
    features: int = 200
    latent: int = 200


@dataclass
class Constants:
    epoch_count: int = 10
    batch_count: int = 50


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
    max_sequence_length: int

    @staticmethod
    def _benchmark_range(seq_range):
        return chain(seq_range, reversed(seq_range))

    @staticmethod
    def _create_dataset(seq, pad: bool = False):
        sequence_lengths = []
        features = []
        max_sequence_length = max(seq)
        for seq_len in Data._benchmark_range(seq):
            sequence_lengths.append([np.array([seq_len for _ in range(Shapes.batch)]).astype(np.float32)
                                     for _ in range(Constants.batch_count)])

            features.append([np.random.rand(Shapes.batch,
                                            max_sequence_length if pad else seq_len,
                                            Shapes.features).astype(np.float32) for _ in range(Constants.batch_count)])
        return Data(sequence_lengths, features, max_sequence_length)

    @staticmethod
    def dynamic(seq: List[int]):
        """Not padded data with different sequence lengths"""
        return Data._create_dataset(seq, pad=False)

    @staticmethod
    def static(seq: List[int]):
        """Padded data with different sequence lengths"""
        return Data._create_dataset(seq, pad=True)


def run_rnn_compat(model_cls: Type[RNN.Class], data: Data, use_sequence_length_info: bool = False):
    seq_len_dim = data.max_sequence_length if model_cls is StaticLSTM else None
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
                with TrackTime(f"epoch_count: {Constants.epoch_count}\t"
                               f" batch_count: " f"{Constants.batch_count}\t"
                               f" batch_size:" f" {Shapes.batch}\t "
                               f"sequence_length:" f" {seq_len[0][0]}\t "
                               f"use_sequence_length_info={use_sequence_length_info}\t "):
                    for _ in range(Constants.epoch_count):
                        for seq_len_b, data_b in zip(seq_len, data):
                            sess.run(lstm, feed_dict={features: data_b,
                                                      sequence_length: seq_len_b})


def run_rnn_keras(model_cls: Type[RNN.Class], data: Data):
    lstm = tf.function(model_cls(latent_units=Shapes.latent).output)
    print(f"Running {model_cls.__name__}")
    with TrackTime("TOTAL"):
        for seq_len, data in zip(data.sequence_lengths, data.features):
            with TrackTime(f"epoch_count: {Constants.epoch_count}\t"
                           f" batch_count: " f"{Constants.batch_count}\t"
                           f" batch_size:" f" {Shapes.batch}\t "
                           f" sequence_length: {seq_len[0][0]}\t "):
                for _ in range(Constants.epoch_count):
                    for seq_len_b, data_b in zip(seq_len, data):
                        lstm(data_b)
