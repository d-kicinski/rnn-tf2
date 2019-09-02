import os

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

from models.rnn import KerasStaticLSTM, KerasDynamicLSTM
from tf_rnn import Data, run_rnn_keras, BenchmarkSequences


def tf_keras(seq):
    data_dynamic = Data.dynamic(seq)

    run_rnn_keras(KerasDynamicLSTM, data_dynamic)
    run_rnn_keras(KerasStaticLSTM, data_dynamic)


if __name__ == '__main__':
    tf_keras(BenchmarkSequences.short)
    tf_keras(BenchmarkSequences.long)
