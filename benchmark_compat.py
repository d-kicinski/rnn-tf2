import os

os.environ['TF_ENABLE_CONTROL_FLOW_V2'] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import tensorflow.compat.v1 as tfc

from models.rnn import StaticLSTM, DynamicLSTM, LiteDynamicLSTM
from tf_rnn import Data, run_rnn_compat, BenchmarkSequences


def tf_compat(seq):
    tfc.disable_eager_execution()

    data_dynamic = Data.dynamic(seq)

    # This doesn't work under tensorflow-2.0.0-rc0, should be run with tensorflow-1.14.0
    run_rnn_compat(LiteDynamicLSTM, data_dynamic)
    run_rnn_compat(LiteDynamicLSTM, data_dynamic, use_sequence_length_info=True)

    run_rnn_compat(DynamicLSTM, data_dynamic)
    run_rnn_compat(DynamicLSTM, data_dynamic, use_sequence_length_info=True)

    data_static = Data.static(seq)

    run_rnn_compat(StaticLSTM, data_static)
    run_rnn_compat(StaticLSTM, data_static, use_sequence_length_info=True)


if __name__ == '__main__':
    tf_compat(BenchmarkSequences.short)
    tf_compat(BenchmarkSequences.long)
