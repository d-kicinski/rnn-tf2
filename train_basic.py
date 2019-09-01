import numpy as np
import tensorflow as tf
from dataclasses import dataclass
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

from models.basic import FullConnected

UNITS_LATENT = 200
UNITS_OUTPUT = 3
EPOCH_COUNT = 5

DEBUG = False


# DEBUG = True


@dataclass
class Set:
    features: np.array
    labels: np.array


class IrisDataset:
    def __init__(self):
        iris_data = load_iris()
        x = iris_data.data
        y = iris_data.target.reshape(-1, 1)
        encoder = OneHotEncoder(sparse=False)
        y = encoder.fit_transform(y)
        train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.2)

        self.train = Set(features=train_x, labels=train_y)
        self.test = Set(features=test_x, labels=test_y)


class TrainFunc:
    def __init__(self, model, optimizer, loss, debug=True):
        self.optimizer = optimizer
        self.model = model
        self.loss_fn = loss

        if debug:
            self.call_fn = self.train_step
        else:
            features_spec = tf.TensorSpec(shape=[None, 4], dtype=tf.float32)
            labels_spec = tf.TensorSpec(shape=[None, 3], dtype=tf.int32)
            self.call_fn = tf.function(func=self.train_step, input_signature=[features_spec, labels_spec])

    @tf.function(input_signature=[tf.TensorSpec(shape=[None, 4], dtype=tf.float32)])
    def predict(self, features):
        return self.model(features)

    @tf.function(input_signature=[tf.TensorSpec(shape=[None, 4], dtype=tf.float32),
                                  tf.TensorSpec(shape=[None, 3], dtype=tf.int32)])
    def train_step(self, features, labels, *args, **kwargs):
        with tf.GradientTape() as tape:
            prediction = self.model(features)
            loss = self.loss_fn(labels, prediction)

        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        return loss

    def __call__(self, *args, **kwargs):
        return self.call_fn(*args, **kwargs)


def batch(features, labels, batch_size=10):
    assert len(features) == len(labels)
    for i in range(0, len(features), batch_size):
        try:
            yield features[i: i + batch_size], labels[i: i + batch_size]
        except IndexError:
            yield features[i:], labels[i:]


def train_basic(debug=True):
    data = IrisDataset()
    model = FullConnected(units_latent=UNITS_LATENT, units_output=UNITS_OUTPUT)

    loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
    optimizer = tf.keras.optimizers.Adam()
    train_fn = TrainFunc(model, optimizer, loss_fn, debug=debug)

    concrete_func = train_fn.predict.get_concrete_function()
    converter = tf.lite.TFLiteConverter([concrete_func])
    model_lite = converter.convert()

    concrete_func = train_fn.train_step.get_concrete_function()
    converter = tf.lite.TFLiteConverter([concrete_func])
    model_lite = converter.convert()

    # for i_epoch in range(EPOCH_COUNT):
    #     loss = None
    #     for features, labels in batch(data.train.features, data.train.labels):
    #         loss = train_fn(features=features, labels=labels)
    #     print(f"Epoch: {i_epoch}, Loss: {loss.numpy()}")
    #
    # return train_fn


if __name__ == '__main__':
    train_basic(debug=DEBUG)

    # begin_debug = time()
    # train_basic(debug=True)
    # end_debug = time()
    # d_debug = end_debug - begin_debug

    # begin = time()
    # train_fn = train_basic(debug=DEBUG)
    # end = time()
    # d = end - begin
    #
    # fn = train_fn.call_fn.get_concrete_function()
    # converter = tf.lite.TFLiteConverter([fn])
    # model_lite = converter.convert()
    #
    # print(f"Training times [s]:")
    # print(f"\tDebug:\t {d_debug:.2f}")
    # print(f"\tProduction:\t {d:.2f}")

    # class BasicModel(tf.Module):
    #     def __init__(self):
    #         self.const = None
    #
    #     @tf.function(input_signature=[tf.TensorSpec(shape=[1], dtype=tf.float32)])
    #     def pow(self, x):
    #         if self.const is None:
    #             self.const = tf.Variable(2.)
    #         return x ** self.const

    # Create the tf.Module object.
    # root = TrainFunc()

    # Get the concrete function.
    # concrete_func = root.pow.get_concrete_function()

    # converter = tf.lite.TFLiteConverter([concrete_func])
    # model_lite = converter.convert()
