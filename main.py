from pathlib import Path
from dataclasses import dataclass
from typing import List, Iterable
from collections import defaultdict
from functools import reduce
from copy import deepcopy

import numpy as np
import os
os.environ['TF_ENABLE_CONTROL_FLOW_V2'] = '1'
import tensorflow as tf

from spacy.lang.en import English, Language

from models.rnn import LiteLSTM

PATH_NER_TRAIN = Path("data/ner/train.corpus")


class ShakespeareDataset:
    @staticmethod
    def _group(sequence: Iterable, separator: str):
        elements = []
        for el in sequence:
            if el == separator:
                if len(elements) > 1:
                    yield elements
                elements = []
            else:
                elements.append(el)
        if len(elements) > 1:
            yield elements
        else:
            yield None

    @staticmethod
    def prepare_corpora(corpora_path: Path, lang: Language = English):
        tokenizer = lang.Defaults.create_tokenizer()
        text = corpora_path.open('rb').read().decode(encoding='utf-8')
        characters = defaultdict(list)
        gen_utterance = ShakespeareDataset._group(text.split(sep='\n'), separator='')
        for utterance in gen_utterance:
            if utterance is not None:
                name = utterance[0].strip(':').lower()
                utterances = " ".join(utterance[1:])
                characters[name].append(utterances)


@dataclass
class RawExample:
    text: List[str]
    labels: List[str]


@dataclass
class Example:
    text: np.array
    labels: np.array

    @staticmethod
    def from_raw_example(raw_example, label2idx, token2idx):
        return Example(text=np.array([token2idx[t] for t in raw_example.text]),
                       labels=np.array([label2idx[l] for l in raw_example.labels]))


class ConllDataset:
    def __init__(self, corpora_path: Path):
        self.path = corpora_path

        text = self.path.open('r').readlines()

        raw_examples: List[RawExample] = []
        tokens: List[str] = []
        labels: List[str] = []

        skip = False
        for line in text:
            line = line.rstrip('\n\r')

            if "DOCSTART" in line:
                skip = True
                continue

            if skip:
                skip = False
                continue

            if line == "":
                assert len(tokens) == len(labels)
                raw_examples.append(RawExample(text=tokens, labels=labels))
                tokens = []
                labels = []
            else:
                line = line.split(" ")
                tokens.append(line[0])
                labels.append(line[3])

        def flatten(sequence: Iterable) -> Iterable:
            def extend(l1, l2):
                l1.extend(l2)
                return l1

            return reduce(lambda l, r: extend(l, r), sequence)

        self.vocab = sorted(set(flatten((e.text for e in deepcopy(raw_examples)))))
        self.token2idx = {u: i for i, u in enumerate(self.vocab)}
        self.idx2token = np.array(self.vocab)

        self.labels = sorted(set(flatten((e.labels for e in raw_examples))))
        self.label2idx = {u: i for i, u in enumerate(self.labels)}
        self.idx2label = np.array(self.labels)

        self.examples = [Example.from_raw_example(raw_example=e,
                                                  label2idx=self.label2idx,
                                                  token2idx=self.token2idx) for e in raw_examples]

    def __str__(self):
        return "Conll dataset: \n" \
               f"\t{len(self.vocab)} unique tokens\n" \
               f"\t{len(self.labels)} unique labels: {self.labels}\n"


if __name__ == '__main__':
    dataset = ConllDataset(PATH_NER_TRAIN)

    HIDDEN_UNITS = 200
    EMBEDDINGS_DIM = 200

    tensor_input = np.random.rand(1, 50, 200)
    print(f"input tensor shape: {tensor_input.shape}")

    # model_keras = KerasLSTM(latent_units=HIDDEN_UNITS,
    #                         embeddings_dim=EMBEDDINGS_DIM,
    #                         vocab_size=len(dataset.vocab),
    #                         class_size=len(dataset.labels))
    # ret = model_keras.predict(tensor_input)
    # converter = tf.lite.TFLiteConverter.from_keras_model(model_keras)
    # model_keras_lite = converter.convert()
    # print(model_keras.summary())

    # model_tf = LiteKerasLSTM(latent_units=HIDDEN_UNITS,
    #                          embeddings_dim=EMBEDDINGS_DIM,
    #                          vocab_size=len(dataset.vocab),
    #                          class_size=len(dataset.labels))
    # ret = model_tf.predict(tensor_input)
    # converter = tf.lite.TFLiteConverter.from_keras_model(model_tf)
    # model_keras_lite = converter.convert()
    # print(model_keras.summary())

    model_lite = LiteLSTM(latent_units=HIDDEN_UNITS,
                          embeddings_dim=EMBEDDINGS_DIM,
                          vocab_size=len(dataset.vocab),
                          class_size=len(dataset.labels))

    model_lite(tensor_input)
