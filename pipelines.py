from camacho.base import Transformer
from camacho.preprocess.sequence.coders import IntCoder
from camacho.preprocess.binarize.embeddings import Word2Vec
from camacho.preprocess.binarize.onehot import AtomBinarizer
from camacho.preprocess.sequence.max_length import MaxLength
from camacho.preprocess.sequence.min_length import MinLength
import numpy as np


def zh_embed_2d():
    data = [
        MaxLength(256),
        MinLength(256),
        IntCoder(min_freq=10),
    ]

    labels = [
        AtomBinarizer(),
    ]

    need_to_embed = True

    return data, labels, need_to_embed


class Int2Str(Transformer):
    """
    Word2Vec expects string inputs.
    """

    def transform(self, nnn):
        return map(lambda nn: map(str, nn), nnn)

    def inverse_transform(self, sss):
        return map(lambda ss: map(int, ss), sss)


def zh_vec_3d():
    data = [
        MaxLength(256),
        MinLength(256),
        IntCoder(min_freq=10),
        Int2Str(),
        Word2Vec(16),
    ]

    labels = [
        AtomBinarizer(),
    ]

    need_to_embed = False

    return data, labels, need_to_embed


class Flatten(Transformer):
    def transform(self, aaaa):
        rrr = []
        for aaa in aaaa:
            rr = []
            for aa in aaa:
                rr.extend(aa)
            rrr.append(rr)
        return np.array(rrr)


def zh_vec_2d():
    data = [
        MaxLength(256),
        MinLength(256),
        IntCoder(min_freq=10),
        Int2Str(),
        Word2Vec(16),
        Flatten(),
    ]

    labels = [
        AtomBinarizer(),
    ]

    need_to_embed = False

    return data, labels, need_to_embed
