from keras.models import Sequential
from keras.layers.advanced_activations import PReLU
from keras.layers.core import Activation, Dense, Dropout, Layer, \
    TimeDistributedDense
from keras.layers.embeddings import Embedding
from keras.layers.normalization import BatchNormalization
from keras.layers.recurrent import GRU, LSTM


def vec_ff_baseline(input_dim, output_dim):
    m = Sequential()
    m.add(Dropout(0.5))

    m.add(Dense(input_dim, 512))
    m.add(PReLU((512,)))
    m.add(BatchNormalization((512,)))
    m.add(Dropout(0.5))

    m.add(Dense(512, 512))
    m.add(PReLU((512,)))
    m.add(BatchNormalization((512,)))
    m.add(Dropout(0.5))

    m.add(Dense(512, 256))
    m.add(PReLU((256,)))
    m.add(BatchNormalization((256,)))
    m.add(Dropout(0.2))

    m.add(Dense(256, output_dim))
    m.add(Activation('softmax'))
    return m


def vec_rnn_baseline(input_dim, output_dim):
    m = Sequential()
    m.add(Dropout(0.5))
    m.add(GRU(input_dim, 16))
    m.add(Dropout(0.5))
    m.add(Dense(16, output_dim))
    m.add(Activation('softmax'))
    return m


def vec_rnn_two(input_dim, output_dim):
    m = Sequential()
    m.add(Dropout(0.5))

    m.add(GRU(input_dim, 512, return_sequences=True))
    m.add(Dropout(0.5))

    m.add(GRU(512, 256))
    m.add(Dropout(0.5))

    m.add(Dense(256, output_dim))
    m.add(Activation('softmax'))
    return m


def int_rnn_baseline(input_dim, output_dim):
    m = Sequential()
    m.add(Embedding(input_dim, 512))
    m.add(Dropout(0.8))
    m.add(GRU(512, 128))
    m.add(Dropout(0.7))
    m.add(Dense(128, output_dim))
    m.add(Activation('softmax'))
    return m


def int_rnn_two(input_dim, output_dim):
    m = Sequential()
    m.add(Embedding(input_dim, 256))
    m.add(Dropout(0.8))
    m.add(GRU(256, 256, return_sequences=True))
    m.add(Dropout(0.8))
    m.add(GRU(256, 64))
    m.add(Dropout(0.6))
    m.add(Dense(64, output_dim))
    m.add(Activation('softmax'))
    return m
