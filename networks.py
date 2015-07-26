from keras.models import Sequential
from keras.layers.advanced_activations import PReLU
from keras.layers.core import Activation, Dense, Dropout, Flatten, Layer, \
    Merge, TimeDistributedDense
from keras.layers.embeddings import Embedding
from keras.layers.normalization import BatchNormalization
from keras.layers.recurrent import GRU, LSTM


def vec_dense(input_dim, output_dim):
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
    return m, 1


def vec_recurrent(input_dim, output_dim):
    m = Sequential()
    m.add(Dropout(0.8))
    m.add(GRU(input_dim, 128))
    m.add(Dropout(0.7))
    m.add(Dense(128, output_dim))
    m.add(Activation('softmax'))
    return m, 1


def vec_recurrent_two(input_dim, output_dim):
    m = Sequential()
    m.add(Dropout(0.5))

    m.add(GRU(input_dim, 512, return_sequences=True))
    m.add(Dropout(0.5))

    m.add(GRU(512, 256))
    m.add(Dropout(0.5))

    m.add(Dense(256, output_dim))
    m.add(Activation('softmax'))
    return m, 1


def int_recurrent(input_dim, output_dim):
    m = Sequential()
    m.add(Embedding(input_dim, 512))
    m.add(Dropout(0.8))
    m.add(GRU(512, 128))
    m.add(Dropout(0.7))
    m.add(Dense(128, output_dim))
    m.add(Activation('softmax'))
    return m, 1


def int_recurrent_two(input_dim, output_dim):
    m = Sequential()
    m.add(Embedding(input_dim, 256))
    m.add(Dropout(0.8))
    m.add(GRU(256, 256, return_sequences=True))
    m.add(Dropout(0.8))
    m.add(GRU(256, 64))
    m.add(Dropout(0.6))
    m.add(Dense(64, output_dim))
    m.add(Activation('softmax'))
    return m, 1


def int_recurrent_multi(input_dim, output_dim):
    def frontend():
        m = Sequential()
        m.add(Embedding(input_dim, 512))
        m.add(Dropout(0.8))
        m.add(GRU(512, 128))
        m.add(Dropout(0.7))
        m.add(Dense(128, 64))
        return m

    frontends = [frontend() for i in range(4)]

    m = Sequential()
    m.add(Merge(frontends, mode='sum'))
    m.add(Dense(64, output_dim))
    m.add(Activation('softmax'))
    return m, len(frontends)


def int_diverse_multi(input_dim, output_dim):
    def gru_frontend():
        m = Sequential()
        m.add(Embedding(input_dim, 64))
        m.add(Dropout(0.8))
        m.add(GRU(64, 64))
        m.add(Dropout(0.8))
        m.add(Dense(64, 64))
        return m

    def dense_frontend():
        m = Sequential()
        m.add(Embedding(input_dim, 64))
        m.add(Dropout(0.8))
        m.add(TimeDistributedDense(64, 64))
        m.add(Flatten())

        m.add(PReLU((256 * 64,)))
        m.add(BatchNormalization((256 * 64,)))
        m.add(Dropout(0.9))
        m.add(Dense(256 * 64, 512))

        m.add(PReLU((512,)))
        m.add(BatchNormalization((512,)))
        m.add(Dropout(0.8))
        m.add(Dense(512, 64))
        return m

    frontends = \
        [gru_frontend() for i in range(2)] + \
        [dense_frontend() for i in range(2)]

    m = Sequential()
    m.add(Merge(frontends, mode='concat'))
    m.add(PReLU((256,)))
    m.add(BatchNormalization((256,)))
    m.add(Dropout(0.8))
    m.add(Dense(256, output_dim))
    m.add(Activation('softmax'))
    return m, len(frontends)
