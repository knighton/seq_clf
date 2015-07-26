from camacho.pipelines import TransformerPipeline
import cPickle
import datasets
from keras.utils import np_utils
from keras_util import SaveModelsAndTerminateEarly, EarlyTermination
import logging
import networks
import numpy as np
import os
import pipelines
import shutil
from time import time


def fit_transform(train_set, val_set, data_pipe, class_pipe):
    t0 = time()
    print 'Transforming...'

    X_train, y_train = train_set
    X_train = data_pipe.fit_transform(X_train)
    y_train = class_pipe.fit_transform(y_train)

    X_val, y_val = val_set
    X_val = data_pipe.transform(X_val)
    y_val = class_pipe.transform(y_val)

    t1 = time()
    print t1 - t0, 'Done transforming.'

    return (X_train, y_train), (X_val, y_val)


def construct(network_func_name, embed_len, output_dim):
    f = getattr(networks, network_func_name)
    kmodel, nb_frontends = f(embed_len, output_dim)

    print 'Compiling...'
    t0 = time()
    kmodel.compile(loss='categorical_crossentropy', optimizer='adam')
    t1 = time()
    print 'Done compiling (%.3f sec).' % (t1 - t0)

    return kmodel, nb_frontends


class SequenceClassifier(object):
    @staticmethod
    def train(pipeline_func_name, network_func_name, train_set, val_set,
              save_dir):
        # Init the data transformers.
        f = getattr(pipelines, pipeline_func_name)
        data_pipe, class_pipe, need_to_embed = f()
        data_pipe = TransformerPipeline(data_pipe)
        class_pipe = TransformerPipeline(class_pipe)

        # Transform the data.
        (X_train, y_train), (X_val, y_val) = \
            fit_transform(train_set, val_set, data_pipe, class_pipe)

        # Build the model, now that we know its input/output dimensions.
        print 'X_train.shape:', X_train.shape
        print 'y_train.shape:', y_train.shape

        if need_to_embed:
            input_dim = max(map(max, X_train)) + 1
        else:
            input_dim = X_train.shape[-1]
        output_dim = y_train.shape[-1]

        print 'Input dim:', input_dim
        print 'Output dim:', output_dim

        kmodel, nb_frontends = \
            construct(network_func_name, input_dim, output_dim)

        # Save the model configuration.
        if os.path.exists(save_dir):
            shutil.rmtree(save_dir)
        os.makedirs(save_dir)
        d = {
            'pipeline_func_name': pipeline_func_name,
            'data_pipe': data_pipe,
            'class_pipe': class_pipe,
            'network_func_name': network_func_name,
            'input_dim': input_dim,
            'output_dim': output_dim,
        }
        fn = os.path.join(save_dir, 'model.pkl')
        cPickle.dump(d, open(fn, 'wb'))

        # Train the model, saving checkpoints.
        print 'Training...'
        if nb_frontends != 1:
            X_train = [X_train] * nb_frontends
            X_val = [X_val] * nb_frontends
        cb = SaveModelsAndTerminateEarly()
        cb.set_params(save_dir)
        try:
            kmodel.fit(
                X_train, y_train, validation_data=(X_val, y_val),
                nb_epoch=10000, batch_size=128, callbacks=[cb],
                show_accuracy=True, verbose=True)
        except EarlyTermination:
            logging.info('Terminated training early.')
            pass
        print 'Done training.'

        return Model(data_pipe, kmodel, class_pipe, nb_frontends)

    @staticmethod
    def load(model_f, weights_f):
        # Load the model config.
        print 'Loading model...'
        t0 = time()
        d = cPickle.load(open(model_f))
        data_pipe = d['data_pipe']
        class_pipe = d['class_pipe']
        network_func_name = d['network_func_name']
        input_dim = d['input_dim']
        output_dim = d['output_dim']
        t1 = time()
        print 'Done loading model (%.3f sec).' % (t1 - t0)

        # Compile the model.
        kmodel, nb_frontends = \
            construct(network_func_name, input_dim, output_dim)

        # Load the weights.
        print 'Loading weights...'
        t0 = time()
        kmodel.load_weights(weights_f)
        t1 = time()
        print 'Done loading weights (%.3f sec).' % (t1 - t0)

        return Model(data_pipe, class_pipe, kmodel, nb_frontends)

    def __init__(self, data_pipe, kmodel, class_pipe, nb_frontends):
        self.data_pipe = data_pipe
        self.class_pipe = class_pipe
        self.kmodel = kmodel
        self.nb_frontends = nb_frontends

    def predict(self, X):
        X = self.data_pipe.transform(X)

        if self.nb_frontends != 1:
            X = [X] * self.nb_frontends

        y = self.kmodel.predict_proba(X)

        y = map(np.argmax, y)
        y = self.class_pipe.inverse_transform(y)
        return y
