from datasets import load_chinese_sent
import logging
from classifiers import SequenceClassifier
import numpy as np
import random
import sys


def zh_w2v_ff_baseline():
    train, val, test = load_chinese_sent()
    pipeline = 'zh_vec_2d'
    network = 'vec_ff_baseline'
    save_dir = 'models/zh_w2v_ff_baseline'
    SequenceClassifier.train(pipeline, network, train, val, save_dir)


def zh_w2v_rnn_baseline():
    train, val, test = load_chinese_sent()
    pipeline = 'zh_vec_3d'
    network = 'vec_rnn_baseline'
    save_dir = 'models/zh_w2v_rnn_baseline'
    SequenceClassifier.train(pipeline, network, train, val, save_dir)


def zh_int_rnn_baseline():
    train, val, test = load_chinese_sent()
    pipeline = 'zh_int'
    network = 'int_rnn_baseline'
    save_dir = 'models/zh_embed_rnn_baseline'
    SequenceClassifier.train(pipeline, network, train, val, save_dir)


def zh_int_rnn_two():
    train, val, test = load_chinese_sent()
    pipeline = 'zh_int'
    network = 'int_rnn_two'
    save_dir = 'models/zh_embed_rnn_two'
    SequenceClassifier.train(pipeline, network, train, val, save_dir)


def main():
    #random.seed(1337)
    #np.random.seed(1337)
    logging.basicConfig(
        format='%(asctime)s %(message)s', datefmt='%4Y-%2m-%2d %2I:%2M:%2S',
        level=logging.INFO)
    logging.info('Begin.')

    func = sys.argv[1]
    globals()[func]()


if __name__ == '__main__':
    main()
