from datasets import load_chinese_sent
import logging
from classifiers import SequenceClassifier
import numpy as np
import random
import sys


def zh_vec_dense():
    train, val, test = load_chinese_sent()
    pipeline = 'zh_vec_2d'
    network = 'vec_dense'
    save_dir = 'models/zh_vec_dense'
    SequenceClassifier.train(pipeline, network, train, val, save_dir)


def zh_vec_recurrent():
    train, val, test = load_chinese_sent()
    pipeline = 'zh_vec_3d'
    network = 'vec_recurrent'
    save_dir = 'models/zh_vec_recurrent'
    SequenceClassifier.train(pipeline, network, train, val, save_dir)


def zh_int_recurrent():
    train, val, test = load_chinese_sent()
    pipeline = 'zh_int'
    network = 'int_recurrent'
    save_dir = 'models/zh_int_recurrent'
    SequenceClassifier.train(pipeline, network, train, val, save_dir)


def zh_int_recurrent_two():
    train, val, test = load_chinese_sent()
    pipeline = 'zh_int'
    network = 'int_recurrent_two'
    save_dir = 'models/zh_int_recurrent_two'
    SequenceClassifier.train(pipeline, network, train, val, save_dir)


def zh_int_multi():
    train, val, test = load_chinese_sent()
    pipeline = 'zh_int'
    network = 'int_diverse_multi'
    save_dir = 'models/zh_int_diverse_multi'
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
