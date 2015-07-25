import os
import random


def load_chinese_sent():
    def each_text(d):
        for fn in os.listdir(d):
            fn = os.path.join(d, fn)
            with open(fn) as f:
                yield f.read().decode('utf-8')

    base = 'data/jingdong/'
    poss = list(each_text(os.path.join(base, 'pos')))
    negs = list(each_text(os.path.join(base, 'neg')))
    pairs = map(lambda s: (s, 'pos'), poss) + map(lambda s: (s, 'neg'), negs)
    random.shuffle(pairs)
    n = len(pairs)
    train = zip(*pairs[:int(n * 0.9)])
    val = zip(*pairs[int(n * 0.8):int(n * 0.9)])
    test = zip(*pairs[int(n * 0.9):])
    return train, val, test
