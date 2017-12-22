__author__ = 'HyNguyen'

import gensim
# setup logging
import logging

import argparse

import numpy as np


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-fi', required=True, type=str)
    parser.add_argument('-fo', required=True, type=str)
    parser.add_argument('-log', type=str, default="logs/w2v.txt")
    parser.add_argument('-worker', type=int, default=4)
    parser.add_argument('-mincount', type=int, default=5)
    parser.add_argument('-size', type=int, default=100)
    parser.add_argument('-epoch', type=int, default=100)

    args = parser.parse_args()
    fi = args.fi
    fo = args.fo
    woker = args.worker
    min_count = args.mincount
    size = args.size
    epoch = args.epoch
    log = args.log

    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO,
                        filename=log, filemode='w')

    model = gensim.models.word2vec.Word2Vec(workers=woker,min_count=min_count,size=size)
    sentences = gensim.models.word2vec.LineSentence(fi)
    model.build_vocab(sentences)
    model.train(sentences, total_examples=model.corpus_count,epochs=epoch)
    model.wv.save_word2vec_format(fo)
