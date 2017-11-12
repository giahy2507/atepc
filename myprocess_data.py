import os
from collections import Counter
import xml.etree.ElementTree as ET

import numpy as np
import cPickle
from collections import defaultdict
import sys, re
import pandas as pd
import nltk
import editdistance
import codecs
import unicodedata


def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Every dataset is lower cased except for TREC
    """
    if isinstance(string, unicode):
        string = unicodedata.normalize('NFKD', string).encode('ascii', 'ignore')
    string = re.sub(r"[^A-Za-z0-9(),\!\?\'\`\/\-\.\;]", " ", string)
    string = re.sub(r"\/", " / ", string)
    string = re.sub(r"[\?]{2,8}", " ? ", string)
    string = re.sub(r"[\.]{2,8}", " ... ", string)
    string = re.sub(r"[\-]{2,8}", " - ", string)
    string = re.sub(r"[\!]{2,8}", " ! ", string)
    string = re.sub(r"\- ", " - ", string)
    words = nltk.word_tokenize(string.lower())
    for i in range(len(words)):
        if len(words[i]) > 1:
            if words[i][0] == "-" and words[i][1:].isalnum():
                print "case pref", words[i], words[i][1:]
                words[i] = words[i][1:]
            if words[i][-1] == "-" and words[i][:-1].isalnum():
                print "case suff", words[i], words[i][:-1]
                words[i] = words[i][:-1]
    return " ".join(words)

def normalize_str(string, clean_string = True):
    if clean_string is True:
        processed_string = clean_str(string)
    else:
        processed_string = string.lower()

    # str start with "-\w" pattern
    if len(processed_string) > 2:
        if processed_string[0].isalpha() is False:
            processed_string = processed_string[0] + " " + processed_string[1:]
    return processed_string

def generate_y(words_pre, words_asp, words_suf, polarity, taskname="APC"):

    if polarity == "positive":
        senti_label = 2
    elif polarity == "negative":
        senti_label = 0
    elif polarity == "neutral":
        senti_label = 1
    else:
        senti_label = None

    if taskname == "APC":
        """
            - negative: 0
            - neutral : 1
            - positive: 2
        """
        return senti_label
    elif taskname == "ATE":
        """
            BIO encoding:
            - O: 0
            - B: 1
            - I: 2
        """
        seq_label = [2]*len(words_asp)
        seq_label[0] = 1
        result = [0]*len(words_pre) + seq_label + [0]*len(words_suf)
        return result
    elif taskname == "ATEPC":
        """
            BIO_sent encoding:
            - 0    : 0
            - B_neg: 1
            - B_neu: 2
            - B_pos: 3
            - I    : 4
        """
        seq_label = [4] * len(words_asp)
        B_senti = 0
        if senti_label == 0:
            B_senti = 1
        elif senti_label == 1:
            B_senti = 2
        elif senti_label == 2:
            B_senti = 3
        seq_label[0] = B_senti
        result = [0]*len(words_pre) + seq_label + [0]*len(words_suf)
        return result

def gen_sequence_label(sentence, asp_terms, clean_string = True, taskname="ATEPC"):
    words = []
    y = []
    pre_idx = 0
    sorted_asp_terms = sorted(asp_terms, key=lambda x: x["from_idx"])
    for asp_term in sorted_asp_terms:
        pre_text = sentence[pre_idx:asp_term["from_idx"]]
        asp_text = sentence[asp_term["from_idx"]:asp_term["to_idx"]]
        processed_pretext = normalize_str(pre_text)
        processed_aspterm = normalize_str(asp_text)
        words_pre = processed_pretext.split()
        words_asp = processed_aspterm.split()
        y_sub = generate_y(words_pre, words_asp, [], asp_term["polarity"], taskname=taskname)
        words += words_pre+words_asp
        y     += y_sub
        pre_idx = asp_term["to_idx"]

    # process remain text
    text_remain = sentence[pre_idx:]
    processed_remaintext = normalize_str(text_remain)
    words_remain = processed_remaintext.split()
    words+=words_remain
    y+=[0]*len(words_remain)

    # raise exeption if len of words and y are not equal
    if len(words) != len(y):
        raise
    return words, y

def read_APC(fname, cv=10 , clean_string=True):
    if os.path.isfile(fname) == False:
        raise("[!] Data %s not found" % fname)

    tree = ET.parse(fname)
    root = tree.getroot()
    vocab = defaultdict(float)
    revs = []
    # prepare vocabulary
    source_words, target_words, max_sent_len = [], [], 0
    for sentence in root:
        text = sentence.find('text').text.strip()
        text = text.encode('ascii', 'ignore').decode('ascii')
        for asp_terms_tag in sentence.iter('aspectTerms'):
            for asp_term_tag in asp_terms_tag.findall('aspectTerm'):
                asp_term = asp_term_tag.get('term')
                asp_term = asp_term.encode('ascii', 'ignore').decode('ascii')
                from_idx = int(asp_term_tag.get('from'))
                to_idx = int(asp_term_tag.get('to'))
                polarity = asp_term_tag.get('polarity')

                if polarity == "conflict":
                    continue

                pre_text = text[:from_idx]
                suf_text = text[to_idx:]

                if clean_string:
                    processed_pretext = clean_str(pre_text)
                    processed_aspterm = clean_str(asp_term)
                    processed_suftext = clean_str(suf_text)
                else:
                    processed_pretext = pre_text.lower()
                    processed_aspterm = asp_term.lower()
                    processed_suftext = suf_text.lower()

                words_pre = processed_pretext.split()
                words_asp = processed_aspterm.split()
                words_suf = processed_suftext.split()

                y = generate_y(words_pre, words_asp, words_suf, polarity=polarity, taskname="APC")


                datum = {"words_pre": words_pre, "words_asp": words_asp, "words_suf": words_suf,
                         "y": y,
                         "polarity": polarity,
                         "num_l_words": len(words_pre),
                         "num_a_words": len(words_asp),
                         "num_r_words": len(words_suf),
                         "split": np.random.randint(0, cv)}
                revs.append(datum)

        processed_text = clean_str(text) if clean_string else text.lower()
        words = set(processed_text.split())
        for word in words:
            vocab[word] += 1
    return revs, vocab

def read_ATEPC(fname, cv=10 , clean_string=True, taskname="ATEPC"):
    if os.path.isfile(fname) == False:
        raise("[!] Data %s not found" % fname)

    tree = ET.parse(fname)
    root = tree.getroot()
    vocab = defaultdict(float)
    revs = []
    # prepare vocabulary
    for sentence in root:
        text = sentence.find('text').text
        asp_terms_tag = sentence.findall('aspectTerms')
        contain_conflict = False
        asp_terms = []
        #TODO check unicode, transform to unicode all data
        if len(asp_terms_tag) != 0:
            for asp_term_tag in asp_terms_tag[0].findall('aspectTerm'):
                asp_term = asp_term_tag.get('term').encode("utf-8")
                from_idx = int(asp_term_tag.get('from'))
                to_idx = int(asp_term_tag.get('to'))
                polarity = asp_term_tag.get('polarity').encode("utf-8")
                if polarity == "conflict":
                    contain_conflict = True
                else:
                    asp_terms.append({"asp_term": asp_term,
                                      "from_idx": from_idx,
                                      "to_idx":to_idx,
                                      "polarity": polarity})
        if contain_conflict is True:
            continue
        else:
            words, y = gen_sequence_label(text, asp_terms, clean_string, taskname)

        if words[-1].strip() == ".":
            words = words[:-1]
            y = y[:-1]

        if len(words) != len(y):
            raise

        revs.append({
                "words":words,
                "y":y,
                "no_aspterms":len(asp_terms),
                "no_words": len(words),
                "split": np.random.randint(0, cv)})
        uni_words = set(words)
        for word in uni_words:
            if isinstance(word, unicode) is True:
                print "sai sai dm"
            vocab[word] += 1
    return revs, vocab

def load_bin_vec(fname, vocab, mapp_vocab):
    """
    Loads 300x1 word vecs from Google (Mikolov) word2vec
    """
    mapp_target_vocab = defaultdict(float)
    for key, value in mapp_vocab.items():
        tokens = value.split("+")[1:]
        for token in tokens:
            mapp_target_vocab[token] += 1

    word_vecs = {}
    with open(fname, "r") as f:
        header = f.readline()
        vocab_size, layer1_size = map(int, header.split())
        binary_len = np.dtype('float32').itemsize * layer1_size
        for line in xrange(vocab_size):
            word = []
            while True:
                ch = f.read(1)
                if ch == ' ':
                    word = ''.join(word)
                    break
                if ch != '\n':
                    word.append(ch)
            if isinstance(word, unicode):
                word = unicodedata.normalize('NFKD', word).encode('ascii', 'ignore')
            if word in vocab or word in mapp_target_vocab:
                word_vecs[word] = np.fromstring(f.read(binary_len), dtype='float32')
            else:
                f.read(binary_len)
    return word_vecs

def load_vocab_w2v(fname):
    tokens = re.split(r"[\/\\]", fname)
    vocab_path = "model/vocab." + tokens[-1]+".cpickle"
    if os.path.isfile(vocab_path) is False:
        vocab = defaultdict(float)
        with codecs.open(fname, "r") as f:
            for line in f:
                word = line.split()[0]
                if isinstance(word, unicode):
                    word = unicodedata.normalize('NFKD', word).encode('ascii', 'ignore')
                vocab[word] += 1
        with codecs.open(vocab_path, mode="wb") as f:
            cPickle.dump(vocab, f)
        return vocab
    else:
        with codecs.open(vocab_path, mode="rb") as f:
            vocab = cPickle.load(f)
        return vocab



def load_data(name="laptop"):
    if name == "laptops":
        print("--Laptops--")
        revs_train, vocab_train = read_ATEPC("data/Laptops_Train_v2.xml", taskname="ATEPC")
        revs_test, vocab_test = read_ATEPC("data/Laptops_Test_Gold.xml", taskname="ATEPC")

    else:
        print("--Restaurant--")
        revs_train, vocab_train = read_ATEPC("data/Restaurants_Train_v2.xml", taskname="ATEPC")
        revs_test, vocab_test = read_ATEPC("data/Restaurants_Test_Gold.xml", taskname="ATEPC")

    print("Train, Test size: ", len(revs_train), len(revs_test))
    vocab = defaultdict(float)
    for key, value in vocab_train.items() + vocab_test.items():
        vocab[key] += value
    print("Vocabulary size: ", len(vocab))
    return revs_train, revs_test, vocab

    # """
    # for rev in revs_train + revs_test:
    #     y = rev["y"]
    #     for i in range(len(y)-1):
    #         if y[i] == 4 and y[i+1] in [1,2,3]:
    #             print "hynguyen sai roi"
    #             print y
    #             print " ".join(rev["words"])
    # """


def mapping_vocab(w2v_vocab, vocab, use_editdistance = False, no_editdistance_words = 1, name="laptops"):
    if os.path.isfile("model/mappvocab.{0}.pickle".format(name)) is False:
        mapp_vocab = defaultdict(str)
        for key in vocab.keys():
            if key not in w2v_vocab:
                tokens = key.split("-")
                if len(tokens) > 1:
                    for token in tokens:
                        if token in w2v_vocab:
                            mapp_vocab[key]+="+{0}".format(token)
                elif use_editdistance is True and len(tokens) == 1:
                    xxx = []
                    for keyw2v in w2v_vocab.keys():
                        if int(editdistance.eval(key, keyw2v)) < 6:
                            xxx.append([key, keyw2v, int(editdistance.eval(key, keyw2v))])
                    sorted_xxx = sorted(xxx, key=lambda item: item[2])
                    for i in range(no_editdistance_words):
                        mapp_vocab[key] += "+{0}".format(sorted_xxx[i][1])
                    print key
                    print " ".join(["{0}_{1}".format(x[1], x[2]) for x in sorted_xxx[:5]])
                    print
        with codecs.open("model/mappvocab.{0}.pickle".format(name), mode="wb") as f:
            cPickle.dump(mapp_vocab, f)
            return mapp_vocab
    else:
        with codecs.open("model/mappvocab.{0}.pickle".format(name), mode="rb") as f:
            return cPickle.load(f)

if __name__ == "__main__":
    w2v_vocab = load_vocab_w2v("C:\\hynguyen\\Data\\vector\\glove.42B.300d\\glove.42B.300d.txt")
    print (len(w2v_vocab))

    revs_train, revs_test, vocab = load_data("restaurants")
    print(len(vocab))
    # print len(w2v_vocab), len(vocab)
    #
    mapp_vocab = mapping_vocab(w2v_vocab, vocab, use_editdistance=True, name = "restaurants")
    print "ahihi"
    #
    # with open("model/mapp_vocab_restaurants.pickle", mode="wb") as f:
    #     cPickle.dump(mapp_vocab, f)

    # for key, value in mapp_vocab.items():
    #     print "{0}\t\t{1}".format(key, value)
    #

