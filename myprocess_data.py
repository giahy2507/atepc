import os
from collections import Counter
import xml.etree.ElementTree as ET

import numpy as np
import pickle
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
                print ("case pref", words[i], words[i][1:])
                words[i] = words[i][1:]
            if words[i][-1] == "-" and words[i][:-1].isalnum():
                print ("case suff", words[i], words[i][:-1])
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

def generate_y(words_pre, words_asp, words_suf, polarity, taskname="APCTE"):

    if polarity == "positive":
        senti_label = 2
    elif polarity == "negative":
        senti_label = 0
    elif polarity == "neutral":
        senti_label = 1
    else:
        senti_label = None

    if taskname == "ATEPC":
        """
            BIO_sent encoding:
            - O    : 0
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
    elif taskname == "ATEPC2":
        """
        Type 2 of BIO_sent encoding:
        - O    : 0
        - B_neg: 1
        - I_neg: 2
        - B_neu: 3
        - I_neu: 4
        - B_pos: 5
        - I_pos: 6
        """
        seq_label = []
        if senti_label == 0:
            seq_label = [1] + [2] * (len(words_asp) - 1)
        elif senti_label == 1:
            seq_label = [3] + [4] * (len(words_asp) - 1)
        elif senti_label == 2:
            seq_label = [5] + [6] * (len(words_asp) - 1)
        result = [0] * len(words_pre) + seq_label + [0] * len(words_suf)
        return result

def reverse_y(ys, taskname = "ATEPC2"):
    result = []
    for y in ys:
        if taskname == "ATE":
            if y == 0: result.append("O")
            elif y == 1: result.append("B")
            elif y == 2: result.append("I")
            elif y == 3: result.append("B")
            elif y == 4: result.append("I")
            elif y == 5: result.append("B")
            elif y == 6: result.append("I")
            else: result.append("<UNK_TAG>")
        elif taskname == "ATEPC":
            if y == 0: result.append("O")
            elif y == 1: result.append("B-NEG")
            elif y == 2: result.append("I")
            elif y == 3: result.append("B-NEU")
            elif y == 4: result.append("I")
            elif y == 5: result.append("B-POS")
            elif y == 6: result.append("I")
            else: result.append("<UNK_TAG>")
        elif taskname == "ATEPC2":
            if y == 0: result.append("O")
            elif y == 1: result.append("B-NEG")
            elif y == 2: result.append("I-NEG")
            elif y == 3: result.append("B-NEU")
            elif y == 4: result.append("I-NEU")
            elif y == 5: result.append("B-POS")
            elif y == 6: result.append("I-POS")
            else: result.append("<UNK_TAG>")
    return result

def gen_sequence_label(sentence, asp_terms, clean_string = True, taskname="ATEPC"):
    words = []
    y = []
    pre_idx = 0
    sorted_asp_terms = sorted(asp_terms, key=lambda x: x["from_idx"])
    for asp_term in sorted_asp_terms:
        pre_text = sentence[pre_idx:asp_term["from_idx"]]
        asp_text = sentence[asp_term["from_idx"]:asp_term["to_idx"]]
        processed_pretext = normalize_str(pre_text,  clean_string)
        processed_aspterm = normalize_str(asp_text, clean_string)
        words_pre = processed_pretext.split()
        words_asp = processed_aspterm.split()
        y_sub = generate_y(words_pre, words_asp, [], asp_term["polarity"], taskname=taskname)
        words += words_pre+words_asp
        y     += y_sub
        pre_idx = asp_term["to_idx"]

    # process remain text
    text_remain = sentence[pre_idx:]
    processed_remaintext = normalize_str(text_remain, clean_string)
    words_remain = processed_remaintext.split()
    words+=words_remain
    y+=[0]*len(words_remain)

    # raise exeption if len of words and y are not equal
    if len(words) != len(y):
        raise
    return words, y

def filter_conflict_sentence(fname):
    if os.path.isfile(fname) == False:
        raise("[!] Data %s not found" % fname)

    tree = ET.parse(fname)
    root = tree.getroot()
    for sentence in root.findall('sentence'):
        asp_terms_tag = sentence.findall('aspectTerms')
        contain_conflict = False
        if len(asp_terms_tag) != 0:
            for asp_term_tag in asp_terms_tag[0].findall('aspectTerm'):
                polarity = asp_term_tag.get('polarity').encode("utf-8")
                if polarity == "conflict":
                    contain_conflict = True
        if contain_conflict is True:
            root.remove(sentence)

    filename = re.split(r"/", fname)[-1]
    new_fname = ".".join(filename.split(".")[:-1]) + ".FilterOutConflict.xml"
    newfile_path = os.path.dirname(fname)+"/" + new_fname
    tree.write(newfile_path)

def read_ATEPC(fname, cv=10 , clean_string=True, taskname="ATEPC"):
    if os.path.isfile(fname) == False:
        raise("[!] Data %s not found" % fname)

    tree = ET.parse(fname)
    root = tree.getroot()
    vocab = defaultdict(float)
    revs = []
    conflict_sents=[]
    # prepare vocabulary
    for sentence in root:
        text = sentence.find('text').text
        asp_terms_tag = sentence.findall('aspectTerms')
        contain_conflict = False
        asp_terms = []
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
            sentid = sentence.get("id")
            conflict_sents.append(sentid)
            continue
        else:
            words, y = gen_sequence_label(text, asp_terms, clean_string, taskname)

        if len(words) != len(y):
            raise()

        revs.append({
                "words":words,
                "y":y,
                "no_aspterms":len(asp_terms),
                "no_words": len(words),
                "split": np.random.randint(0, cv)})
        uni_words = set(words)
        for word in uni_words:
            if isinstance(word, unicode) is True:
                print ("sai sai dm")
            vocab[word] += 1
    print (conflict_sents)
    return revs, vocab

def load_bin_vec(fname, vocab, mapp_vocab, name="laptops"):
    """
    Loads 300x1 word vecs from Google (Mikolov) word2vec
    """
    if os.path.isfile("model/wordvector.{0}.pickle".format(name)) is False:
        mapp_target_vocab = defaultdict(float)
        for key, value in mapp_vocab.items():
            tokens = value.split("+")[1:]
            for token in tokens:
                mapp_target_vocab[token] += 1

        word_vecs = {}
        vocab_keys = vocab.keys()
        mapp_vocab_keys = mapp_target_vocab.keys()

        fi = open(fname, "r")
        if fname.find("w2v")!=-1 or fname.find("word2vec")!=-1:
            header = fi.readline()
        for line in fi:
            tokens = line.split()
            if isinstance(tokens[0], unicode):
                tokens[0] = unicodedata.normalize('NFKD', tokens[0]).encode('ascii', 'ignore')
            if tokens[0] in vocab_keys or tokens[0] in mapp_vocab_keys:
                word_vecs[tokens[0]] = np.array(tokens[1:], dtype=np.float32)
        fi.close()
        with codecs.open("model/wordvector.{0}.pickle".format(name), mode="wb") as f:
            pickle.dump(word_vecs, f)
        return word_vecs
    else:
        with codecs.open("model/wordvector.{0}.pickle".format(name), mode="rb") as f:
            word_vecs = pickle.load(f)
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
            pickle.dump(vocab, f)
        return vocab
    else:
        with codecs.open(vocab_path, mode="rb") as f:
            vocab = pickle.load(f)
        return vocab



def load_data(name="laptop", taskname = "ATEPC"):
    if name == "laptops":
        print("--Laptops--")
        revs_train, vocab_train = read_ATEPC("data/Laptops_Train_v2.xml", taskname=taskname)
        revs_test, vocab_test = read_ATEPC("data/Laptops_Test_Gold.xml", taskname=taskname)
    else:
        print("--Restaurant--")
        revs_train, vocab_train = read_ATEPC("data/Restaurants_Train_v2.xml", taskname=taskname)
        revs_test, vocab_test = read_ATEPC("data/Restaurants_Test_Gold.xml", taskname=taskname)

    print("Train, Test size: ", len(revs_train), len(revs_test))
    vocab = defaultdict(float)
    for key, value in vocab_train.items() + vocab_test.items():
        vocab[key] += value
    print("Vocabulary size: ", len(vocab))
    return revs_train, revs_test, vocab

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
                    print (key)
                    print( " ".join(["{0}_{1}".format(x[1], x[2]) for x in sorted_xxx[:5]]))
                    print()
        with codecs.open("model/mappvocab.{0}.pickle".format(name), mode="wb") as f:
            pickle.dump(mapp_vocab, f)
            return mapp_vocab
    else:
        with codecs.open("model/mappvocab.{0}.pickle".format(name), mode="rb") as f:
            return pickle.load(f)

def convert_2_tsv(revs, fname, taskname="ATEPC"):
    with open(fname, mode="w") as f:
        for rev in revs:
            words = rev["words"]
            ys = rev["y"]
            reversed_ys = reverse_y(ys, taskname)
            for word, y in zip(words, reversed_ys):
                f.write("{0}\t{1}\n".format(word, y))
            f.write("\n")


def data_generation():
    dataname = "restaurants"

    w2v_vocab = load_vocab_w2v("C:\\hynguyen\\Data\\vector\\glove.42B.300d\\glove.42B.300d.txt")
    print (len(w2v_vocab))

    revs_train, revs_test, vocab = load_data(dataname, taskname="ATEPC2")
    print(len(vocab))

    mapp_vocab = mapping_vocab(w2v_vocab, vocab, use_editdistance=True, name = dataname)
    print ("ahihi")


    print (len(vocab), len(mapp_vocab))

    words_vector = load_bin_vec("C:\\hynguyen\\Data\\vector\\glove.42B.300d\\glove.42B.300d.txt", vocab, mapp_vocab)
    print (len(words_vector.keys()))

    taskname = "ATEPC"
    convert_2_tsv(revs_test, "data/{0}.{1}.test.tsv".format(dataname, taskname), taskname=taskname)
    convert_2_tsv(revs_train, "data/{0}.{1}.train.tsv".format(dataname, taskname), taskname=taskname)

    taskname = "ATEPC2"
    convert_2_tsv(revs_test, "data/{0}.{1}.test.tsv".format(dataname, taskname), taskname=taskname)
    convert_2_tsv(revs_train, "data/{0}.{1}.train.tsv".format(dataname, taskname), taskname=taskname)

def filter_conflict_sentence_data():
    filter_conflict_sentence("data/Laptops_Test_Gold.xml")
    filter_conflict_sentence("data/Laptops_Train_v2.xml")
    filter_conflict_sentence("data/Restaurants_Test_Gold.xml")
    filter_conflict_sentence("data/Restaurants_Train_v2.xml")

if __name__ == "__main__":
    filter_conflict_sentence_data()



