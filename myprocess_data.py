import os
from collections import Counter
import xml.etree.ElementTree as ET

import numpy as np
import pickle
from collections import defaultdict
import sys, re
import pandas as pd
import editdistance
import codecs
import unicodedata
from utils import normalize_str, parse_conll_dept

try:
    import nltk
    # dependency parser
    dept_parser = nltk.CoreNLPDependencyParser(url='http://localhost:9000')

except ImportError:
    print( '[!] You need to install nltk (http://nltk.org/index.html)')
    print( "You also need run CoreNLP server at http://localhost:9000")

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

def match(words_a, words_b):
    for i in range(len(words_a)):
        if words_a[i] != words_b[i]:
            return False
    return True

def identify(words, as_words, from_idx):
    results = []
    for i in range(len(words) - len(as_words) +1):
        if match(words[i:i+len(as_words)], as_words):
            results.append((i, i + len(as_words)))
    if len(results) == 0:
        print(words)
        print(as_words)
        raise ( "Sai roi")
    elif len(results) == 1:
        return results[0]
    else:
        def find_best_result(results, from_idx):
            min_dis = 1000
            best_result = None
            for i in range(0,len(results)):
                result = results[i]
                first_idx = result[0]
                res_idx = sum([len(words[j]) + 1 for j in range(first_idx)])
                if abs(from_idx - res_idx) < min_dis:
                    min_dis = abs(from_idx - res_idx)
                    best_result = results[i]
            return best_result
        best_result = find_best_result(results, from_idx)
        return best_result

def adjust_y(Y, from_idx, to_idx, polarity):

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

    if polarity == "positive":
        Y[from_idx] = 5
        for i in range(from_idx+1, to_idx):
            Y[i] = 6
    elif polarity == "negative":
        Y[from_idx] = 1
        for i in range(from_idx + 1, to_idx):
            Y[i] = 2
    elif polarity == "neutral":
        Y[from_idx] = 3
        for i in range(from_idx + 1, to_idx):
            Y[i] = 4
    else:
        raise ("Bad polarity")
    return Y

def gen_sequence_label(pre_text, aspect_terms):

    dept_iter = dept_parser.raw_parse(pre_text)
    dept = next(dept_iter)
    dept_str = dept.to_conll(4)
    conll_dept = parse_conll_dept(dept_str=dept_str)
    words = list(dept_parser.tokenize(pre_text))

    if len(conll_dept) != len(words):
        print (words)

    Y = [0]*len(words)
    for aspect_term in aspect_terms:
        as_words = list(dept_parser.tokenize(aspect_term["asp_term"]))
        from_idx, to_idx = identify(words, as_words, aspect_term["from_idx"])
        polarity = aspect_term["polarity"]
        adjust_y(Y, from_idx, to_idx, polarity)
    if len(words) != len(Y):
        raise ("Length are not equal !")
    return words, Y, conll_dept

def read_ATEPC(fname, cv=10 , clean_string=True):
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
                from_idx = int(asp_term_tag.get('from'))
                to_idx = int(asp_term_tag.get('to'))
                tmp = "xxx " + text[from_idx:to_idx] + " xxx"
                asp_term = normalize_str(tmp, clean_string=clean_string)
                asp_term = " ".join(asp_term.split()[1:-1])
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
            pre_text = normalize_str(text, clean_string=clean_string)
            X , Y, dept = gen_sequence_label(pre_text, aspect_terms=asp_terms)

        revs.append({
                "X": X,
                "Y": Y,
                "dept": dept,
                "no_aspterms": len(asp_terms),
                "no_words": len(X),
                "split": np.random.randint(0, cv)})
        uni_words = set(X)
        for word in uni_words:
            vocab[word] += 1
    return revs, vocab

def reverse_y(ys):
    result = []
    for y in ys:
        if y == 0: result.append("O")
        elif y == 1: result.append("B-NEG")
        elif y == 2: result.append("I-NEG")
        elif y == 3: result.append("B-NEU")
        elif y == 4: result.append("I-NEU")
        elif y == 5: result.append("B-POS")
        elif y == 6: result.append("I-POS")
        else: result.append("<UNK_TAG>")
    return result

def convert_2_tsv(revs, fname):
    with open(fname, mode="w") as f:
        for rev in revs:
            words = rev["X"]
            ys = rev["Y"]
            reversed_ys = reverse_y(ys)
            for word, y in zip(words, reversed_ys):
                f.write("{0}\t{1}\n".format(word, y))
            f.write("\n")

def convert_2_dept_tsv(revs, fname):
    with open(fname, mode="w") as f:
        for rev in revs:
            dept = rev["dept"]
            for tokens in dept:
                f.write("{0}\t{1}\t{2}\t{3}\n".format(tokens[0], tokens[1], tokens[2], tokens[3]))
            f.write("\n")

def load_data(name="laptop"):
    if name == "laptops":
        print("--Laptops--")
        revs_train, vocab_train = read_ATEPC("data/Laptops_Train_v2.xml")
        revs_test, vocab_test = read_ATEPC("data/Laptops_Test_Gold.xml")
    else:
        print("--Restaurant--")
        revs_train, vocab_train = read_ATEPC("data/Restaurants_Train_v2.xml")
        revs_test, vocab_test = read_ATEPC("data/Restaurants_Test_Gold.xml")

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

def data_generation():
    dataname = "restaurants"

    w2v_vocab = load_vocab_w2v("C:\\hynguyen\\Data\\vector\\glove.42B.300d\\glove.42B.300d.txt")
    print (len(w2v_vocab))

    revs_train, revs_test, vocab = load_data(dataname)
    print(len(vocab))

    mapp_vocab = mapping_vocab(w2v_vocab, vocab, use_editdistance=True, name = dataname)
    print ("ahihi")


    print (len(vocab), len(mapp_vocab))

    words_vector = load_bin_vec("C:\\hynguyen\\Data\\vector\\glove.42B.300d\\glove.42B.300d.txt", vocab, mapp_vocab)
    print (len(words_vector.keys()))

    taskname = "ATEPC2"
    convert_2_tsv(revs_test, "data/{0}.{1}.test.tsv".format(dataname, taskname))
    convert_2_tsv(revs_train, "data/{0}.{1}.train.tsv".format(dataname, taskname))

def filter_conflict_sentence_data():
    filter_conflict_sentence("data/Laptops_Test_Gold.xml")
    filter_conflict_sentence("data/Laptops_Train_v2.xml")
    filter_conflict_sentence("data/Restaurants_Test_Gold.xml")
    filter_conflict_sentence("data/Restaurants_Train_v2.xml")

if __name__ == "__main__":
    dataname = "laptops"
    taskname = "ATEPC2"
    revs_train, revs_test, vocab = load_data(dataname)
    convert_2_tsv(revs_test, "data/{0}.{1}.test.tsv".format(dataname, taskname))
    convert_2_tsv(revs_train, "data/{0}.{1}.train.tsv".format(dataname, taskname))
    convert_2_dept_tsv(revs_test, "data/{0}.{1}.test.dep.tsv".format(dataname, taskname))
    convert_2_dept_tsv(revs_train, "data/{0}.{1}.train.dep.tsv".format(dataname, taskname))

    dataname = "restaurants"
    taskname = "ATEPC2"
    revs_train, revs_test, vocab = load_data(dataname)
    convert_2_tsv(revs_test, "data/{0}.{1}.test.tsv".format(dataname, taskname))
    convert_2_tsv(revs_train, "data/{0}.{1}.train.tsv".format(dataname, taskname))
    convert_2_dept_tsv(revs_test, "data/{0}.{1}.test.dep.tsv".format(dataname, taskname))
    convert_2_dept_tsv(revs_train, "data/{0}.{1}.train.dep.tsv".format(dataname, taskname))



