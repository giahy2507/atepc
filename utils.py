import os
from collections import OrderedDict, defaultdict

import numpy as np
import json
import unicodedata
import codecs
import nltk
import re
import xml.etree.ElementTree as ET
import copy


try:
    import nltk
    from nltk import wordpunct_tokenize
    from nltk.corpus import stopwords
    # dependency parser
    dept_parser = nltk.CoreNLPDependencyParser(url='http://localhost:9000')

except ImportError:
    print( '[!] You need to install nltk (http://nltk.org/index.html)')
    print( "You also need run CoreNLP server at http://localhost:9000")


def batch_iter(dataset, batch_size, shuffle=True, preprocessor=None):
    num_batches_per_epoch = int(len(dataset)/ batch_size) + 1
    def data_generator():
        """
        Generates a batch iterator for a dataset.
        """
        data = np.array(dataset)
        data_size = len(data)
        while True:
            # Shuffle the data at each epoch
            shuffle_indices = list(range(data_size))
            if shuffle:
                np.random.shuffle(shuffle_indices)
            for batch_num in range(num_batches_per_epoch):
                start_index = batch_num * batch_size
                end_index = min((batch_num + 1) * batch_size, data_size)
                #TODO
                X, y  = zip(*data[shuffle_indices[start_index:end_index]])
                if preprocessor:
                    yield preprocessor.transform(list(X), list(y))
                else:
                    yield X, y
    return num_batches_per_epoch, data_generator()

def _pad_sequences(sequences, pad_tok, max_length):
    """
    Args:
        sequences: a generator of list or tuple.
        pad_tok: the char to pad with.

    Returns:
        a list of list where each sublist has same length.
    """
    sequence_padded, sequence_length = [], []

    for seq in sequences:
        seq = list(seq)
        seq_ = seq[:max_length] + [pad_tok] * max(max_length - len(seq), 0)
        sequence_padded += [seq_]
        sequence_length += [min(len(seq), max_length)]

    return sequence_padded, sequence_length

def pad_sequences(sequences, pad_tok, nlevels=1):
    """
    Args:
        sequences: a generator of list or tuple.
        pad_tok: the char to pad with.

    Returns:
        a list of list where each sublist has same length.
    """
    if nlevels == 1:
        max_length = len(max(sequences, key=len))
        sequence_padded, sequence_length = _pad_sequences(sequences, pad_tok, max_length)
    elif nlevels == 2:
        max_length_word = max(len(max(seq, key=len)) for seq in sequences)
        sequence_padded, sequence_length = [], []
        for seq in sequences:
            # all words are same length now
            sp, sl = _pad_sequences(seq, pad_tok, max_length_word)
            sequence_padded += [sp]
            sequence_length += [sl]

        max_length_sentence = max(map(lambda x: len(x), sequences))
        sequence_padded, _ = _pad_sequences(sequence_padded, [pad_tok] * max_length_word, max_length_sentence)
        sequence_length, _ = _pad_sequences(sequence_length, 0, max_length_sentence)
    else:
        raise ValueError('nlevels can take 1 or 2, not take {}.'.format(nlevels))

    return sequence_padded, sequence_length

def dense_to_one_hot(labels_dense, num_classes, nlevels=1):
    """Convert class labels from scalars to one-hot vectors."""
    if nlevels == 1:
        num_labels = labels_dense.shape[0]
        index_offset = np.arange(num_labels) * num_classes
        labels_one_hot = np.zeros((num_labels, num_classes), dtype=np.int32)
        labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
        return labels_one_hot
    elif nlevels == 2:
        # assume that labels_dense has same column length
        num_labels = labels_dense.shape[0]
        num_length = labels_dense.shape[1]
        labels_one_hot = np.zeros((num_labels, num_length, num_classes), dtype=np.int32)
        layer_idx = np.arange(num_labels).reshape(num_labels, 1)
        # this index selects each component separately
        component_idx = np.tile(np.arange(num_length), (num_labels, 1))
        # then we use `a` to select indices according to category label
        labels_one_hot[layer_idx, component_idx, labels_dense] = 1
        return labels_one_hot
    else:
        raise ValueError('nlevels can take 1 or 2, not take {}.'.format(nlevels))

def collect_data_from_tsv(tsvfile):
    if os.path.isfile(tsvfile) == False:
        raise ("[!] Data %s not found" % tsvfile)
    # Collect sentences in tsv file
    sents, labels, pred_labels = [], [], []
    with open(tsvfile) as f:
        words, tags, preds = [], [], []
        for line in f:
            line = line.rstrip()
            if len(line) == 0 or line.startswith('-DOCSTART-'):
                if len(words) != 0:
                    sents.append(words)
                    labels.append(tags)
                    pred_labels.append(preds)
                    words, tags, preds = [], [], []
            else:
                tokens = line.split('\t')
                words.append(tokens[0])
                tags.append(tokens[1])
                if len(tokens) == 3:
                    preds.append(tokens[2])
                else:
                    preds.append("")
    return np.asarray(sents), np.asarray(labels), np.asarray(pred_labels)

def collect_dept_data_from_tsv(tsvfile):
    if os.path.isfile(tsvfile) == False:
        raise Exception("[!] Data %s not found" % tsvfile)
    # Collect sentences in tsv file
    depts = []
    depts_2 = []
    with open(tsvfile) as f:
        dept = []
        for line in f:
            line = line.rstrip()
            if len(line) == 0 or line.startswith('-DOCSTART-'):
                if len(dept) != 0:
                    # extract dependency form
                    dependents = [[] for i in range(len(dept) + 1)]
                    governor = [-1] * (len(dept) + 1)
                    for idx, tokens in enumerate(dept):
                        word, pos, governor_idx, relation = tokens[0], tokens[1], tokens[2], tokens[3]
                        governor[idx + 1] = (int(governor_idx), relation)
                        dependents[int(governor_idx)].append(idx + 1)
                    depts_2.append((dependents, governor))
                    depts.append(dept)
                    dept = []
            else:
                tokens = line.split('\t')
                dept.append(tokens)
    return np.asarray(depts_2)

def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Every dataset is lower cased except for TREC
    """
    string = string.strip(".,;:")
    string = re.sub(r"https?:\/\/(www\.)?[-a-zA-Z0-9@:%._\+~#=]{2,256}\.[a-z]{2,6}\b([-a-zA-Z0-9@:%_\+.~#?&//=]*)", " ", string)
    string = re.sub(r"`", "'", string)
    string = re.sub(r"\'\'", "\"", string)
    string = re.sub(r"[^A-Za-z0-9(),\!\?\'\`\/\.\;\$\"]", " ", string)
    string = re.sub(r"\/", " / ", string)
    string = re.sub(r"[\?]{2,10}", " ? ", string)
    string = re.sub(r"[\.]{2,10}", " ... ", string)
    string = re.sub(r"[\-]{2,10}", " - ", string)
    string = re.sub(r"[\!]{2,10}", " ! ", string)
    words = list(dept_parser.tokenize(string.lower()))
    return " ".join(words)

def normalize_str(string, clean_string = True):
    if clean_string is True:
        processed_string = clean_str(string)
    else:
        processed_string = string.lower()

    # str start with "-\w" pattern
    if len(processed_string) > 2:
        if processed_string[0].isalnum() is False:
            processed_string = processed_string[0] + " " + processed_string[1:]
    return processed_string

def search_all(pattern, string):
    result = []
    finded = re.finditer(pattern=pattern, string=string)
    for find in finded:
        result.append(find.regs[0])
    return result

def match(words_a, words_b):
    for i in range(len(words_a)):
        if words_a[i] != words_b[i]:
            return False
    return True

def get_aspecterm( x, y):
    result = []
    i = 0
    y.append("O")
    while i < len(y):
        if y[i].split("-")[0] == "B":
            aspecterm = []
            aspecterm.append(x[i])
            approx_pos = sum([len(word) + 1 for word in x[:i]])
            i += 1
            while y[i].split("-")[0] == "I" and i < len(y):
                aspecterm.append(x[i])
                i += 1
            result.append({"aspect_term": aspecterm, "approx_pos": approx_pos})
        else:
            i += 1
    return result

def detect_english(text):
    tokens = wordpunct_tokenize(text)
    words = [word.lower() for word in tokens]
    stopwords_set = set(stopwords.words("english"))
    words_set = set(words)
    common_elements = words_set.intersection(stopwords_set)
    if len(common_elements) < 6:
        return False
    else:
        return True

def prepare_yelp_data(data_path = ""):
    fo = codecs.open(data_path+".fil", mode="w", encoding='utf-8')
    counter = 1
    with codecs.open(data_path, mode="r", errors='ignore',encoding='utf-8') as f:
        for x in f:
            if x[0] != "{":
                continue
            # if counter == 100:
            #     break
            if x.find("\"text\"") != -1:
                # TIPS
                counter += 1
                a = json.loads(x, encoding="utf-8")
                text = a["text"]
                is_en = detect_english(text)
                sents = text.split("\n")
                for sent in sents:
                    words = sent.split()
                    if is_en is True and len(words) > 1:
                        fo.write(sent + "\n")
                if counter % 200000 == 0:
                    print(counter)
    fo.close()

def prepare_amazone_data(data_path = ""):
    fo = codecs.open(data_path + ".fil", mode="w", encoding='utf-8')
    counter = 0
    with codecs.open(data_path, mode="r", errors='ignore', encoding='utf-8') as f:
        sents_buffer = []
        flag = False
        for x in f:
            if x.strip() == "<review_text>":
                flag = True
            elif x.strip() == "</review_text>":
                flag = False
                for sent in sents_buffer:
                    fo.write(sent + "\n")
                sents_buffer = []
            elif flag is True:
                if x.strip() == "":
                    continue
                sents_buffer.append(". ".join(x.strip().split("\n")))
                counter +=1
        if counter % 200000:
            print(counter)
    fo.close()

def pre_data(file_name = "/home/s1610434/Documents/Data/yelp_dataset/yelp_dataset.fil"):
    fo = open(file_name + ".pre", mode="w")
    print ("Preprocess: ", file_name)
    with open(file_name, mode="r") as f:
        counter = 1
        for line in f:
            is_en = detect_english(line)
            if is_en is True:
                clean_line =  normalize_str(line)
                fo.write(clean_line + "\n")
            if counter % 100000 == 0:
                print(counter)
            counter+=1
    fo.close()

def pre_pos(file_name = "/home/s1610434/Documents/Data/yelp_dataset/yelp_dataset.fil"):
    fo = open(file_name + ".unipos", mode="w")
    fo2 = open(file_name + ".pos", mode="w")
    print ("POS: ", file_name)
    with open(file_name, mode="r") as f:
        counter = 1
        for line in f:
            words = line.split()
            pos_word1 = nltk.pos_tag(words, tagset= "universal")
            pos_word2 = nltk.pos_tag(words)
            pos1 = [token[1] for token in pos_word1]
            pos2 = [token[1] for token in pos_word2]
            fo2.write(" ".join(pos2) + "\n")
            fo.write(" ".join(pos1) + "\n")
            if counter % 200000 == 0:
                print(counter)
            counter += 1
    fo.close()
    fo2.close()

def pre_char(file_name = "/home/s1610434/Documents/Data/yelp_dataset/yelp_dataset.fil"):
    fo = open(file_name + ".char", mode="w")
    print ("Char: ", file_name)
    with open(file_name, mode="r") as f:
        counter = 1
        for line in f:
            rep_line = line.strip()
            rep_line = " ".join(rep_line)
            rep_line = rep_line.replace("   ", " <space> ")
            fo.write(rep_line + "\n")
            if counter % 200000 == 0:
                print(counter)
            counter += 1
    fo.close()

import pickle

import argparse

def pre_embedding():
    parser = argparse.ArgumentParser()
    parser.add_argument('-ifile', type=str)
    args = parser.parse_args()
    data_file = args.ifile
    if os.path.isfile(data_file) is False:
        print("THis is not a file")
    pre_data(data_file)
    pre_pos(data_file + ".pre")
    pre_char(data_file + ".pre")

def parse_conll_dept(dept_str):
    sent_dept = []
    for line in dept_str.strip().split("\n"):
        tokens =[ re.sub(r"\'\'", "\"", token) for token in line.split("\t")]
        sent_dept.append(tokens)
    return sent_dept

from itertools import combinations

if __name__ == '__main__':

    # pre_embedding()

    # prepare_amazone_data("C:\\hynguyen\\Data\\Amazon\\unprocessed\\sorted_data\\all.review")
    # prepare_yelp_data("/home/s1610434/Documents/Data/yelp_dataset/yelp_dataset")

    # pre_data("/home/s1610434/Documents/Data/yelp_dataset/yelp_dataset.fil")
    # pre_data("/home/s1610434/Documents/Data/Amazon/amazon.all.review.fil")

    # data_file = "/home/s1610434/Documents/Data/yelp_dataset/yelp_dataset.fil"
    # pre_data(data_file)
    # pre_pos(data_file+".pre")
    # pre_char(data_file+".pre")
    #
    # data_file = "/home/s1610434/Documents/Data/Amazon/amazon.all.review.fil"
    # pre_data(data_file)
    # pre_pos(data_file + ".pre")
    # pre_char(data_file + ".pre")
    POS_dict = []
    with open('/home/s1610434/Documents/Data/Vector/AmaYelp/GloVe/glove.pos.100.txt', mode="r") as f:
        for line in f:
            word = line.split()[0]
            POS_dict.append(word)
    POS_dict = sorted(POS_dict)
    with open("models/features_pos.txt", mode="w") as f:
        for pos in POS_dict:
            f.write(pos+"\n")
