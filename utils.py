import os
from collections import OrderedDict, defaultdict
import nltk
import numpy as np
UNK = '<UNK>'
PAD = '<PAD>'



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
    return sents, labels, pred_labels


class FeatureExtractor(object):
    def __init__(self, name="Feature", features_dict = None):
        self.name = name
        self.features_dict = OrderedDict() if features_dict is None else features_dict

class POSExtractor(FeatureExtractor):
    def __init__(self, name="POS Feature", features_dict = None):
        super(POSExtractor, self).__init__(name, features_dict)
        self.savepath = os.path.join(os.getcwd(), "models/features_pos.pickle")
        self.tag_set = None

    def create_features_dict(self, fea_dict):
        self.features_dict[PAD] = 0
        self.features_dict[UNK] = 1
        counter = len(self.features_dict)
        for key, value in fea_dict.items():
            if value > 5:
                self.features_dict[key] = counter
                counter += 1
        return self.features_dict

    def collect_pos_features(self, sents, tagset=None):
        fea_dict = defaultdict(int)
        for words in sents:
            words_pos = nltk.pos_tag(words, tagset=tagset)
            list_pos = set([word_pos[1] for word_pos in words_pos])
            for pos in list_pos:
                fea_dict[pos]+=1
        return self.create_features_dict(fea_dict)

    def fit(self, X):
        return self.collect_pos_features(X, tagset=None)

    def words_2_posid(self, words):
        result = []
        words_pos = nltk.pos_tag(words, self.tag_set)
        for word_pos in words_pos:
            if word_pos[1] in self.features_dict:
                result.append(self.features_dict[word_pos[1]])
            else:
                result.append(self.features_dict[UNK])
        return result

    def sents_2_posid(self, sents):
        result = []
        for sent in sents:
            words = sent
            result.append(self.words_2_posid(words))
        return result

    def words_2_onehost(self, words):
        result = np.zeros((len(words),len(self.features_dict)), dtype=np.int32)
        words_posid = self.words_2_posid(words)
        for i in range(len(words)):
            result[i][words_posid[i]] = 1
        return result

    def sent_2_onehost(self, sents):
        result = []
        for sent in sents:
            words = sent
            result.append(self.words_2_posid(words))
        aaa, _ = pad_sequences(result, pad_tok=0)
        result_array = np.zeros((len(sents), len(aaa[0]), len(self.features_dict)), dtype=np.float32)
        for i in range(len(aaa)):
            for j in range(len(aaa[i])):
                k = aaa[i][j]
                if k != 0:
                    result_array[i, j, k] = 1
        return result_array

    def transform(self, X, one_host = False):
        if one_host is True:
            return self.sent_2_onehost(X)
        else:
            poses = self.sents_2_posid(X)
            poses, length = pad_sequences(poses, pad_tok=0)
            return np.array(poses, dtype=np.int32)

class UniversalPOSExtractor(POSExtractor):
    def __init__(self, name="POS Feature", features_dict = None):
        super(UniversalPOSExtractor, self).__init__(name, features_dict)
        self.savepath = os.path.join(os.getcwd(), "models/features_unipos.pickle")
        self.tag_set = "universal"

    def fit(self, X):
        return self.collect_pos_features(X, self.tag_set)


class BingLiu2014FeatureExtractor(FeatureExtractor):
    def __init__(self, name="Bingliu 2014 lexicon Feature", features_dict = None):
        super(BingLiu2014FeatureExtractor, self).__init__(name, features_dict)
        nega_path = os.path.join(os.getcwd(), "data/lexicon/2014BingLiu/negative-words.txt")
        posi_path = os.path.join(os.getcwd(), "data/lexicon/2014BingLiu/positive-words.txt")
        with open(nega_path, mode="r") as f:
            for line in f:
                if line[0].isalnum() is True:
                    self.features_dict[line.strip()] = -1

        with open(posi_path, mode="r") as f:
            for line in f:
                if line[0].isalnum() is True:
                    self.features_dict[line.strip()] = 1

    def transform(self, X, one_host=False):
        results = []
        for words in X:
            result = []
            for word in words:
                if word in self.features_dict.keys():
                    result.append(self.features_dict[word])
                else:
                    result.append(0)
            results.append(result)
        padded_result = pad_sequences(results, pad_tok=0)
        return padded_result

class SentiWordNetFeatureExtractor(FeatureExtractor):
    def __init__(self, name="Bingliu 2014 lexicon Feature", features_dict = None):
        super(SentiWordNetFeatureExtractor, self).__init__(name, features_dict)

        wordnet_path = os.path.join(os.getcwd(), "data/lexicon/SentiWordNet_3.0.0_20130122.txt")
        with open(wordnet_path, mode="r") as f:
            for line in f:
                if line[0].isalnum() is True:
                    tokens = line.split("\t")
                    nega_score = float(tokens[3])
                    posi_score = float(tokens[2])
                    terms = [token.split("#")[0] for token in tokens[4].split()]
                    obj_score = 1 - (posi_score + nega_score)
                    if posi_score > obj_score and posi_score > nega_score:
                        ss = posi_score
                    elif nega_score > obj_score and nega_score > posi_score:
                        ss = - nega_score
                    else:
                        ss = 0
                    for term in terms:
                        self.features_dict[term] = ss

    def transform(self, X, one_host=False):
        results = []
        for words in X:
            result = []
            for word in words:
                if word in self.features_dict.keys():
                    result.append(self.features_dict[word])
                else:
                    result.append(0)
            results.append(result)
        padded_result = pad_sequences(results, pad_tok=0)
        return padded_result

class NegationFeatureExtractor(FeatureExtractor):
    def __init__(self, name="Negation Feature", features_dict = None):
        super(NegationFeatureExtractor, self).__init__(name, features_dict)
        negation_path = os.path.join(os.getcwd(), "data/lexicon/negation_words.txt")
        with open(negation_path, mode="r") as f:
            for line in f:
                self.features_dict[line.strip()] = 0

    def transform(self, X, one_host=False):
        results = []
        for words in X:
            result = []
            for word in words:
                if word in self.features_dict.keys():
                    result.append(1)
                else:
                    result.append(0)
            results.append(result)
        padded_result = pad_sequences(results, pad_tok=0)
        return padded_result

class HandcraftFeatureExtractor(FeatureExtractor):
    def __init__(self, name="Handcraft Feature", features_dict = None):
        super(HandcraftFeatureExtractor, self).__init__(name, features_dict)
        self.negafe = NegationFeatureExtractor()
        self.bingfe = BingLiu2014FeatureExtractor()
        self.sentife = SentiWordNetFeatureExtractor()

    def transform(self, X, one_host=False):
        aaa1, _ = self.negafe.transform(X)
        aaa2, _ = self.negafe.transform(X)
        aaa3, _ = self.negafe.transform(X)
        aaa1 = np.array(aaa1, dtype=np.float32).reshape((len(aaa1), len(aaa1[0]),1 ))
        aaa2 = np.array(aaa2, dtype=np.float32).reshape((len(aaa2), len(aaa2[0]),1 ))
        aaa3 = np.array(aaa3, dtype=np.float32).reshape((len(aaa3), len(aaa3[0]),1 ))
        result = np.concatenate((aaa1, aaa2, aaa3), -1)
        return result

if __name__ == '__main__':
    sents, labels, pred_labels = collect_data_from_tsv("data/restaurants.ATEPC2.train.tsv")
    print(len(sents), len(labels), len(pred_labels))

    unife = UniversalPOSExtractor()
    unife.fit(sents)
    aaa  = unife.transform(sents, one_host=True)
    print("ahihi")
    print(aaa.shape)
