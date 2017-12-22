import nltk
from evaluation import ResultConverter
from collections import OrderedDict, defaultdict
from utils import collect_data_from_tsv, get_aspecterm, collect_dept_data_from_tsv, pad_sequences, dense_to_one_hot, batch_iter
import pickle
import numpy as np
import os
import re
from sklearn.model_selection import KFold
import copy
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.externals import joblib
import itertools

UNK = '<UNK>'
PAD = '<PAD>'

# Feature extraction
class FeatureExtractor(object):

    def __init__(self, name="Feature", features_dict = None):
        self.name = name
        self.features_dict = OrderedDict() if features_dict is None else features_dict
        self.no_fe = 0

    def fit(self, X, Y = None):
        return

    def fit2(self, X_all, Y_all, X_train, Y_train):
        return

    def transform(self, X):
        return

    @classmethod
    def load(self, fname=""):
        with open(fname, mode="rb") as f:
            return pickle.load(f)

    def save(self, fname, data):
        with open(fname, mode="wb") as f:
            pickle.dump(data, f)

class POSExtractor(FeatureExtractor):
    def __init__(self, name="POS Feature", features_dict = None):
        super(POSExtractor, self).__init__(name, features_dict)
        self.savepath = os.path.join(os.getcwd(), "models/features_pos.txt")
        self.tag_set = None
        self.load_features_dict()

    def load_features_dict(self):
        self.features_dict[PAD] = 0
        self.features_dict[UNK] = 1
        counter = 2
        with open(self.savepath, mode="r") as f:
            for line in f:
                pos = line.strip()
                self.features_dict[pos] = counter
                counter += 1
        self.no_fe = len(self.features_dict.keys())
        return self.features_dict

    def fit(self, X, Y = None):
        return

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
        for sent, depency in sents:
            words = sent
            result.append(self.words_2_posid(words))
        return result

    def words_2_onehost(self, words):
        result = np.zeros((len(words),len(self.features_dict)), dtype=np.int32)
        words_posid = self.words_2_posid(words)
        for i in range(len(words)):
            result[i][words_posid[i]] = 1
        return result

    def sents_2_onehost(self, sents):
        result = []
        for sent, depency in sents:
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
            return self.sents_2_onehost(X)
        else:
            poses = self.sents_2_posid(X)
            poses, length = pad_sequences(poses, pad_tok=0)
            return np.array(poses, dtype=np.int32)

class UniversalPOSExtractor(POSExtractor):
    def __init__(self, name="POS Feature", features_dict = None):
        super(UniversalPOSExtractor, self).__init__(name, features_dict)
        self.savepath = os.path.join(os.getcwd(), "models/features_unipos.txt")
        self.tag_set = "universal"
        self.load_features_dict()

class BingLiu2014FeatureExtractor(FeatureExtractor):
    def __init__(self, name="Bingliu 2014 lexicon Feature", features_dict = None, binary = False):
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
        self.no_fe = 1
        self.binary = binary

    def transform(self, X):
        results = []
        fea_keys = self.features_dict.keys()
        for words, depency in X:
            result = []
            for word in words:
                if word in fea_keys:
                    if self.binary:
                        result.append(1)
                    else:
                        result.append(self.features_dict[word])
                else:
                    result.append(0)
            results.append(result)
        padded_result = pad_sequences(results, pad_tok=0)
        return padded_result

class SentiWordNetFeatureExtractor(FeatureExtractor):
    def __init__(self, name="Bingliu 2014 lexicon Feature", features_dict = None, binary = False):
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
        self.no_fe = 1
        self.binary = binary

    def transform(self, X):
        results = []
        fea_keys = self.features_dict.keys()
        for words, depency in X:
            result = []
            for word in words:
                if word in fea_keys:
                    if self.binary:
                        result.append(1)
                    else:
                        result.append(self.features_dict[word])
                else:
                    result.append(0)
            results.append(result)
        padded_result = pad_sequences(results, pad_tok=0)
        return padded_result

class NegationFeatureExtractor(FeatureExtractor):
    def __init__(self, name="Negation Feature", features_dict = None, binary = False):
        super(NegationFeatureExtractor, self).__init__(name, features_dict)
        negation_path = os.path.join(os.getcwd(), "data/lexicon/negation_words.txt")
        with open(negation_path, mode="r") as f:
            for line in f:
                self.features_dict[line.strip()] = 0
        self.no_fe = 1
        self.binary=binary

    def transform(self, X):
        results = []
        fea_keys = self.features_dict.keys()
        for words, depency in X:
            result = []
            for word in words:
                if word in fea_keys:
                    result.append(1)
                else:
                    result.append(0)
            results.append(result)
        padded_result = pad_sequences(results, pad_tok=0)
        return padded_result

class NameListFeatureExtractor(FeatureExtractor):
    def __init__(self, name="POS Feature", features_dict = None, c1 = 1, c2 = 1, t=0.2):
        super(NameListFeatureExtractor, self).__init__(name, features_dict)
        self.savepath = os.path.join(os.getcwd(), "models/features_namelist.pickle")
        self.tag_set = None
        self.features_dict2 = OrderedDict()
        self.c1 = c1
        self.c2 = c2
        self.t = t
        self.no_fe = 2
        """
        2 settings: 
        - like Toh and Wang
        - first feature like Toh and Wang, second feature: get probability of word in aspect term.
        """

    def match(self, words_a, words_b):
        for i in range(len(words_a)):
            if words_a[i] != words_b[i]:
                return False
        return True

    def identify2(self, words, as_words):
        results = []
        for i in range(len(words) - len(as_words) + 1):
            if self.match(words[i:i + len(as_words)], as_words):
                results.append((i, i + len(as_words)))
        return results

    def collect_namelist_features(self, X, Y = None):
        vocab = defaultdict(float)
        vocab_asp = defaultdict(float)
        for x,y in zip(X,Y):
            # Vocab for all
            words, depency = x
            for word in set(words):
                pre_word = re.sub(r"\d{1,10}", "0", word)
                vocab[pre_word] +=1

            aspectterms = get_aspecterm(words, y)
            for aspectterm in aspectterms:
                aspectterm_str = " ".join(aspectterm["aspect_term"])
                aspectterm_str = re.sub(r"\d{1,10}", "0", aspectterm_str)
                # Vocab for only aspect term
                for term in aspectterm_str.split():
                    vocab_asp[term] +=1

                if aspectterm_str in self.features_dict:
                    self.features_dict[aspectterm_str] +=1
                else:
                    self.features_dict[aspectterm_str] = 1
        self.vocab = OrderedDict(sorted(vocab.items(), key=lambda item: item[1], reverse=True))
        self.features_dict = OrderedDict(sorted(self.features_dict.items(), key=lambda item: item[1], reverse=True))
        self.features_dict2 = OrderedDict(sorted(vocab_asp.items(), key=lambda item: item[1], reverse=True))
        fe1_dict = list(self.features_dict.keys())
        fe2_dict = list(self.features_dict2.keys())

        #save to see
        self.t2_dict = OrderedDict()
        self.features_dict2_save = copy.deepcopy(self.features_dict2)
        self.vocab_of_fe2_dict = OrderedDict()
        for key in fe2_dict:
            self.vocab_of_fe2_dict[key] = self.vocab[key]

        # Filter self.feature_dict1
        for key in fe1_dict:
            if self.features_dict[key] < self.c1:
                self.features_dict.pop(key)

        # filter self.feature_dict2
        for key in fe2_dict:
            if self.features_dict2[key] < self.c2:
                self.features_dict2.pop(key)
            else:
                p_aspterm = float(self.features_dict2[key])/float(self.vocab[key])
                self.t2_dict[key] = p_aspterm
                if p_aspterm < self.t:
                    self.features_dict2.pop(key)

        return self.features_dict

    def fit(self, X, Y = None):
        return self.collect_namelist_features(X, Y)

    def transform(self, X):
        results1 = []
        results2 = []
        fea_keys1 = self.features_dict.keys()
        fea_keys2 = self.features_dict2.keys()
        for words, depency in X:
            result1 = [0]*len(words)
            result2 = [0]*len(words)
            pre_words = [re.sub(r"\d{1,10}", "0", word) for word in words]
            for key in fea_keys1:
                asp_words = key.split()
                idents_idx = self.identify2(pre_words, asp_words)
                if len(idents_idx) > 0:
                    for ident_idx in idents_idx:
                        from_idx, to_idx = ident_idx
                        for j in range(from_idx, to_idx):
                            result1[j]=1
            for idx, word in enumerate(pre_words):
                if word in fea_keys2:
                    result2[idx] = 1
            results1.append(result1)
            results2.append(result2)
        padded_result1, sequence_length  = pad_sequences(results1, pad_tok=0)
        padded_result2, _ = pad_sequences(results2, pad_tok=0)
        padded_result1_np = np.array(padded_result1, dtype=np.float32).reshape(len(padded_result1), len(padded_result1[0]), 1)
        padded_result2_np = np.array(padded_result2, dtype=np.float32).reshape(len(padded_result2), len(padded_result2[0]), 1)
        xxx = np.concatenate((padded_result1_np, padded_result2_np), axis=-1)
        return xxx, sequence_length

class DependencyRelationExtraction(FeatureExtractor):
    def __init__(self, name="Dependency Relation Feature", features_dict=None, c1=1, c2=1, t=0.2):
        super(DependencyRelationExtraction, self).__init__(name, features_dict)
        self.savepath = os.path.join(os.getcwd(), "models/features_dency.pickle")

    def transform(self, X):
        results1 = []
        results2 = []
        feature_set1 = ["amod", "nsubj", "dep"]
        feature_set2 = ["dobj", "nsubj", "dep"]
        for sent, depency in X:
            result1 = [0] * len(sent)
            result2 = [0] * len(sent)
            dependents, governor = depency
            for i in range(1, len(governor)):
                gov_idx, relation = governor[i]
                if relation in feature_set2:
                    result2[i - 1] = 1
                if relation in feature_set1:
                    result1[gov_idx - 1] = 1
            results1.append(result1)
            results2.append(result2)

        padded_result1, sequence_length = pad_sequences(results1, pad_tok=0)
        padded_result2, _ = pad_sequences(results2, pad_tok=0)
        padded_result1_np = np.array(padded_result1, dtype=np.float32).reshape(len(padded_result1),
                                                                               len(padded_result1[0]), 1)
        padded_result2_np = np.array(padded_result2, dtype=np.float32).reshape(len(padded_result2),
                                                                               len(padded_result2[0]), 1)
        xxx = np.concatenate((padded_result1_np, padded_result2_np), axis=-1)
        return xxx, sequence_length

class HandcraftFeatureExtractor(FeatureExtractor):
    def __init__(self, name="Handcraft Feature", features_dict = None, hand_features = None):
        super(HandcraftFeatureExtractor, self).__init__(name, features_dict)
        self.negafe = NegationFeatureExtractor()
        self.bingfe = BingLiu2014FeatureExtractor()
        self.bingbinfe = BingLiu2014FeatureExtractor(binary=True)
        self.sentife = SentiWordNetFeatureExtractor()
        self.namelistfe = NameListFeatureExtractor()
        self.dencyfe = DependencyRelationExtraction()
        self.uniposfe = UniversalPOSExtractor()
        self.posfe = POSExtractor()
        self.hand_features = hand_features

    def fit(self, X, Y = None):
        self.uniposfe.fit(X)
        self.posfe.fit(X)
        self.namelistfe.fit(X,Y)

    def transform(self, X):
        # Negation
        NEGAT, _ = self.negafe.transform(X)
        NEGAT = np.array(NEGAT, dtype=np.float32).reshape((len(NEGAT), len(NEGAT[0]), 1))

        # Sentiment Score Bingliu
        BING, _ = self.bingfe.transform(X)
        BING = np.array(BING, dtype=np.float32).reshape((len(BING), len(BING[0]), 1))
        BINGBIN, _ = self.bingbinfe.transform(X)
        BINGBIN = np.array(BINGBIN, dtype=np.float32).reshape((len(BINGBIN), len(BINGBIN[0]), 1))

        # Sentiment Score SentimentWordnet
        SWN, _ = self.sentife.transform(X)
        SWN = np.array(SWN, dtype=np.float32).reshape((len(SWN), len(SWN[0]), 1))

        # POS tag feature, one host
        POS = self.posfe.transform(X, one_host=True)
        UNIPOS = self.uniposfe.transform(X, one_host=True)

        # Name List
        NAMEL, _ = self.namelistfe.transform(X)

        # Dependency Relation
        DEPENCY, _ = self.dencyfe.transform(X)

        mapping_dict = {"POS": POS, "UNIPOS": UNIPOS , "NEGAT": NEGAT, "BING": BING, "BINGBIN": BINGBIN , "SWN": SWN, "NAMEL": NAMEL, "DEPENCY": DEPENCY}

        if self.hand_features is None:
            result = np.concatenate((POS, NEGAT, BING, SWN, NAMEL, DEPENCY), -1)
        else:
            result = mapping_dict[self.hand_features[0]]
            for feature in self.hand_features[1:]:
                if feature in mapping_dict.keys():
                    result = np.concatenate((result,mapping_dict[feature]), -1)
                else:
                    print("Your feature is missing !!: {0}".format(feature))
        return result



# PreProcessor
# WC
class WordCharPreprocessor(BaseEstimator, TransformerMixin):

    def __init__(self,
                 lowercase=True,
                 num_norm=True,
                 char_feature=True,
                 vocab_init=None,
                 padding=True,
                 return_lengths=True):

        self.lowercase = lowercase
        self.num_norm = num_norm
        self.char_feature = char_feature
        self.padding = padding
        self.return_lengths = return_lengths
        self.vocab_word = None
        self.vocab_char = None
        self.vocab_tag = None
        self.vocab_init = vocab_init or {}

    def fit(self, X, Y):
        vocab_word =  {PAD: 0, UNK: 1}
        vocab_char = {PAD: 0, UNK: 1}
        vocab_tag = {PAD: 0}

        for words, depency in X:
            for w in words:
                w = self._lower(w)
                w = self._normalize_num(w)
                if w not in vocab_word:
                    vocab_word[w] = len(vocab_word)

                if not self.char_feature:
                    continue
                for c in w:
                    if c not in vocab_char:
                        vocab_char[c] = len(vocab_char)

        for t in itertools.chain(*Y):
            if t not in vocab_tag:
                vocab_tag[t] = len(vocab_tag)

        self.vocab_word = vocab_word
        self.vocab_char = vocab_char
        self.vocab_tag = vocab_tag
        return self

    def transform(self, X, y=None):
        words = []
        chars = []
        lengths = []
        for sent, depency in X:
            word_ids = []
            char_ids = []
            lengths.append(len(sent))
            for w in sent:
                if self.char_feature:
                    char_ids.append(self._get_char_ids(w))

                w = self._lower(w)
                w = self._normalize_num(w)
                if w in self.vocab_word:
                    word_id = self.vocab_word[w]
                else:
                    word_id = self.vocab_word[UNK]
                word_ids.append(word_id)

            words.append(word_ids)
            if self.char_feature:
                chars.append(char_ids)

        if y is not None:
            y = [[self.vocab_tag[t] for t in sent] for sent in y]

        if self.padding:
            sents, y = self.pad_sequence(words, chars, y)
        else:
            sents = [words, chars]

        if self.return_lengths:
            lengths = np.asarray(lengths, dtype=np.int32)
            lengths = lengths.reshape((lengths.shape[0], 1))
            sents.append(lengths)

        return (sents, y) if y is not None else sents

    def inverse_transform(self, y):
        indice_tag = {i: t for t, i in self.vocab_tag.items()}
        return [indice_tag[y_] for y_ in y]

    def _get_char_ids(self, word):
        return [self.vocab_char.get(c, self.vocab_char[UNK]) for c in word]

    def _lower(self, word):
        return word.lower() if self.lowercase else word

    def _normalize_num(self, word):
        if self.num_norm:
            return re.sub(r"\d{1,10}", "0", word)
        else:
            return word

    def pad_sequence(self, word_ids, char_ids, labels=None):
        if labels:
            labels, _ = pad_sequences(labels, 0)
            labels = np.asarray(labels)
            labels = dense_to_one_hot(labels, len(self.vocab_tag), nlevels=2)

        word_ids, sequence_lengths = pad_sequences(word_ids, 0)
        word_ids = np.asarray(word_ids)

        if self.char_feature:
            char_ids, word_lengths = pad_sequences(char_ids, pad_tok=0, nlevels=2)
            char_ids = np.asarray(char_ids)
            return [word_ids, char_ids], labels
        else:
            return [word_ids], labels

    def save(self, file_path):
        joblib.dump(self, file_path)

    @classmethod
    def load(cls, file_path):
        p = joblib.load(file_path)
        return p
# W
class WordPreprocessor(WordCharPreprocessor):
    def __init__(self,
                 lowercase=True,
                 num_norm=True,
                 char_feature=True,
                 vocab_init=None,
                 padding=True,
                 return_lengths=True):
        super(WordPreprocessor, self).__init__(lowercase, num_norm, char_feature, vocab_init, padding, return_lengths)

    def fit(self, X, y):
        # X = (X, X_dep)
        if isinstance(X, tuple) and len(X) == 2:
            X_1, _ = X
        else:
            X_1 = X

        # fit
        super(WordPreprocessor, self).fit(X_1,y)
        return self

    def transform(self, X, y=None):
        # X = (X, X_dep)
        if isinstance(X, tuple) and len(X) == 2:
            X_1, _ = X
        else:
            X_1 = X

        sents_y = super(WordPreprocessor, self).transform(X_1, y)
        if len(sents_y) == 2:
            sents, y = sents_y
        else:
            sents = sents_y
        sents = [sents[0], sents[-1]]
        if y is None:
            return sents
        else:
            return sents, y
# C
class CharPreprocessor(WordCharPreprocessor):
    def __init__(self,
                 lowercase=True,
                 num_norm=True,
                 char_feature=True,
                 vocab_init=None,
                 padding=True,
                 return_lengths=True):
        super(CharPreprocessor, self).__init__(lowercase, num_norm, char_feature, vocab_init, padding, return_lengths)

    def fit(self, X, y):
        super(CharPreprocessor, self).fit(X,y)
        return self

    def transform(self, X, y=None):
        sents_y = super(CharPreprocessor, self).transform(X, y)
        if len(sents_y) == 2:
            sents, y = sents_y
        else:
            sents = sents_y
        sents = [sents[1], sents[-1]]
        if y is None:
            return sents
        else:
            return sents, y
# WCP
class WordCharPosPreprocessor(WordCharPreprocessor):
    def __init__(self,
                 lowercase=True,
                 num_norm=True,
                 char_feature=True,
                 vocab_init=None,
                 padding=True,
                 return_lengths=True,
                 hand_features = None):
        super(WordCharPosPreprocessor, self).__init__(lowercase, num_norm, char_feature, vocab_init, padding, return_lengths)
        if hand_features is None:
            self.pos_extractor = POSExtractor()
        else:
            if "UNIPOS" in hand_features:
                self.pos_extractor = UniversalPOSExtractor()
            else:
                self.pos_extractor = POSExtractor()

    def fit(self, X, y):
        super(WordCharPosPreprocessor, self).fit(X,y)
        self.pos_extractor.fit(X)
        return self

    def transform(self, X, y=None):
        sents_y = super(WordCharPosPreprocessor, self).transform(X, y)
        if len(sents_y) == 2:
            sents, y = sents_y
        else:
            sents = sents_y
        hands = self.pos_extractor.transform(X)
        sents_result = [sents[0], sents[1], hands, sents[2]]
        if y is None:
            return sents_result
        else:
            return sents_result, y
# WCPH
class WordCharPosHandPreprocessor(WordCharPosPreprocessor):
    def __init__(self,
                 lowercase=True,
                 num_norm=True,
                 char_feature=True,
                 vocab_init=None,
                 padding=True,
                 return_lengths=True,
                 hand_features = None):
        super(WordCharPosHandPreprocessor, self).__init__(lowercase, num_norm, char_feature, vocab_init, padding, return_lengths, hand_features)
        self.hand_extractor = HandcraftFeatureExtractor(hand_features=hand_features)
        self.hand_features = hand_features


    def fit(self, X, Y):
        super(WordCharPosHandPreprocessor, self).fit(X,Y)
        self.hand_extractor.fit(X,Y)
        return self

    def transform(self, X, Y=None):
        sents_y = super(WordCharPosHandPreprocessor, self).transform(X, Y)
        if len(sents_y) == 2:
            sents, y = sents_y
        else:
            sents = sents_y
        hands = self.hand_extractor.transform(X)
        sents_result = [sents[0], sents[1], sents[2], hands ,sents[-1]]
        if Y is None:
            return sents_result
        else:
            return sents_result, Y

class WordPosHandPreprocessor(WordCharPosHandPreprocessor):
    def __init__(self,
                 lowercase=True,
                 num_norm=True,
                 char_feature=True,
                 vocab_init=None,
                 padding=True,
                 return_lengths=True,
                 hand_features=None):
        super(WordPosHandPreprocessor, self).__init__(lowercase, num_norm, char_feature, vocab_init,
                                                          padding, return_lengths, hand_features)

    def transform(self, X, Y=None):
        sents_y = super(WordPosHandPreprocessor, self).transform(X, Y)
        if len(sents_y) == 2:
            sents, y = sents_y
        else:
            sents = sents_y
        sents_result = [sents[0], sents[2], sents[3], sents[-1]]
        if Y is None:
            return sents_result
        else:
            return sents_result, Y

# WP
class WordPosPreprocessor(WordCharPosPreprocessor):
    def __init__(self,
                 lowercase=True,
                 num_norm=True,
                 char_feature=True,
                 vocab_init=None,
                 padding=True,
                 return_lengths=True,
                 hand_features = None):
        super(WordPosPreprocessor, self).__init__(lowercase, num_norm, char_feature, vocab_init, padding, return_lengths, hand_features)

    def transform(self, X, y=None):
        sents_y = super(WordPosPreprocessor, self).transform(X, y)
        if len(sents_y) == 2:
            sents, y = sents_y
        else:
            sents = sents_y
        sents_result = [sents[0], sents[2], sents[-1]]
        if y is None:
            return sents_result
        else:
            return sents_result, y
# WCH
class WordCharHandPreprocessor(WordCharPreprocessor):
    def __init__(self,
                 lowercase=True,
                 num_norm=True,
                 char_feature=True,
                 vocab_init=None,
                 padding=True,
                 return_lengths=True,
                 hand_features = None):
        super(WordCharHandPreprocessor, self).__init__(lowercase, num_norm, char_feature, vocab_init, padding, return_lengths)
        self.hand_extractor = HandcraftFeatureExtractor(hand_features=hand_features)
        self.hand_features = hand_features

    def fit(self, X, y):
        super(WordCharHandPreprocessor, self).fit(X,y)
        self.hand_extractor.fit(X, y)
        return self

    def transform(self, X, y=None):
        sents_y = super(WordCharHandPreprocessor, self).transform(X,y)
        if len(sents_y) == 2:
            sents, y = sents_y
        else:
            sents = sents_y
        poses = self.hand_extractor.transform(X)
        return_sents = [sents[0], sents[1], poses, sents[-1]]
        if y is None:
            return return_sents
        else:
            return return_sents, y
# WH
class WordHandPreprocessor(WordCharHandPreprocessor):
    def __init__(self,
                 lowercase=True,
                 num_norm=True,
                 char_feature=True,
                 vocab_init=None,
                 padding=True,
                 return_lengths=True,
                 hand_features = None):
        super(WordHandPreprocessor, self).__init__(lowercase, num_norm, char_feature, vocab_init, padding, return_lengths)
        self.hand_extractor = HandcraftFeatureExtractor(hand_features=hand_features)
        self.hand_features = hand_features

    def transform(self, X, y=None):
        sents_y = super(WordHandPreprocessor, self).transform(X,y)
        if len(sents_y) == 2:
            sents, y = sents_y
        else:
            sents = sents_y
        return_sents = [sents[0], sents[2], sents[-1]]
        if y is None:
            return return_sents
        else:
            return return_sents, y


def prepare_preprocessor(X_train, Y_train, keras_model_name = "WC", hand_features = None):
    """
    :param X:
    :param y:
    :param keras_model_name:
    :param hand_features:
    :return:

    WCPH
    #TODO: W, C, P, H, WC, WP, WH, WCH, WCP, CPH, WCPH
    #DONE: WC, WCP, WCH
    """
    if keras_model_name == "W":
        p = WordCharPreprocessor(char_feature=False)
    elif keras_model_name == "C":
        p = CharPreprocessor()
    elif keras_model_name == "WC":
        p = WordCharPreprocessor()
    elif keras_model_name == "WP":
        p = WordPosPreprocessor(hand_features=hand_features)
    elif keras_model_name == "WCP":
        p = WordCharPosPreprocessor(hand_features=hand_features)
    elif keras_model_name == "WCH1" or keras_model_name == "WCH2" or keras_model_name == "WCH":
        p = WordCharHandPreprocessor(hand_features = hand_features)
    elif keras_model_name == "WH1" or keras_model_name == "WH2" or keras_model_name == "WH":
        p = WordHandPreprocessor(hand_features = hand_features)
    elif keras_model_name == "WPH": #NO
        if hand_features is None:
            hand_features = ['NEGAT','BING','BINGBIN','SWN','NAMEL','DEPENCY']
        p = WordPosHandPreprocessor(hand_features=hand_features)
    elif keras_model_name == "WCPH": #WCPH
        if hand_features is None:
            hand_features = ['NEGAT','BING','BINGBIN','SWN','NAMEL','DEPENCY']
        p = WordCharPosHandPreprocessor(hand_features=hand_features)
    else:
        p = WordCharPreprocessor()
    p.fit(X_train, Y_train)
    return p

if __name__ == "__main__":
    data_name = "laptops"
    task_name = "ATEPC2"
    DATA_ROOT = 'data'
    SAVE_ROOT = './models'  # trained models
    LOG_ROOT = './logs'  # checkpoint, tensorboard
    embedding_path = '/home/s1610434/Documents/Data/Vector/glove.twitter.27B.100d.txt'

    keras_model_name = "WCH"
    hand_features = None

    hand_features_dict = {"POS": 0, "UNIPOS": 0 , "NEGAT": 0, "BING": 0, "BINGBIN": 0 , "SWN": 0, "NAMEL": 0, "DEPENCY": 0}

    print("-----{0}-----{1}-----{2}-----{3}-----".format(task_name, data_name, keras_model_name, hand_features))
    save_path = SAVE_ROOT + "/{0}/{1}".format(data_name, task_name)
    train_path = os.path.join(DATA_ROOT, '{0}.{1}.train.tsv'.format(data_name, task_name))
    test_path = os.path.join(DATA_ROOT, '{0}.{1}.test.tsv'.format(data_name, task_name))
    train_dep_path = os.path.join(DATA_ROOT, '{0}.{1}.train.dep.tsv'.format(data_name, task_name))
    test_dep_path = os.path.join(DATA_ROOT, '{0}.{1}.test.dep.tsv'.format(data_name, task_name))

    # train set
    x_train_valid, y_train_valid, _ = collect_data_from_tsv(train_path)
    x_train_valid_dep = collect_dept_data_from_tsv(train_dep_path)

    # test set
    X_test, Y_test, _ = collect_data_from_tsv(test_path)
    X_test_dep = collect_dept_data_from_tsv(test_dep_path)

    # TODO Kfold split
    kf = KFold(n_splits=10)
    i_fold = 0

    pos_fe = POSExtractor()
    print(len(pos_fe.features_dict))

    count_train = 0
    count_valid = 0
    count_test = 0
    for train_index, valid_index in kf.split(x_train_valid):
        model_name = "{0}.{1}.{2}".format(keras_model_name, hand_features, i_fold)
        X_train, X_valid = x_train_valid[train_index], x_train_valid[valid_index]
        X_train_dep, X_valid_dep = x_train_valid_dep[train_index], x_train_valid_dep[valid_index]
        Y_train, Y_valid = y_train_valid[train_index], y_train_valid[valid_index]


        print(X_train[0])
        print(X_train[-1])

        print(X_valid[0])
        print(X_valid[-1])

        print(X_test[0])
        print(X_test[-1])
        print("-----")

        # print("Data train: ", X_train.shape, Y_train.shape)
        # print("Data valid: ", X_valid.shape, Y_valid.shape)
        # print("Data  test: ", X_test.shape, Y_test.shape)

        # p = prepare_preprocessor(list(zip(X_train, X_train_dep)), Y_train, keras_model_name=keras_model_name,hand_features=hand_features)
        #
        # data_t, label_v = p.transform(list(zip(X_train, X_train_dep)), Y_train)
        # data, label = p.transform(list(zip(X_valid, X_valid_dep)) ,Y_valid)

        # print(data_t[2].shape)
        # print(data[2].shape)
        # print(count_train)
        # print(count_valid)
        # print(count_test)






    # hc_fe = prepare_preprocessor(x_train_valid, y_train_valid, keras_model_name="WH1", hand_features=["UNIPOS","BINGBIN", "NAMEL"])
    # bbb, _ =  hc_fe.transform(x_train_valid, y_train_valid)
    # print(len(bbb))
    # aaa, _ = hc_fe.transform(x_test, y_test)
    # print(len(aaa))
    #
    # for b in bbb: print(b.shape)
    # for a in aaa: print(a.shape)



