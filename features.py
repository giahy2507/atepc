import nltk
from evaluation import ResultConverter
from collections import OrderedDict, defaultdict
from utils import *
import pickle
import numpy as np
from anago.data.preprocess import pad_sequences
from anago.data.preprocess import WordCharPreprocessor


UNK = '<UNK>'
PAD = '<PAD>'

class FeatureExtractor(object):

    def __init__(self, name="Feature", features_dict = None):
        self.name = name
        self.features_dict = OrderedDict() if features_dict is None else features_dict

    def fit(self, X):
        return

    def transform(self, X, one_host = False):
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
        super(POSExtractor, self).save(self.savepath, self.features_dict)
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

    @classmethod
    def load(self, fname=""):
        result = POSExtractor()
        result.features_dict = super(POSExtractor, self).load(result.savepath)
        return result

    def save(self, fname, data):
        super(POSExtractor, self).save(self.savepath, self.features_dict)

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
            result.append(self.words_2_onehost(words))
        return np.array(result, dtype=np.int32)

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


class SentimentScoreExtractor(FeatureExtractor):
    def __init__(self, name="POS Feature", features_dict = None):
        super(SentimentScoreExtractor, self).__init__(name, features_dict)
        self.savepath = os.path.join(os.getcwd(), "models/features_sentimentscore.pickle")




class WordCharPosPreprocessor(WordCharPreprocessor):
    def __init__(self,
                 lowercase=True,
                 num_norm=True,
                 char_feature=True,
                 vocab_init=None,
                 padding=True,
                 return_lengths=True,
                 pos_feature = "POS"):
        super(WordCharPosPreprocessor, self).__init__(lowercase, num_norm, char_feature, vocab_init, padding, return_lengths)
        if pos_feature == "UniversalPOS":
            self.pos_extractor = UniversalPOSExtractor()
        else:
            self.pos_extractor = POSExtractor()

    def fit(self, X, y):
        super(WordCharPosPreprocessor, self).fit(X,y)
        self.pos_extractor.fit(X)
        return self

    def transform(self, X, y=None):
        sents_y = super(WordCharPosPreprocessor, self).transform(X,y)
        if len(sents_y) == 2:
            sents, y = sents_y
        else:
            sents = sents_y
        poses = self.pos_extractor.transform(X)
        sents.insert(-2, poses)
        return sents, y if y is not None else sents

def prepare_preprocessor(X, y, keras_model_name = "WC"):
    if keras_model_name == "W":
        p = WordCharPreprocessor()
    elif keras_model_name == "WC":
        p = WordCharPreprocessor()
    elif keras_model_name == "WCP":
        p = WordCharPosPreprocessor()
    else:
        p = WordCharPreprocessor()
    p.fit(X, y)
    return p

if __name__ == "__main__":
    sents1, labels1, pred_labels = collect_data_from_tsv("data/restaurants.ATEPC2.test.tsv")
    sents2, labels2, pred_labels2 = collect_data_from_tsv("data/restaurants.ATEPC2.train.tsv")
    sents3, labels3, pred_labels2 = collect_data_from_tsv("data/laptops.ATEPC2.test.tsv")
    sents4, labels4, pred_labels2 = collect_data_from_tsv("data/laptops.ATEPC2.train.tsv")

    pos_p = WordCharPosPreprocessor(pos_feature="UniversalPOS")
    sents = sents1 + sents2 + sents3 +sents4
    labels = labels1 + labels2 + labels3 + labels4
    pos_p.fit(sents, labels)

    print ("ahihi")

    a, b = pos_p.transform(sents[1:5], labels[1:5])
    print(len(a))


