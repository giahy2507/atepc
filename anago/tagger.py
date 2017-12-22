import os
from collections import defaultdict

import numpy as np

import anago.models
from anago.metrics import get_entities


class Tagger(object):

    def __init__(self,
                 config,
                 weights,
                 save_path='',
                 preprocessor=None,
                 tokenizer=str.split,
                 keras_model_name="WCP"):

        self.preprocessor = preprocessor
        self.tokenizer = tokenizer
        self.keras_model_name = keras_model_name + "SeqLabeling"

        class_ = getattr(anago.models, self.keras_model_name)
        # Build the model
        self.model = class_(config, ntags=len(self.preprocessor.vocab_tag))
        self.model.load(filepath=os.path.join(save_path, weights))

    def predict(self, words_depency):
        sequence_lengths = [len(words_depency)]
        X = self.preprocessor.transform([words_depency], y = None)
        pred = self.model.predict(X, sequence_lengths)
        pred = np.argmax(pred, -1)
        pred = self.preprocessor.inverse_transform(pred[0])
        return pred

    def tag(self, sent):
        """Tags a sentence named entities.

        Args:
            sent: a sentence

        Return:
            labels_pred: list of (word, tag) for a sentence

        Example:
            >>> sent = 'President Obama is speaking at the White House.'
            >>> print(self.tag(sent))
            [('President', 'O'), ('Obama', 'PERSON'), ('is', 'O'),
             ('speaking', 'O'), ('at', 'O'), ('the', 'O'),
             ('White', 'LOCATION'), ('House', 'LOCATION'), ('.', 'O')]
        """
        assert isinstance(sent, str)

        words = self.tokenizer(sent)
        pred = self.predict(words)
        pred = [t.split('-')[-1] for t in pred]  # remove prefix: e.g. B-Person -> Person

        return list(zip(words, pred))

    def get_entities(self, sent):
        """Gets entities from a sentence.

        Args:
            sent: a sentence

        Return:
            labels_pred: dict of entities for a sentence

        Example:
            sent = 'President Obama is speaking at the White House.'
            result = {'Person': ['Obama'], 'LOCATION': ['White House']}
        """
        assert isinstance(sent, str)

        words = self.tokenizer(sent)
        pred = self.predict(words)
        entities = self._get_chunks(words, pred)

        return entities

    def _get_chunks(self, words, tags):
        """
        Args:
            words: sequence of word
            tags: sequence of labels

        Returns:
            dict of entities for a sequence

        Example:
            words = ['President', 'Obama', 'is', 'speaking', 'at', 'the', 'White', 'House', '.']
            tags = ['O', 'B-Person', 'O', 'O', 'O', 'O', 'B-Location', 'I-Location', 'O']
            result = {'Person': ['Obama'], 'LOCATION': ['White House']}
        """
        chunks = get_entities(tags)
        res = defaultdict(list)
        for chunk_type, chunk_start, chunk_end in chunks:
            res[chunk_type].append(' '.join(words[chunk_start: chunk_end]))  # todo delimiter changeable

        return res
