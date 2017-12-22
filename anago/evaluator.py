import os

from anago.reader import batch_iter

import anago.models
from anago.metrics import F1score, MyF1score


class Evaluator(object):

    def __init__(self,
                 config,
                 weights,
                 save_path='',
                 preprocessor=None, keras_model_name = "WCP"):

        self.config = config
        self.weights = weights
        self.save_path = save_path
        self.preprocessor = preprocessor
        self.keras_model_name = keras_model_name + "SeqLabeling"

    def eval(self, X_test , Y_test):

        # Prepare test data(steps, generator)
        X_train, Y_train = self.preprocessor.transform(X_test, Y_test)

        # Build the model
        class_ = getattr(anago.models, self.keras_model_name)
        model = class_(config = self.config,
                       ntags =len(self.preprocessor.vocab_tag))
        model.load_weights(filepath=os.path.join(self.save_path, self.weights))

        # Build the evaluator and evaluate the model
        f1score = MyF1score(X_train, Y_train, self.preprocessor)
        f1score.model = model
        f1 = f1score.on_epoch_end(epoch=-1)  # epoch takes any integer.
        return f1
