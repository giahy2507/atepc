import os

from anago.reader import batch_iter, batch_iter2
from keras.optimizers import Adam

import anago.models
from anago.metrics import get_mycallbacks, get_callbacks


class Trainer(object):
    def __init__(self,
                 model_config,
                 training_config,
                 checkpoint_path='',
                 save_path='',
                 tensorboard=True,
                 preprocessor=None,
                 W_embeddings=None,
                 C_embeddings=None,
                 POS_embeddings=None,
                 model_name='defaut', keras_model_name ="WC"):
        self.model_config = model_config
        self.training_config = training_config
        self.checkpoint_path = checkpoint_path
        self.save_path = save_path
        self.tensorboard = tensorboard
        self.preprocessor = preprocessor
        self.W_embeddings = W_embeddings
        self.C_embeddings = C_embeddings
        self.POS_embeddings = POS_embeddings
        self.model_name = model_name
        self.keras_model_name = keras_model_name + "SeqLabeling"

    def train(self, X_train, Y_train, X_valid=None, Y_valid=None):

        # Prepare training and validation data(steps, generator)
        train_steps, train_batches = batch_iter2(list(zip(X_train, Y_train)), self.training_config.batch_size, preprocessor=self.preprocessor)
        valid_steps, valid_batches = batch_iter2(list(zip(X_valid, Y_valid)), self.training_config.batch_size, preprocessor=self.preprocessor)

        # Build the model
        class_ = getattr(anago.models, self.keras_model_name)

        model = class_(config = self.model_config,
                       w_embeddings=self.W_embeddings,
                       c_embeddings = self.C_embeddings,
                       pos_embeddings=self.POS_embeddings,
                       ntags =len(self.preprocessor.vocab_tag))

        model.compile(loss=model.crf.loss,
                      optimizer=Adam(lr=self.training_config.learning_rate))

        # Prepare callbacks for training
        callbacks = get_callbacks(save_path=os.path.join(self.save_path,self.model_name),
                                  eary_stopping=self.training_config.early_stopping,
                                  valid=(valid_steps, valid_batches , self.preprocessor),
                                  patience=self.training_config.patience)

        # Train the model
        model.fit_generator(generator=train_batches,
                            steps_per_epoch=train_steps,
                            epochs=self.training_config.max_epoch,
                            callbacks=callbacks)

        # # Saving the model is included in callback function
        # model.save(os.path.join(self.save_path,self.model_name))


class Trainer2(object):
    def __init__(self,
                 model_config,
                 training_config,
                 checkpoint_path='',
                 save_path='',
                 tensorboard=True,
                 preprocessor=None,
                 W_embeddings=None,
                 C_embeddings=None,
                 POS_embeddings=None,
                 model_name='defaut', keras_model_name ="WC"):
        self.model_config = model_config
        self.training_config = training_config
        self.checkpoint_path = checkpoint_path
        self.save_path = save_path
        self.tensorboard = tensorboard
        self.preprocessor = preprocessor
        self.W_embeddings = W_embeddings
        self.C_embeddings = C_embeddings
        self.POS_embeddings = POS_embeddings
        self.model_name = model_name
        self.keras_model_name = keras_model_name + "SeqLabeling"

    def train(self, X_train, Y_train, X_valid=None, Y_valid=None):

        # Prepare training and validation data(steps, generator)
        train_steps, train_batches = batch_iter(list(zip(X_train, Y_train)), self.training_config.batch_size, preprocessor=self.preprocessor)
        X_valid, Y_valid = self.preprocessor.transform(X_valid, Y_valid)

        # Build the model
        class_ = getattr(anago.models, self.keras_model_name)

        model = class_(config = self.model_config,
                       w_embeddings=self.W_embeddings,
                       c_embeddings = self.C_embeddings,
                       pos_embeddings=self.POS_embeddings,
                       ntags =len(self.preprocessor.vocab_tag))

        model.compile(loss=model.crf.loss,
                      optimizer=Adam(lr=self.training_config.learning_rate))

        # Prepare callbacks for training
        callbacks = get_mycallbacks(save_path=os.path.join(self.save_path,self.model_name),
                                  eary_stopping=self.training_config.early_stopping,
                                  valid=(X_valid, Y_valid , self.preprocessor),
                                  patience=self.training_config.patience)

        # Train the model
        model.fit_generator(generator=train_batches,
                            steps_per_epoch=train_steps,
                            epochs=self.training_config.max_epoch,
                            callbacks=callbacks)