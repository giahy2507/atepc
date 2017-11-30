import os

from keras.optimizers import Adam

from anago.data.metrics import get_callbacks, F1score, get_mycallbacks
from anago.data.reader import batch_iter
import anago.models
from keras import metrics

class Trainer(object):
    def __init__(self,
                 model_config,
                 training_config,
                 checkpoint_path='',
                 save_path='',
                 tensorboard=True,
                 preprocessor=None,
                 embeddings=None,
                 model_name='defaut', keras_model_name ="WC"):
        self.model_config = model_config
        self.training_config = training_config
        self.checkpoint_path = checkpoint_path
        self.save_path = save_path
        self.tensorboard = tensorboard
        self.preprocessor = preprocessor
        self.embeddings = embeddings
        self.model_name = model_name
        self.keras_model_name = keras_model_name + "SeqLabeling"

    def train(self, x_train, y_train, x_valid=None, y_valid=None):

        # Prepare training and validation data(steps, generator)
        train_steps, train_batches = batch_iter(
            list(zip(x_train, y_train)), self.training_config.batch_size, preprocessor=self.preprocessor)
        valid_steps, valid_batches = batch_iter(
            list(zip(x_valid, y_valid)), self.training_config.batch_size, preprocessor=self.preprocessor)

        # Build the model
        class_ = getattr(anago.models, self.keras_model_name)
        model = class_(self.model_config, self.embeddings, len(self.preprocessor.vocab_tag))
        model.compile(loss=model.crf.loss,
                      optimizer=Adam(lr=self.training_config.learning_rate),
                      )

        # Prepare callbacks for training
        callbacks = get_mycallbacks(save_path=os.path.join(self.save_path,self.model_name),
                                  eary_stopping=self.training_config.early_stopping,
                                  valid=(valid_steps, valid_batches, self.preprocessor),
                                  patience=self.training_config.patience)

        # Train the model
        model.fit_generator(generator=train_batches,
                            steps_per_epoch=train_steps,
                            epochs=self.training_config.max_epoch,
                            callbacks=callbacks)

        # # Saving the model is included in callback function
        # model.save(os.path.join(self.save_path,self.model_name))


