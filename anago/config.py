class WModelConfig(object):
    def __init__(self):
        """Sets the default model hyperparameters."""

        # Number of unique words in the vocab (plus 2, for <UNK>, <PAD>).
        self.vocab_size = None
        # Batch size.
        self.batch_size = 32
        # Scale used to initialize model variables.
        self.initializer_scale = 0.08

        # LSTM input and output dimensionality, respectively.
        self.word_embedding_size = 100
        self.num_word_lstm_units = 100

        # If < 1.0, the dropout keep probability applied to LSTM variables.
        self.dropout = 0.5

        # If True, use crf.
        self.crf = True

class WCModelConfig(WModelConfig):
    """Wrapper class for model hyperparameters."""

    def __init__(self):
        super(WCModelConfig, self).__init__()
        """Sets the default model hyperparameters."""

        # Number of unique words in the vocab (plus 2, for <UNK>, <PAD>).
        self.char_vocab_size = None

        # LSTM input and output dimensionality, respectively.
        self.char_embedding_size = 100
        self.num_char_lstm_units = 100

        # If True, use character feature.
        self.char_feature = True


class WCPModelConfig(WCModelConfig):
    def __init__(self):
        super(WCPModelConfig, self).__init__()
        self.pos_feature = True
        self.pos_vocab_size = None
        self.pos_embedding_size = 100

class WCHModelConfig(WCModelConfig):
    def __init__(self):
        super(WCHModelConfig, self).__init__()
        self.hand_feature_size = None

class WCPHModelConfig(WCPModelConfig):
    def __init__(self):
        super(WCPHModelConfig, self).__init__()
        self.hand_feature_size = None

def prepare_modelconfig(keras_model_name = "WC"):
    if keras_model_name == "W":
        return WModelConfig()
    if keras_model_name == "C":
        return WCModelConfig()
    elif keras_model_name == "WC":
        return WCModelConfig()
    elif keras_model_name == "WCP" or keras_model_name == "WP":
        return WCPModelConfig()
    elif keras_model_name == "WCH1" or keras_model_name == "WCH2" or keras_model_name == "WCH":
        return WCHModelConfig()
    elif keras_model_name == "WCPH" or keras_model_name == "WPH":
        return WCPHModelConfig()
    else:
        return WCModelConfig()


class TrainingConfig(object):
    """Wrapper class for training hyperparameters."""

    def __init__(self):
        """Sets the default training hyperparameters."""

        # Batch size
        self.batch_size = 20

        # Optimizer for training the model.
        self.optimizer = 'adam'

        # Learning rate for the initial phase of training.
        self.learning_rate = 0.001
        self.lr_decay = 0.9

        # If not None, clip gradients to this value.
        self.clip_gradients = 5.0

        # The number of max epoch size
        self.max_epoch = 50

        # Parameters for early stopping
        self.early_stopping = True
        self.patience =20

        # Fine-tune word embeddings
        self.train_embeddings = True

        # How many model checkpoints to keep.
        self.max_checkpoints_to_keep = 5
