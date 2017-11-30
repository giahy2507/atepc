import os
import anago
from anago.data.reader import load_data_and_labels, load_word_embeddings
from anago.config import TrainingConfig, prepare_modelconfig
from features import prepare_preprocessor
import numpy as np
from sklearn.model_selection import KFold

def train_anago(keras_model_name= "WCP"):
    DATA_ROOT = 'data'
    SAVE_ROOT = './models'  # trained models
    LOG_ROOT = './logs'  # checkpoint, tensorboard
    embedding_path = "C:\\hynguyen\\Data\\vector\\glove.twitter.27B.100d.txt"

    model_config = prepare_modelconfig(keras_model_name)
    training_config = TrainingConfig()

    list_task_name = ["ATEPC2"]
    list_data_name = ["laptops", "restaurants"]

    for task_name in list_task_name:
        for data_name in list_data_name:
            save_path = SAVE_ROOT + "/{0}/{1}".format(data_name, task_name)
            print("----------{0}-----{1}----------".format(task_name, data_name))
            train_path = os.path.join(DATA_ROOT, '{0}.{1}.train.tsv'.format(data_name, task_name))
            test_path = os.path.join(DATA_ROOT, '{0}.{1}.test.tsv'.format(data_name, task_name))
            x_train_valid, y_train_valid = load_data_and_labels(train_path)
            x_test, y_test = load_data_and_labels(test_path)
            x_train_test = np.concatenate((x_train_valid, x_test), axis=0)
            y_train_test = np.concatenate((y_train_valid, y_test), axis=0)

            p = prepare_preprocessor(x_train_test, y_train_test, keras_model_name=keras_model_name)
            embeddings = load_word_embeddings(p.vocab_word, embedding_path, model_config.word_embedding_size)
            model_config.vocab_size = len(p.vocab_word)
            model_config.char_vocab_size = len(p.vocab_char)
            if keras_model_name.find("P") != -1:
                model_config.pos_size = len(p.pos_extractor.features_dict)

            # TODO Kfold split
            kf = KFold(n_splits=10)
            i_fold = 0
            for train_index, valid_index in kf.split(x_train_valid):

                model_name = "defaut.{0}".format(i_fold)
                X_train, X_valid = x_train_valid[train_index], x_train_valid[valid_index]
                y_train, y_valid = y_train_valid[train_index], y_train_valid[valid_index]

                print(len(X_train), len(X_valid), len(y_train), len(y_valid))
                trainer = anago.Trainer(model_config, training_config, checkpoint_path=LOG_ROOT, save_path=save_path,
                                        preprocessor=p, embeddings=embeddings, keras_model_name = keras_model_name, model_name=model_name)
                trainer.train(X_train, y_train, X_valid, y_valid)
                print("")
                i_fold+=1


if __name__ == "__main__":
    train_anago("WCP")