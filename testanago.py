import os
import anago
from anago.data.reader import load_data_and_labels, load_word_embeddings
from anago.data.preprocess import prepare_preprocessor
from anago.config import ModelConfig, TrainingConfig
from sklearn.model_selection import KFold
import numpy as np


def get_aspecterm(x,y):
    i = 0
    result = []
    y.append("O")
    while i < len(y):
        if y[i] == "B":
            aspecterm = []
            aspecterm.append(x[i])
            i+=1
            while y[i] == "I" and i < len(y):
                aspecterm.append(x[i])
                i+=1
            result.append(aspecterm)
        else:
            i+=1
    return result


if __name__ == "__main__":
    DATA_ROOT = './data'
    SAVE_ROOT = './models'  # trained models
    LOG_ROOT = './logs'  # checkpoint, tensorboard
    embedding_path = '/home/s1610434/Documents/Data/Vector/glove.twitter.27B.100d.txt'
    model_config = ModelConfig()
    training_config = TrainingConfig()
    training_config.max_epoch = 100

    list_task_name = ["ATEPC", "ATEPC2"]
    list_data_name = ["laptops", "restaurants"]

    for task_name in list_task_name:
        for data_name in list_data_name:
            save_path = SAVE_ROOT + "/{0}/{1}".format(data_name, task_name)
            print("----------{0}-----{1}----------".format(task_name, data_name))
            train_path = os.path.join(DATA_ROOT, '{0}.{1}.train.tsv'.format(data_name, task_name))
            valid_path = os.path.join(DATA_ROOT, '{0}.{1}.test.tsv'.format(data_name, task_name))
            test_path = os.path.join(DATA_ROOT, '{0}.{1}.test.tsv'.format(data_name, task_name))
            x_train_valid, y_train_valid = load_data_and_labels(train_path)
            x_test, y_test = load_data_and_labels(test_path)
            x_train_test =  np.concatenate((x_train_valid,x_test),axis=0)
            y_train_test = np.concatenate((y_train_valid,y_test),axis=0)
            p = prepare_preprocessor(x_train_test, y_train_test)
            embeddings = load_word_embeddings(p.vocab_word, embedding_path, model_config.word_embedding_size)
            model_config.vocab_size = len(p.vocab_word)
            model_config.char_vocab_size = len(p.vocab_char)
            kf = KFold(n_splits=10)
            k_fold_data = []

            best_f1 = 0
            best_model_name = ""
            list_model_name =  [f_name for f_name in os.listdir(save_path) if f_name.find("model_weights")!=-1]
            for model_name in list_model_name:
                i_fold = model_name[-1:]
                evaluator = anago.Evaluator(model_config, model_name, save_path=save_path, preprocessor=p)
                f1_score_train = evaluator.eval(x_train_valid, y_train_valid)
                f1_score_test = evaluator.eval(x_test, y_test)
                print(model_name)
                print(' - f1_train: {:04.2f}'.format(f1_score_train * 100))
                print(' - f1_test: {:04.2f}\n'.format(f1_score_test * 100))
                if f1_score_test > best_f1:
                    best_f1 = f1_score_test
                    best_model_name = model_name

            f_out_name = "data/{0}.{1}.test.pred.tsv".format(data_name, task_name)
            f_out = open(f_out_name, "w")
            ## Tagging
            tagger = anago.Tagger(model_config, best_model_name, save_path=save_path, preprocessor=p)
            for x,y in zip(x_test, y_test):
                result = tagger.predict(x)
                for word, label, pred in zip(x,y,result):
                    f_out.write("{0}\t{1}\t{2}\n".format(word, label, pred))
                f_out.write("\n")
            f_out.close()
