import os

import numpy as np

import anago
from anago.config import TrainingConfig, prepare_modelconfig
from anago.reader import load_data_and_labels
from evaluation import ATEPCEvaluator
from features import prepare_preprocessor
from utils import collect_dept_data_from_tsv, collect_data_from_tsv
from sklearn.model_selection import KFold

from trainanagoWCH import gen_no_hand_dimension

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


def test_anago(keras_model_name= "WCP", hand_features = None, task_name="ATEPC2", data_name="laptops"):
    DATA_ROOT = 'data'
    SAVE_ROOT = './models'  # trained models
    LOG_ROOT = './logs'  # checkpoint, tensorboard
    w_embedding_path = '/home/s1610434/Documents/Data/Vector/glove.twitter.27B.100d.txt'
    c_embedding_path = '/home/s1610434/Documents/Data/Vector/AmaYelp/GloVe/glove.char.100.txt'
    pos_embedding_path = '/home/s1610434/Documents/Data/Vector/AmaYelp/GloVe/glove.pos.100.txt'
    unipos_embedding_path = '/home/s1610434/Documents/Data/Vector/AmaYelp/GloVe/glove.unipos.100.txt'

    model_config = prepare_modelconfig(keras_model_name)
    training_config = TrainingConfig()
    training_config.max_epoch = 100
    training_config.early_stopping = 20

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
    results = []
    atepc_evaluator = ATEPCEvaluator()
    for train_index, valid_index in kf.split(x_train_valid):
        model_name = "{0}.{1}.{2}".format(keras_model_name, "{0}".format(hand_features), i_fold)
        X_train, X_valid = x_train_valid[train_index], x_train_valid[valid_index]
        X_train_dep, X_valid_dep = x_train_valid_dep[train_index], x_train_valid_dep[valid_index]
        Y_train, Y_valid = y_train_valid[train_index], y_train_valid[valid_index]

        print("Data train: ", X_train.shape, Y_train.shape)
        print("Data valid: ", X_valid.shape, Y_valid.shape)
        print("Data  test: ", X_test.shape, Y_test.shape)

        p = prepare_preprocessor(list(zip(X_train, X_train_dep)), Y_train, keras_model_name=keras_model_name,
                                 hand_features=hand_features)
        model_config.vocab_size = len(p.vocab_word)
        model_config.char_vocab_size = len(p.vocab_char)
        if keras_model_name.find("P") != -1:
            if hand_features is not None:
                if "UNIPOS" in hand_features:
                    pos_embedding_path = unipos_embedding_path
            model_config.pos_vocab_size = len(p.pos_extractor.features_dict)
        if keras_model_name.find("H") != -1:
            # model_config.hand_feature_size = gen_no_hand_dimension(data_name, hand_features, keras_model_name)
            model_config.hand_feature_size = 53
            print("model_config.hand_feature_size: ", str(model_config.hand_feature_size))

        filepath = os.path.join(save_path, model_name)
        if os.path.isfile(filepath) is False:
            continue

        evaluator = anago.Evaluator(model_config, weights=model_name, save_path=save_path, preprocessor=p,
                                    keras_model_name=keras_model_name)
        print("--- Test phrase --- " + model_name)
        # print("Train ")
        # f1_score_train = evaluator.eval(list(zip(X_train, X_train_dep)), Y_train)
        # print("Validation ")
        # f1_score_valid = evaluator.eval(list(zip(X_valid, X_valid_dep)), Y_valid)
        # print("Test ")
        f1_score_test = evaluator.eval(list(zip(X_test, X_test_dep)), Y_test)
        print("---")
        i_fold += 1

        # Kfold cross validation
        f_out_name = "data/{0}.{1}.test.pred.tsv".format(data_name, task_name)
        f_out = open(f_out_name, "w")
        ## Tagging
        tagger = anago.Tagger(model_config, model_name, save_path=save_path, preprocessor=p,
                              keras_model_name=keras_model_name)
        for x, y in zip(list(zip(X_test, X_test_dep)), Y_test):
            result = tagger.predict(x)
            for word, label, pred in zip(x[0], y, result):
                f_out.write("{0}\t{1}\t{2}\n".format(word, label, pred))
            f_out.write("\n")
        f_out.close()
        ate_f1, apc_acc, c_apc_acc = atepc_evaluator.evaluate(f_out_name)
        results.append([ate_f1, apc_acc, c_apc_acc])
        print(results[-1])

    print("-----All-----{0}--{1}".format(keras_model_name, data_name))
    for result in results:
        print(result)
    print("-----AVG-----")
    results_np = np.array(results, dtype=np.float32)
    print(results_np.mean(axis=0))
    print("-------------")

if __name__ == "__main__":
    # test_anago(keras_model_name="W", hand_features=None, task_name="ATEPC2", data_name="laptops")
    # test_anago(keras_model_name="WP", hand_features=None, task_name="ATEPC2", data_name="laptops")
    test_anago(keras_model_name="WC", hand_features=None, task_name="ATEPC2", data_name="laptops")
    # test_anago(keras_model_name="WP", hand_features=None, task_name="ATEPC2", data_name="restaurants")