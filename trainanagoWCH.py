import argparse
import os

from sklearn.model_selection import KFold

import anago
from anago.config import TrainingConfig, prepare_modelconfig
from anago.reader import load_word_embeddings
from features import prepare_preprocessor
from utils import collect_dept_data_from_tsv, collect_data_from_tsv
from anago.trainer import Trainer2, Trainer
from evaluation import ATEPCEvaluator
import numpy as np

def gen_no_hand_dimension(data_name = "laptops", hand_features = None, keras_model_name="WPH"):
    result = 0
    if data_name == "laptops":
        features_dict = {'POS':47,
                         'UNIPOS':14,
                         'NEGAT':1,
                         'BING':1,
                         'BINGBIN':1,
                         'SWN':1,
                         'NAMEL':2,
                         'DEPENCY':2}

    else:
        features_dict = {'POS': 47,
                         'UNIPOS': 14,
                         'NEGAT': 1,
                         'BING': 1,
                         'BINGBIN': 1,
                         'SWN': 1,
                         'NAMEL': 2,
                         'DEPENCY': 2}

    if hand_features is None:
        if keras_model_name=="WPH" or keras_model_name=="WCPH":
            hand_features = ['NEGAT', 'BING', 'BINGBIN', 'SWN', 'NAMEL', 'DEPENCY']
        else:
            hand_features = ['POS', 'NEGAT', 'BING', 'BINGBIN', 'SWN', 'NAMEL', 'DEPENCY']

    for feature in hand_features:
        result += features_dict[feature]
    return result

def train_anago(keras_model_name= "WCP", data_name="laptops", task_name="ATEPC2", hand_features = None):
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
    training_config.early_stopping = 30

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

    # train_test set
    X_train_test = np.concatenate((x_train_valid, X_test), 0)
    X_train_test_dep = np.concatenate((x_train_valid_dep, X_test_dep), 0)
    Y_train_test = np.concatenate((y_train_valid, Y_test), 0)

    # preprocessor
    p = prepare_preprocessor(list(zip(X_train_test, X_train_test_dep)), Y_train_test, keras_model_name=keras_model_name,
                             hand_features=hand_features)

    print(len(p.vocab_word))
    print(len(p.vocab_char))
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

    # load embedding
    W_embeddings = load_word_embeddings(p.vocab_word, w_embedding_path, model_config.word_embedding_size)
    print("Load W_embeddings: {0}".format(W_embeddings.shape))
    C_embeddings = None
    POS_embeddings = None
    # if "C" in keras_model_name:
    #     C_embeddings = load_word_embeddings(p.vocab_char, c_embedding_path, model_config.char_embedding_size)
    #     print("Load C_embeddings: {0}".format(C_embeddings.shape))
    # if "P" in keras_model_name:
    #     POS_embeddings = load_word_embeddings(p.pos_extractor.features_dict, pos_embedding_path, model_config.pos_embedding_size)
    #     print("Load POS_embeddings: {0}".format(POS_embeddings.shape))

    atepc_evaluator = ATEPCEvaluator()
    results = []

    # TODO Kfold split
    kf = KFold(n_splits=10)
    i_fold = 0
    for train_index, valid_index in kf.split(x_train_valid):
        model_name = "{0}.{1}.{2}".format(keras_model_name,  "{0}".format(hand_features), i_fold)
        X_train, X_valid = x_train_valid[train_index], x_train_valid[valid_index]
        X_train_dep, X_valid_dep = x_train_valid_dep[train_index], x_train_valid_dep[valid_index]
        Y_train, Y_valid = y_train_valid[train_index], y_train_valid[valid_index]

        print("Data train: ", X_train.shape, Y_train.shape)
        print("Data valid: ", X_valid.shape, Y_valid.shape)
        print("Data  test: ", X_test.shape, Y_test.shape)

        trainer = Trainer(model_config=model_config,
                                training_config=training_config,
                                checkpoint_path=LOG_ROOT,
                                save_path=save_path,
                                preprocessor=p,
                                W_embeddings=W_embeddings,
                                C_embeddings=C_embeddings,
                                POS_embeddings=POS_embeddings,
                                keras_model_name = keras_model_name,
                                model_name=model_name)

        # trainer = Trainer2(model_config=model_config,
        #                         training_config=training_config,
        #                         checkpoint_path=LOG_ROOT,
        #                         save_path=save_path,
        #                         preprocessor=p,
        #                         W_embeddings=W_embeddings,
        #                         C_embeddings=C_embeddings,
        #                         POS_embeddings=POS_embeddings,
        #                         keras_model_name = keras_model_name,
        #                         model_name=model_name)

        trainer.train(list(zip(X_train, X_train_dep)), Y_train, list(zip(X_valid, X_valid_dep)), Y_valid)

        evaluator = anago.Evaluator(model_config, weights=model_name, save_path=save_path, preprocessor=p,keras_model_name=keras_model_name)
        print("--- Test phrase --- " + model_name)
        print("Train ")
        f1_score_train = evaluator.eval(list(zip(X_train, X_train_dep)), Y_train)
        print("Validation ")
        f1_score_valid = evaluator.eval(list(zip(X_valid, X_valid_dep)), Y_valid)
        print("Test ")
        f1_score_test = evaluator.eval(list(zip(X_test, X_test_dep)), Y_test)
        print("---")
        i_fold+=1

        f_out_name = "data/{0}.{1}.test.pred.tsv".format(data_name, task_name)
        f_out = open(f_out_name, "w")
        tagger = anago.Tagger(model_config, model_name, save_path=save_path, preprocessor=p, keras_model_name=keras_model_name)
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
    parser = argparse.ArgumentParser()
    parser.add_argument('-kr_name', type=str, default="WCH1", help='keras_model_name')
    parser.add_argument('-data_name', type=str, default="laptops", help='data_name')
    parser.add_argument('-hand_features', type=str, default=None, help='hand_features')
    args = parser.parse_args()

    kr_names = ["WCH1", "WCH2", "WH1", "WH2", "WH", "WCH", "WC", "W", "WCP", "WP", "WPH", "WCPH"]
    data_names = ["laptops", "restaurants"]

    if args.hand_features is None:
        hand_features = None
    else:
        hand_features = args.hand_features.split(",")

    if args.kr_name in kr_names and args.data_name in data_names:
        train_anago(keras_model_name=args.kr_name, data_name=args.data_name, hand_features=hand_features)
    else:
        print("Wrong parameter, please choose params from these lists: ")
        print("-kr_name", kr_names)
        print("-data_name",data_names)