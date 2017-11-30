from collections import defaultdict
import os
import xml.etree.ElementTree as ET
import re
import nltk
import copy
from shutil import copyfile
import subprocess
import numpy as np
from utils import *

class ResultConverter(object):
    def __init__(self, name = ""):
        self.name = name

    @classmethod
    def search_all(cls, pattern, string):
        result = []
        finded = re.finditer(pattern=pattern, string=string)
        for find in finded:
            result.append(find.regs[0])
        return result

    @classmethod
    def get_aspecterm(self, x, y):
        result = []
        i = 0
        y.append("O")
        while i < len(y):
            if y[i].split("-")[0] == "B":
                aspecterm = []
                aspecterm.append(x[i])
                approx_pos = sum([len(word) + 1 for word in x[:i]])
                i += 1
                while y[i].split("-")[0] == "I" and i < len(y):
                    aspecterm.append(x[i])
                    i += 1
                result.append({"aspect_term": aspecterm, "approx_pos": approx_pos})
            else:
                i += 1
        return result

    def convert(self, filename, **kwargs):
        return

class ATEPC2APCConverter(ResultConverter):
    def __init__(self, name = "ATEPC2APC"):
        super(ATEPC2APCConverter, self).__init__(name)

    def convert(self, filename, **kwargs):
        fo_path = filename[:-4] + ".APC.tsv"
        if kwargs is not None:
            if "fo_path" in kwargs:
                fo_path = kwargs["fo_path"]
        copyfile(filename, fo_path)
        return fo_path

class ATEPC2ATEConverter(ResultConverter):
    def __init__(self, name = "ATEPC2ATE"):
        super(ATEPC2ATEConverter, self).__init__(name)

    def convert(self, filename, **kwargs):
        sents, labels, pred_labels = collect_data_from_tsv(filename)
        fo_path = filename[:-4] + ".ATE.tsv"
        if kwargs is not None:
            if "fo_path" in kwargs:
                fo_path = kwargs["fo_path"]

        fo = open(fo_path, mode="w")
        for words, tags, preds in zip(sents, labels, pred_labels):
            for word, tag, pred in zip(words, tags, preds):
                fo.write("{0}\t{1}\t{2}\n".format(word, self.convert_label(tag), self.convert_label(pred)))
            fo.write("\n")
        fo.close()
        return fo_path

    def convert_label(self, label):
        return label.split("-")[0]

class ATE2Semeval(ResultConverter):
    def __init__(self, name = "ATE2Semeval"):
        super(ATE2Semeval, self).__init__(name)

    def gen_aspecterm_positions(self, aspecterm, text):
            if text[-1] in [".", "?", "!", ","]:
                text = text[:-1]
            if len(aspecterm["aspect_term"]) == 1:
                aspecterm_str = re.escape(aspecterm["aspect_term"][0].strip("\\[].*?();"))
                regs = self.search_all(aspecterm_str, text.lower())
                if len(regs) == 1:
                    from_idx = regs[0][0]
                    to_idx = regs[0][1]
                    aspectterm_str_origin = text[from_idx:to_idx]
                    return aspectterm_str_origin, str(from_idx), str(to_idx)
                else:
                    suit_reg = None
                    shortest_leng = 100
                    for reg in regs:
                        average_idx = reg[0]
                        if abs(average_idx - aspecterm["approx_pos"]) < shortest_leng:
                            shortest_leng = abs(average_idx - aspecterm["approx_pos"])
                            suit_reg = copy.deepcopy(reg)
                    from_idx = suit_reg[0]
                    to_idx = suit_reg[1]
                    aspectterm_str_origin = text[from_idx:to_idx]
                    return aspectterm_str_origin, str(from_idx), str(to_idx)
            else:
                sta_aspecterm_str = re.escape(aspecterm["aspect_term"][0].strip("\\[].*?();"))
                end_aspecterm_str = re.escape(aspecterm["aspect_term"][-1].strip("\\[].*?();"))
                sta_regs = self.search_all(sta_aspecterm_str, text.lower())
                end_regs = self.search_all(end_aspecterm_str, text.lower())
                if len(sta_regs) == 0 or len(end_regs) == 0:
                    print ("ahihi")
                    return None, None, None

                suit_sta_reg = sta_regs[0]
                suit_end_reg = end_regs[0]
                shortest_leng = 100
                for sta_reg in sta_regs:
                    for end_reg in end_regs:
                        if sta_reg[1] < end_reg[0] and abs(end_reg[0] - sta_reg[1]) < shortest_leng:
                            suit_sta_reg = sta_reg
                            suit_end_reg = end_reg
                            shortest_leng = abs(end_reg[0] - sta_reg[1])
                from_idx = suit_sta_reg[0]
                to_idx = suit_end_reg[1]
                aspectterm_str_origin = text[from_idx:to_idx]
                return aspectterm_str_origin, str(from_idx), str(to_idx)

    def convert(self, tsvfile, **kwargs):

        if kwargs is not None:
            semfile = kwargs["semfile"]
        else:
            raise ("[!] semfile not found")

        if os.path.isfile(tsvfile) == False:
            raise ("[!] Data %s not found" % tsvfile)
        if os.path.isfile(semfile) == False:
            raise ("[!] Data %s not found" % semfile)

        # Collect sentences in tsv file
        sents, labels, pred_labels = collect_data_from_tsv(tsvfile)

        sent_idx = 0
        conflict_sent = ['912:1', '799:1', '1027:1', '2:1', '231:1', '463:26', '1008:1', '762:1', '786:1301', '786:1054', '416:1', '347:1', '11:1', '29:1' ,'32894246#870052#0', '33070309#423221#1', '32464601#418474#1', '11350539#680470#4', '11302357#835238#2', '35177381#521555#3', '11351451#805713#4', '11513049#499488#8', '33072753#1351349#2', '32936760#1397861#8', '32896473#439063#0', '32464601#418474#0', '11432442#650772#2', '11313290#1139539#1']

        # Collect sentences in semfile
        tree = ET.parse(semfile)
        root = tree.getroot()
        for sentence_tag in root.findall('sentence'):
            if sentence_tag.get("id") in conflict_sent:
                root.remove(sentence_tag)
                continue

            text = sentence_tag.find('text').text
            words, preds = sents[sent_idx], pred_labels[sent_idx]
            aspectterms = self.get_aspecterm(words, preds)
            aspectterms_tag = ET.Element("aspectTerms")

            for aspectterm in aspectterms:
                aspectterm_str_origin, from_idx, to_idx = self.gen_aspecterm_positions(aspectterm, sentence_tag.find('text').text)
                aspectterm_tag = ET.Element("aspectTerm")
                aspectterm_tag.set("term", aspectterm_str_origin)
                aspectterm_tag.set("from", str(from_idx))
                aspectterm_tag.set("to", str(to_idx))
                aspectterms_tag.append(aspectterm_tag)
            sentence_tag.append(aspectterms_tag)
            sent_idx +=1
        fo_path = os.path.join(os.getcwd(), "data/evaluation/ate_pred.xml")
        tree.write(fo_path, encoding="utf-8")
        return fo_path

class ResultEvaluator(object):
    def __init__(self, name = ""):
        self.name = name

    def evaluate(self, pred_file, gold_file = ""):
        return

class ATEEvaluator(ResultEvaluator):

    def __init__(self, name="ATE Evaluation"):
        super(ATEEvaluator, self).__init__(name)

    def evaluate(self, pred_file, gold_file=""):
        semeval_eval_jar = os.path.join(os.getcwd(),"data/evaluation/eval.jar")
        pred_file_path = pred_file
        # sai sai nha, cai laptops.ATEGold.xml nay la cua laptop dataset
        gold_file_path = gold_file
        tmp_file = os.path.join(os.getcwd(),"data/evaluation/tmp.txt")
        subprocess.call("java -cp {0} Main.Aspects {1} {2} > {3}".format(semeval_eval_jar, pred_file_path, gold_file_path, tmp_file), shell=True)
        result = {}
        with open(tmp_file, mode="r") as f:
            lines = f.readlines()
            if len(lines) < 5:
                return None
            result["precision"] = "{0:.2f}".format(float(lines[5].split()[1])*100)
            result["recall"] = "{0:.2f}".format(float(lines[6].split()[1])*100)
            result["f1-score"] = "{0:.2f}".format(float(lines[7].split()[1])*100)
        subprocess.call("rm {0}".format(tmp_file), shell=True)
        return result

class APCEvaluator(ResultEvaluator):
    def __init__(self, name="APC Evaluation"):
        super(APCEvaluator, self).__init__(name)

    def evaluate(self, pred_file, gold_file=""):
        senti_gold = []
        senti_pred = []
        sents, labels, pred_labels = collect_data_from_tsv(pred_file)
        for sent, label, pred in zip(sents, labels, pred_labels):
            sub_senti_gold, sub_senti_pred =  self.evaluate_sentence(sent, label, pred)
            senti_gold+=sub_senti_gold
            senti_pred+=sub_senti_pred

        senti_gold = np.array(senti_gold, dtype=np.int32)
        senti_pred = np.array(senti_pred, dtype=np.int32)
        accuracy = float(sum(senti_gold == senti_pred))/len(senti_gold)
        return {"accuracy": "{0:.2f}".format(accuracy*100)}

    def BIO2Sentiment(self, BIO_seq):
        term_mapping = {"POS": 1, "NEG": -1}

        sentiment_score = 0
        for term in BIO_seq:
            if term in term_mapping:
                sentiment_score += term_mapping[term]

        if sentiment_score > 0: return 2
        elif sentiment_score == 0: return 1
        else: return 0

    def evaluate_sentence(self, x, y, pred):
        senti_gold = []
        senti_pred = []
        i = 0
        y.append("O")
        while i < len(y):
            if y[i].split("-")[0] == "B":
                aspecterm = []
                gold_senti_seq = []
                pred_senti_seq = []
                aspecterm.append(x[i])
                gold_senti_seq.append(y[i].split("-")[-1])
                pred_senti_seq.append(pred[i].split("-")[-1])
                approx_pos = sum([len(word) + 1 for word in x[:i]])
                i += 1
                while y[i].split("-")[0] == "I" and i < len(y):
                    aspecterm.append(x[i])
                    gold_senti_seq.append(y[i].split("-")[-1])
                    pred_senti_seq.append(pred[i].split("-")[-1])
                    i += 1
                senti_gold.append(self.BIO2Sentiment(gold_senti_seq))
                senti_pred.append(self.BIO2Sentiment(pred_senti_seq))
            else:
                i += 1
        return senti_gold, senti_pred

class ContraintAPCEvaluator(APCEvaluator):
    def __init__(self, name="APC Evaluation under Constraint of ATE"):
        super(ContraintAPCEvaluator, self).__init__(name)

    def compare_2_lists(self, lista, listb):
        if len(lista) != len(listb):
            return False
        for i in range(len(lista)):
            if lista[i].split("-")[0] != listb[i].split("-")[0]:
                return False
        return True

    def get_pred_infor(self, x, y, pred):
        aspecterms = []
        labels_gold = []
        labels_pred = []
        i = 0
        y.append("O")
        while i < len(y):
            if y[i].split("-")[0] == "B":
                aspecterm = []
                label_gold = []
                label_pred = []
                aspecterm.append(x[i])
                label_gold.append(y[i])
                label_pred.append(pred[i])
                i += 1
                while y[i].split("-")[0] == "I" and i < len(y):
                    aspecterm.append(x[i])
                    label_gold.append(y[i])
                    label_pred.append(pred[i])
                    i += 1
                aspecterms.append(aspecterm)
                labels_gold.append(label_gold)
                labels_pred.append(label_pred)
            else:
                i += 1
        return aspecterms, labels_gold, labels_pred

    def evaluate(self, pred_file, gold_file=""):
        sentis_gold = []
        sentis_pred = []
        no_incorrect_ate = 0
        no_correct_ate = 0
        sents, labels, pred_labels = collect_data_from_tsv(pred_file)
        for sent, label, pred in zip(sents, labels, pred_labels):
            aspecterms, aspect_labels_gold, aspect_labels_pred = self.get_pred_infor(sent, label, pred)
            for aspectterm, aspect_label_gold,  aspect_label_pred in zip(aspecterms, aspect_labels_gold, aspect_labels_pred):
                is_correct_ate = self.compare_2_lists(aspect_label_gold, aspect_label_pred)
                if is_correct_ate is False:
                    no_incorrect_ate += 1
                    continue
                else:
                    no_correct_ate +=1
                    senti_gold = self.BIO2Sentiment([label.split("-")[-1] for label in aspect_label_gold])
                    senti_pred = self.BIO2Sentiment([label.split("-")[-1] for label in aspect_label_pred])
                    sentis_gold.append(senti_gold)
                    sentis_pred.append(senti_pred)
        sentis_gold = np.array(sentis_gold, dtype=np.int32)
        sentis_pred = np.array(sentis_pred, dtype=np.int32)
        accuracy = float(sum(sentis_gold == sentis_pred))/len(sentis_gold)
        return {"accuracy": "{0:.2f}".format(accuracy*100),
                "no_incorrect_ate": no_incorrect_ate,
                "no_correct_ate": no_correct_ate,
                "ate_correct_rate": "{0:.2f}".format((float(no_correct_ate)/(no_correct_ate+ no_incorrect_ate))*100)}


class ATEPCEvaluator(ResultEvaluator):
    def __init__(self, name="ATEPC Evaluation"):
        super(ATEPCEvaluator, self).__init__(name)
        self.ate_evaluator = ATEEvaluator()
        self.apc_evaluator = APCEvaluator()
        self.constraint_apc_evaluator = ContraintAPCEvaluator()
        self.ate_converter = ATEPC2ATEConverter()
        self.apc_converter = ATEPC2APCConverter()
        self.sem_converter = ATE2Semeval()

    def result_2_str(self, ate_result, apc_result, constraint_apc_result):
        result_str = ""
        result_str+="-- ATEPC without constraint --\n"
        result_str+="-- ATE --\n"
        result_str+="P: {0}, R: {1}, F1: {2}\n".format(ate_result["precision"], ate_result["recall"], ate_result["f1-score"])
        result_str+="-- APC --\n"
        result_str+="Acc: {0}\n".format(apc_result["accuracy"])
        result_str+="\n"
        result_str+="-- ATEPC with constraint --\n"
        result_str+="Acc: {0}, ATE correct rate: {1}\n".format(constraint_apc_result["accuracy"], constraint_apc_result["ate_correct_rate"])
        return result_str


    def evaluate(self, pred_file, gold_file = ""):
        if pred_file.find("laptops") != -1:
            ate_testphrase_file = os.path.join(os.getcwd(),"data/evaluation/laptops.ATETestPhrase.xml")
            ate_goldfile = os.path.join(os.getcwd(),"data/evaluation/laptops.ATEGold.xml")
        else:
            ate_testphrase_file = os.path.join(os.getcwd(), "data/evaluation/restaurants.ATETestPhrase.xml")
            ate_goldfile = os.path.join(os.getcwd(), "data/evaluation/restaurants.ATEGold.xml")

        # ATE evaluation
        ate_filepath = self.ate_converter.convert(pred_file)
        pred_sem_file = self.sem_converter.convert(ate_filepath, semfile=ate_testphrase_file)
        ate_result = self.ate_evaluator.evaluate(pred_sem_file, gold_file=ate_goldfile)
        subprocess.call("rm {0}".format(pred_sem_file), shell=True)

        # APC evaluation
        apc_filepath = self.apc_converter.convert(pred_file)
        apc_result = self.apc_evaluator.evaluate(apc_filepath)

        # Constraint APC Evaluation
        constraint_apc_result = self.constraint_apc_evaluator.evaluate(pred_file)

        print(self.result_2_str(ate_result, apc_result, constraint_apc_result))


if __name__ == "__main__":
    atepc_evaluator = ATEPCEvaluator()
    atepc_evaluator.evaluate("data/restaurants.ATEPC2.test.pred.tsv")
    atepc_evaluator.evaluate("data/laptops.ATEPC2.test.pred.tsv")


