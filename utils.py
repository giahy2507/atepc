import os

def collect_data_from_tsv(tsvfile):
    if os.path.isfile(tsvfile) == False:
        raise ("[!] Data %s not found" % tsvfile)
    # Collect sentences in tsv file
    sents, labels, pred_labels = [], [], []
    with open(tsvfile) as f:
        words, tags, preds = [], [], []
        for line in f:
            line = line.rstrip()
            if len(line) == 0 or line.startswith('-DOCSTART-'):
                if len(words) != 0:
                    sents.append(words)
                    labels.append(tags)
                    pred_labels.append(preds)
                    words, tags, preds = [], [], []
            else:
                tokens = line.split('\t')
                words.append(tokens[0])
                tags.append(tokens[1])
                if len(tokens) == 3:
                    preds.append(tokens[2])
                else:
                    preds.append("")
    return sents, labels, pred_labels