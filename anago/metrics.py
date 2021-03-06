import os

import numpy as np
from keras.callbacks import Callback, TensorBoard, EarlyStopping, ModelCheckpoint

def get_callbacks(valid=(), eary_stopping=True, save_path = None ,patience = 20):
    """Get callbacks.

    Args:
        log_dir (str): the destination to save logs(for TensorBoard).
        valid (tuple): data for validation.
        tensorboard (bool): Whether to use tensorboard.
        eary_stopping (bool): whether to use early stopping.

    Returns:
        list: list of callbacks
    """
    callbacks = []

    if valid:
        callbacks.append(F1score(*valid))

    if save_path:
        save_callback = ModelCheckpoint(save_path,
                                        monitor='f1',
                                        save_best_only=True, mode="max", save_weights_only=True, verbose=0)
        callbacks.append(save_callback)

    if eary_stopping:
        callbacks.append(EarlyStopping(monitor='f1', patience=patience, mode='max'))

    return callbacks

def get_mycallbacks(valid=(), eary_stopping=True, save_path = None ,patience = 20):
    """Get callbacks.

    Args:
        log_dir (str): the destination to save logs(for TensorBoard).
        valid (tuple): data for validation.
        tensorboard (bool): Whether to use tensorboard.
        eary_stopping (bool): whether to use early stopping.

    Returns:
        list: list of callbacks
    """
    callbacks = []

    if valid:
        callbacks.append(MyF1score(*valid))

    if save_path:
        save_callback = ModelCheckpoint(save_path,
                                        monitor='f1',
                                        save_best_only=True, mode="max", save_weights_only=True, verbose=0)
        callbacks.append(save_callback)

    if eary_stopping:
        callbacks.append(EarlyStopping(monitor='f1', patience=patience, mode='max'))

    return callbacks



def get_entities(seq):
    """Gets entities from sequence.

    Args:
        seq (list): sequence of labels.

    Returns:
        list: list of (chunk_type, chunk_start, chunk_end).

    Example:
        >>> seq = ['B-PER', 'I-PER', 'O', 'B-LOC']
        >>> print(get_entities(seq))
        [('PER', 0, 2), ('LOC', 3, 4)]
    """
    i = 0
    chunks = []
    seq = seq + ['O']  # add sentinel
    types = [tag.split('-')[-1] for tag in seq]
    while i < len(seq):
        if seq[i].startswith('B'):
            for j in range(i+1, len(seq)):
                if seq[j].startswith('I') and types[j] == types[i]:
                    continue
                break
            chunks.append((types[i], i, j))
            i = j
        else:
            i += 1
    return chunks

def f1_score(y_true, y_pred, sequence_lengths):
    """Evaluates f1 score.

    Args:
        y_true (list): true labels.
        y_pred (list): predicted labels.
        sequence_lengths (list): sequence lengths.

    Returns:
        float: f1 score.

    Example:
        >>> y_true = []
        >>> y_pred = []
        >>> sequence_lengths = []
        >>> print(f1_score(y_true, y_pred, sequence_lengths))
        0.8
    """
    correct_preds, total_correct, total_preds = 0., 0., 0.
    for lab, lab_pred, length in zip(y_true, y_pred, sequence_lengths):
        lab = lab[:length]
        lab_pred = lab_pred[:length]

        lab_chunks = set(get_entities(lab))
        lab_pred_chunks = set(get_entities(lab_pred))

        correct_preds += len(lab_chunks & lab_pred_chunks)
        total_preds += len(lab_pred_chunks)
        total_correct += len(lab_chunks)

    p = correct_preds / total_preds if correct_preds > 0 else 0
    r = correct_preds / total_correct if correct_preds > 0 else 0
    f1 = 2 * p * r / (p + r) if correct_preds > 0 else 0
    return f1


class F1score(Callback):

    def __init__(self, valid_steps, valid_batches, preprocessor=None):
        super(F1score, self).__init__()
        self.valid_steps = valid_steps
        self.valid_batches = valid_batches
        self.p = preprocessor


    def on_epoch_end(self, epoch, logs={}):
        correct_preds, total_correct, total_preds = 0., 0., 0.
        count = 0
        for i, (data, label) in enumerate(self.valid_batches):
            if i == self.valid_steps:
                break
            y_true = label
            y_true = np.argmax(y_true, -1)
            count+=y_true.shape[0]
            sequence_lengths = data[-1] # shape of (batch_size, 1)
            sequence_lengths = np.reshape(sequence_lengths, (-1,))
            #y_pred = np.asarray(self.model_.predict(data, sequence_lengths))
            y_pred = self.model.predict_on_batch(data)
            y_pred = np.argmax(y_pred, -1)

            y_pred = [self.p.inverse_transform(y[:l]) for y, l in zip(y_pred, sequence_lengths)]
            y_true = [self.p.inverse_transform(y[:l]) for y, l in zip(y_true, sequence_lengths)]

            a, b, c = self.count_correct_and_pred(y_true, y_pred, sequence_lengths)
            correct_preds += a
            total_preds += b
            total_correct += c
            count+=1

        f1 = self._calc_f1(correct_preds, total_correct, total_preds)
        logs['f1'] = f1
        print('\nF1 on {0}, {1:.2f}'.format(count ,f1 * 100))
        return f1

    def _calc_f1(self, correct_preds, total_correct, total_preds):
        p = correct_preds / total_preds if correct_preds > 0 else 0
        r = correct_preds / total_correct if correct_preds > 0 else 0
        f1 = 2 * p * r / (p + r) if correct_preds > 0 else 0
        return f1

    def count_correct_and_pred(self, y_true, y_pred, sequence_lengths):
        correct_preds, total_correct, total_preds = 0., 0., 0.
        for lab, lab_pred, length in zip(y_true, y_pred, sequence_lengths):
            lab = lab[:length]
            lab_pred = lab_pred[:length]

            lab_chunks = set(get_entities(lab))
            lab_pred_chunks = set(get_entities(lab_pred))

            correct_preds += len(lab_chunks & lab_pred_chunks)
            total_preds += len(lab_pred_chunks)
            total_correct += len(lab_chunks)
        return correct_preds, total_correct, total_preds


class MyF1score(Callback):

    def __init__(self, X_valid, Y_valid ,  preprocessor=None):
        super(MyF1score, self).__init__()
        self.X_valid = X_valid
        self.Y_valid = Y_valid
        self.p = preprocessor


    def on_epoch_end(self, epoch, logs={}):
        correct_preds, total_correct, total_preds = 0., 0., 0.
        count = 0

        data, label = self.X_valid, self.Y_valid
        y_true = label
        y_true = np.argmax(y_true, -1)
        count += y_true.shape[0]
        sequence_lengths = data[-1]  # shape of (batch_size, 1)
        sequence_lengths = np.reshape(sequence_lengths, (-1,))
        # y_pred = np.asarray(self.model_.predict(data, sequence_lengths))
        y_pred = self.model.predict_on_batch(data)
        y_pred = np.argmax(y_pred, -1)

        y_pred = [self.p.inverse_transform(y[:l]) for y, l in zip(y_pred, sequence_lengths)]
        y_true = [self.p.inverse_transform(y[:l]) for y, l in zip(y_true, sequence_lengths)]

        a, b, c = self.count_correct_and_pred(y_true, y_pred, sequence_lengths)
        correct_preds += a
        total_preds += b
        total_correct += c

        f1 = self._calc_f1(correct_preds, total_correct, total_preds)
        logs['f1'] = f1
        print('. F1 on {0}, {1:.2f}'.format(count ,f1 * 100))
        return f1

    def _calc_f1(self, correct_preds, total_correct, total_preds):
        p = correct_preds / total_preds if correct_preds > 0 else 0
        r = correct_preds / total_correct if correct_preds > 0 else 0
        f1 = 2 * p * r / (p + r) if correct_preds > 0 else 0
        return f1

    def count_correct_and_pred(self, y_true, y_pred, sequence_lengths):
        correct_preds, total_correct, total_preds = 0., 0., 0.
        for lab, lab_pred, length in zip(y_true, y_pred, sequence_lengths):
            lab = lab[:length]
            lab_pred = lab_pred[:length]

            lab_chunks = set(get_entities(lab))
            lab_pred_chunks = set(get_entities(lab_pred))

            correct_preds += len(lab_chunks & lab_pred_chunks)
            total_preds += len(lab_pred_chunks)
            total_correct += len(lab_chunks)
        return correct_preds, total_correct, total_preds