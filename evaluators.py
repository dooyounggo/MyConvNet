"""
Model evaluation methods.
Includes accuracy, IoU, etc.
"""

from abc import abstractmethod, abstractproperty
import numpy as np


class Evaluator(object):
    def __init__(self, **kwargs):
        self.check_params(**kwargs)
        print('Evaluator: ' + self.name)
        print('')

    def check_params(self, **kwargs):
        pass

    @property
    @abstractmethod
    def name(self):
        """
        Name of the evaluator
        :return: string
        """
        pass

    @property
    @abstractmethod
    def worst_score(self):
        """
        The worst performance score.
        :return: float.
        """
        pass

    @property
    @abstractmethod
    def mode(self):
        """
        The mode for performance score, either 'max' or 'min'.
        e.g. 'max' for accuracy, AUC, precision, and recall,
             and 'min' for error rate, FNR, and FPR
        :return: str.
        """
        pass

    @abstractmethod
    def score(self, y_true, y_pred):
        """
        Performance metric for a given prediction.
        This should be implemented.
        :param y_true: np.ndarray.
        :param y_pred: np.ndarray.
        :return: float.
        """
        pass

    def is_better(self, curr, best, **kwargs):
        """
        Function to return whether current performance score is better than current best.
        This should be implemented.
        :param curr: float, current performance to be evaluated.
        :param best: float, current best performance
        :param kwargs:
        :return: bool.
        """
        score_threshold = kwargs.pop('score_threshold', 1e-3)
        relative_eps = 1.0 + score_threshold
        return curr >= best*relative_eps


class AccuracyEvaluator(Evaluator):
    @property
    def name(self):
        return 'Accuracy'

    @property
    def worst_score(self):
        return 0.0

    @property
    def mode(self):
        return 'max'

    def score(self, y_true, y_pred):
        if y_true.shape[-1] == 1:
            y_t = y_true[..., 0].astype(int)
            valid = np.greater_equal(y_t, 0)
        else:
            y_t = y_true.argmax(axis=-1)
            valid = np.isclose(y_true.sum(axis=-1), 1)
        if y_pred.shape[-1] == 1:
            y_p = y_pred[..., 0].astype(int)
        else:
            y_p = y_pred.argmax(axis=-1)
        right = np.equal(y_t, y_p)*valid

        accuracies = np.empty(y_t.shape[0], dtype=float)
        for i, (r, v) in enumerate(zip(right, valid)):
            valid_pixels = v.sum()
            if valid_pixels == 0:
                acc = 1
            else:
                acc = r.sum()/valid_pixels
            accuracies[i] = acc
        score = np.mean(accuracies)

        return score


class AccuracyTopNEvaluator(Evaluator):
    def check_params(self, **kwargs):
        if not hasattr(self, 'top'):
            self.top = kwargs.get('top', 1)

    @property
    def name(self):
        return 'Top-{} Accuracy'.format(self.top)

    @property
    def worst_score(self):
        return 0.0

    @property
    def mode(self):
        return 'max'

    def score(self, y_true, y_pred):
        assert y_pred.shape[-1] > 1, 'Labels must be one-hot encoded.'
        if y_true.shape[-1] > 1:
            y_t = np.expand_dims(y_true.argmax(axis=-1), axis=-1)
            valid = np.isclose(y_true.sum(axis=-1), 1)
        else:
            y_t = y_true
            valid = np.greater_equal(y_true[..., 0], 0)
        y_t = np.tile(y_t, self.top)
        y_p = np.argsort(y_pred, axis=-1)
        y_p = y_p[..., -self.top:]
        right = np.equal(y_t, y_p).sum(axis=-1)*valid

        accuracies = np.empty(y_t.shape[0], dtype=float)
        for i, (r, v) in enumerate(zip(right, valid)):
            valid_pixels = v.sum()
            if valid_pixels == 0:
                acc = 1
            else:
                acc = r.sum()/valid_pixels
            accuracies[i] = acc
        score = np.mean(accuracies)

        return score


class AccuracyTop1Evaluator(AccuracyTopNEvaluator):
    top = 1


class AccuracyTop2Evaluator(AccuracyTopNEvaluator):
    top = 2


class AccuracyTop3Evaluator(AccuracyTopNEvaluator):
    top = 3


class AccuracyTop5Evaluator(AccuracyTopNEvaluator):
    top = 5


class AccuracyTop10Evaluator(AccuracyTopNEvaluator):
    top = 10


class AccuracyCutMixEvaluator(AccuracyEvaluator):
    def score(self, y_true, y_pred):
        assert y_true.shape[-1] > 1 and y_pred.shape[-1] > 1, 'Labels must be one-hot encoded.'
        if np.sum(np.amax(y_true, axis=-1)) == np.prod(y_true.shape[:-1]).astype(float):  # No CutMix
            score = super().score(y_true, y_pred)
        else:
            y_t = np.argsort(y_true, axis=-1)
            y_t = y_t[..., -2:]  # Pick top-2
            y_p = np.argmax(y_pred, axis=-1)
            y_p = np.tile(np.expand_dims(y_p, axis=-1), 2)
            valid = np.isclose(y_true.sum(axis=-1), 1)
            right = np.equal(y_t, y_p).sum(axis=-1)*valid

            accuracies = np.empty(y_t.shape[0], dtype=float)
            for i, (r, v) in enumerate(zip(right, valid)):
                valid_pixels = v.sum()
                if valid_pixels == 0:
                    acc = 1
                else:
                    acc = r.sum()/valid_pixels
                accuracies[i] = acc
            score = np.mean(accuracies)

        return score


class ErrorEvaluator(AccuracyEvaluator):
    @property
    def name(self):
        return 'Error'

    @property
    def worst_score(self):
        return 1.0

    @property
    def mode(self):
        return 'min'

    def score(self, y_true, y_pred):
        return 1.0 - super().score(y_true, y_pred)

    def is_better(self, curr, best, **kwargs):
        score_threshold = kwargs.pop('score_threshold', 1e-3)
        relative_eps = 1.0 + score_threshold
        return curr <= best*relative_eps


class ErrorTopNEvaluator(AccuracyTopNEvaluator):
    @property
    def name(self):
        return 'Top-{} Error'.format(self.top)

    @property
    def worst_score(self):
        return 1.0

    @property
    def mode(self):
        return 'min'

    def score(self, y_true, y_pred):
        return 1.0 - super().score(y_true, y_pred)

    def is_better(self, curr, best, **kwargs):
        score_threshold = kwargs.pop('score_threshold', 1e-3)
        relative_eps = 1.0 + score_threshold
        return curr <= best*relative_eps


class ErrorTop1Evaluator(ErrorTopNEvaluator):
    top = 1


class ErrorTop2Evaluator(ErrorTopNEvaluator):
    top = 2


class ErrorTop3Evaluator(ErrorTopNEvaluator):
    top = 3


class ErrorTop5Evaluator(ErrorTopNEvaluator):
    top = 5


class ErrorTop10Evaluator(ErrorTopNEvaluator):
    top = 10


class ErrorCutMixEvaluator(AccuracyCutMixEvaluator):
    @property
    def name(self):
        return 'Error'

    @property
    def worst_score(self):
        return 1.0

    @property
    def mode(self):
        return 'min'

    def score(self, y_true, y_pred):
        return 1.0 - super().score(y_true, y_pred)

    def is_better(self, curr, best, **kwargs):
        score_threshold = kwargs.pop('score_threshold', 1e-3)
        relative_eps = 1.0 + score_threshold
        return curr <= best*relative_eps


class F1Evaluator(Evaluator):
    @property
    def name(self):
        return 'F1 Score'

    @property
    def worst_score(self):
        return 0.0

    @property
    def mode(self):
        return 'max'

    def score(self, y_true, y_pred):
        if y_true.shape[-1] == 1:
            y_t = y_true[..., 0].astype(int)
            valid = np.greater_equal(y_t, 0)
        else:
            y_t = y_true.argmax(axis=-1)
            valid = np.isclose(y_true.sum(axis=-1), 1)
        if y_pred.shape[-1] == 1:
            y_p = y_pred[..., 0].astype(int)
        else:
            y_p = y_pred.argmax(axis=-1)

        if len(y_t.shape) > 1:
            score = np.empty(y_t.shape[0], dtype=float)
            for i, (t, p, v) in enumerate(zip(y_t, y_p, valid)):
                precision, recall = precision_and_recall(t, p, valid=v)
                if precision == 0 or recall == 0:
                    val = 0
                else:
                    val = 2*precision*recall/(precision + recall)
                score[i] = val
            score = np.mean(score)
        else:
            precision, recall = precision_and_recall(y_t, y_p, valid=valid)
            score = 2*precision*recall/(precision + recall)

        return score


class MeanF1Evaluator(Evaluator):
    def check_params(self, **kwargs):
        self.bkgd_idx = kwargs.get('bkgd_idx', None)  # Background class to be ignored
        self.num_classes = kwargs.get('num_classes', None)

    @property
    def name(self):
        return 'Mean F1 Score'

    @property
    def worst_score(self):
        return 0.0

    @property
    def mode(self):
        return 'max'

    def score(self, y_true, y_pred):
        if y_true.shape[-1] == 1:
            y_t = y_true[..., 0].astype(int)
            valid = np.greater_equal(y_t, 0)
            num_classes = np.amax(y_t) + 1 if self.num_classes is None else self.num_classes
        else:
            y_t = y_true.argmax(axis=-1)
            valid = np.isclose(y_true.sum(axis=-1), 1)
            num_classes = y_true.shape[-1]
        if y_pred.shape[-1] == 1:
            y_p = y_pred[..., 0].astype(int)
        else:
            y_p = y_pred.argmax(axis=-1)

        score = []
        bkgd = self.bkgd_idx if isinstance(self.bkgd_idx, (list, tuple)) else [self.bkgd_idx]
        for n in range(num_classes):
            if n in bkgd:
                continue
            t = np.equal(y_t, n)
            p = np.equal(y_p, n)
            precision, recall = precision_and_recall(t, p, valid=valid)
            if precision == 0 or recall == 0:
                val = 0
            else:
                val = 2*precision*recall/(precision + recall)
            score.append(val)
        score = np.mean(score)

        return score


class IoUEvaluator(Evaluator):
    @property
    def name(self):
        return 'Intersection over Union'

    @property
    def worst_score(self):
        return 0.0

    @property
    def mode(self):
        return 'max'

    def score(self, y_true, y_pred):
        if y_true.shape[-1] == 1:
            y_t = y_true[..., 0].astype(int)
            valid = np.greater_equal(y_t, 0)
        else:
            y_t = y_true.argmax(axis=-1)
            valid = np.isclose(y_true.sum(axis=-1), 1)
        if y_pred.shape[-1] == 1:
            y_p = y_pred[..., 0].astype(int)
        else:
            y_p = y_pred.argmax(axis=-1)

        if len(y_t.shape) > 1:
            score = np.empty(y_t.shape[0], dtype=float)
            for i, (t, p, v) in enumerate(zip(y_t, y_p, valid)):
                tp, fp, _, fn = conditions(t, p, valid=v)
                if tp == 0:
                    val = 0
                else:
                    val = tp/(tp + fp + fn)
                score[i] = val
            score = np.mean(score)
        else:
            tp, fp, _, fn = conditions(y_t, y_p, valid=valid)
            score = tp/(tp + fp + fn)

        return score


class MeanIoUEvaluator(Evaluator):
    def check_params(self, **kwargs):
        self.bkgd_idx = kwargs.get('bkgd_idx', None)  # Background class to be ignored
        self.num_classes = kwargs.get('num_classes', None)

    @property
    def name(self):
        return 'Mean Intersection over Union'

    @property
    def worst_score(self):
        return 0.0

    @property
    def mode(self):
        return 'max'

    def score(self, y_true, y_pred):
        if y_true.shape[-1] == 1:
            y_t = y_true[..., 0].astype(int)
            valid = np.greater_equal(y_t, 0)
            num_classes = np.amax(y_t) + 1 if self.num_classes is None else self.num_classes
        else:
            y_t = y_true.argmax(axis=-1)
            valid = np.isclose(y_true.sum(axis=-1), 1)
            num_classes = y_true.shape[-1]
        if y_pred.shape[-1] == 1:
            y_p = y_pred[..., 0].astype(int)
        else:
            y_p = y_pred.argmax(axis=-1)

        score = []
        bkgd = self.bkgd_idx if isinstance(self.bkgd_idx, (list, tuple)) else [self.bkgd_idx]
        for n in range(num_classes):
            if n in bkgd:
                continue
            t = np.equal(y_t, n)
            p = np.equal(y_p, n)
            tp, fp, _, fn = conditions(t, p, valid=valid)
            if tp == 0:
                val = 0
            else:
                val = tp/(tp + fp + fn)
            score.append(val)
            # print('class {}: {}'.format(n, val))
        score = np.mean(score)

        return score


class NullEvaluator(Evaluator):
    Y_TRUE = 'return_y_true'
    Y_PRED = 'return_y_pred'

    def check_params(self, **kwargs):
        self.to_return = kwargs.get('to_return', 0.0)
        self.score_name = kwargs.get('score_name', 'Score')
        self.score_mode = kwargs.get('score_mode', 'max')
        self._worst_score = kwargs.get('worst_score', 0.0)

    @property
    def name(self):
        return self.score_name

    @property
    def worst_score(self):
        return self._worst_score

    @property
    def mode(self):
        return self.score_mode

    def score(self, y_true, y_pred):
        if self.to_return == NullEvaluator.Y_TRUE:
            score = y_true.mean()
        elif self.to_return == NullEvaluator.Y_PRED:
            score = y_pred.mean()
        else:
            score = self.to_return

        return score


def precision_and_recall(y_t, y_p, valid=None):     # Precision and recall.
    tp, fp, _, fn = conditions(y_t, y_p, valid)
    if tp == 0:
        prec = 0
        rec = 0
    else:
        prec = tp/(tp + fp)
        rec = tp/(tp + fn)

    return prec, rec


def conditions(y_t, y_p, valid=None):   # Number of true positives, false positives, true negatives, and false negatives
    if valid is None:
        valid = np.ones(y_t.shape)

    positive = np.greater(y_p, 0)*valid
    negative = np.equal(y_p, 0)*valid
    right = np.equal(y_t, y_p)
    wrong = np.not_equal(y_t, y_p)

    tp = np.sum(right*positive)
    fp = np.sum(wrong*positive)
    tn = np.sum(right*negative)
    fn = np.sum(wrong*negative)

    return tp, fp, tn, fn
