import os

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, matthews_corrcoef, accuracy_score, recall_score, precision_score

from src.self_logger import logger


class EarlyStopper:
    def __init__(self, patience=10, min_delta=1e-6):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_loss = np.Inf

    def __call__(self, loss):
        if np.isnan(loss):
            return True

        if loss < self.min_loss:
            self.min_loss = loss
            self.counter = 0
        elif loss > (self.min_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


def compute_metrics_f1_score(eval_pred):
    """Computes F1 score for binary classification"""
    predictions = np.argmax(eval_pred.predictions, axis=-1)
    references = eval_pred.label_ids
    r = {'f1_score': f1_score(references, predictions)}
    return r


def compute_metrics_with_class(label, predict):
    # TODO: 后续要兼容多分类，回归类等任务的评估
    binary_classification = len(np.unique(label)) == 2
    if binary_classification:
        result = {
            'mcc_score': matthews_corrcoef(label, predict),
            'f1_score': f1_score(label, predict),
            'accuracy_score': accuracy_score(label, predict),
            'recall_score': recall_score(label, predict),
            'precision_score': precision_score(label, predict),
        }
    else:
        result = {
            'mcc_score': matthews_corrcoef(label, predict),
            'f1_score': f1_score(label, predict, average='macro'),
            'accuracy_score': accuracy_score(label, predict),
            'recall_score': recall_score(label, predict, average='macro'),
            'precision_score': precision_score(label, predict, average='macro'),
        }
    return result


def compute_metrics(eval_pred):
    """Computes Matthews correlation coefficient (MCC score) for binary classification"""
    if isinstance(eval_pred.predictions, tuple) and len(eval_pred.predictions) == 2:
        _predictions, _ = eval_pred.predictions
        predictions = np.argmax(_predictions, axis=-1)
        references = eval_pred.label_ids
    elif isinstance(eval_pred.predictions, tuple) and isinstance(eval_pred.label_ids, dict):
        # 针对multi task设计的评估方法, 暂时这样实现，后续要进行优化
        results_metrics = {}
        for label_key, label_value in eval_pred.label_ids.items():
            if label_value.ndim != 1:  # 仅针对分类/回归损失
                continue
            for prediction in eval_pred.predictions:
                if isinstance(prediction, dict) and label_key in prediction:
                    _predictions = np.argmax(prediction[label_key], axis=-1)
                    _references = eval_pred.label_ids[label_key]
                    if _predictions.shape == _references.shape:
                        predictions = _predictions
                        references = _references
                        metric_result = compute_metrics_with_class(references, predictions)
                        for name, value in metric_result.items():
                            results_metrics[f"{label_key}_{name}"] = value
            return results_metrics

    else:
        predictions = np.argmax(eval_pred.predictions, axis=-1)
        references = eval_pred.label_ids
    result = compute_metrics_with_class(references, predictions)
    return result


def read_structure_data(path: str, start_colum: int = None, label=None, sep=None, **kwargs):
    """
    read data from path
    :param path: str, path of data
    :param start_colum: int, start column
    :param label: str, label column
    :param sep: str, separator
    :param kwargs: dict, other parameters
    :return: pd.DataFrame, data
    """
    assert os.path.exists(path), f"{path} is not exits"

    if os.path.basename(path).endswith("xlsx"):
        df = pd.read_excel(path)
    else:
        df = pd.read_csv(path, sep=sep, **kwargs)
    if start_colum:
        df = df.iloc[:, start_colum:]

    if label:
        return df.drop(label, axis=1), df[label]
    return df


def check_confi(config: dict, obj: object):
    """check config keys in obj
    Args:
        config: dict, config
        obj: object, object
    Returns:
        valid_keys: dict, valid keys
        redundant_keys: dict, redundant keys
    """
    keys = dir(obj)
    redundant_keys = {}
    valid_keys = {}
    for key in config.keys():
        if key not in keys:
            redundant_keys[key] = config[key]
        else:
            valid_keys[key] = config[key]
    if redundant_keys:
        logger.info(f"{str(obj)}: redundant keys: {redundant_keys}")
    return valid_keys, redundant_keys
