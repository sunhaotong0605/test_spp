import numpy as np


class EvalFunc:
    """Evaluation functions for the model. such as accuracy, precision, recall, f1-score."""
    @staticmethod
    def get_acc_p_r_f1(trues, preds):
        labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        TP, FP, FN, TN = 0, 0, 0, 0
        for label in labels:
            preds_tmp = np.array([1 if pred == label else 0 for pred in preds])
            trues_tmp = np.array([1 if true == label else 0 for true in trues])
            TP += ((preds_tmp == 1) & (trues_tmp == 1)).sum()
            TN += ((preds_tmp == 0) & (trues_tmp == 0)).sum()
            FN += ((preds_tmp == 0) & (trues_tmp == 1)).sum()
            FP += ((preds_tmp == 1) & (trues_tmp == 0)).sum()
        # print(TP, FP, FN)
        precision = TP / (TP + FP)
        recall = TP / (TP + FN)
        f1 = 2 * precision * recall / (precision + recall)
        return precision, recall, f1

    @staticmethod
    def get_acc(trues, preds):
        accuracy = (np.array(trues) == np.array(preds)).sum() / len(trues)
        return accuracy
