import json
import os
from typing import Optional

import joblib
import numpy as np
from sklearn import linear_model
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import matthews_corrcoef, f1_score, accuracy_score, precision_score, recall_score, roc_auc_score, confusion_matrix
from sklearn.model_selection import GridSearchCV
from torch.utils.data import DataLoader, Dataset
from transformers import DataCollator, set_seed
from xgboost import XGBClassifier
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.decomposition import PCA, KernelPCA


from src.self_logger.base import logger

 
class MLClassModel(object):
    def __init__(
            self, 
            model_name, 
            final_model_name = None, 
            param_grid: dict = None, 
            cv: int = 5, 
            max_length = 1000, 
            split_num=-1,
            method="stacked",
            svc_max_iter=-1,
            **kwargs):
        self.param_grid = dict(param_grid) if param_grid is not None else {}
        self.model_name = model_name
        self.max_length = max_length
        self.final_model_name = final_model_name
        self.split_num = 100 if split_num == -1 else split_num
        self.cv = cv
        self.method = method
        self.svc_max_iter = svc_max_iter
        self.pca = PCA(n_components=1)
        self.kpca = KernelPCA(n_components=1, kernel='rbf')
        
        if self.model_name == "XGBClassifier":
            self.model = XGBClassifier()
        elif self.model_name == "RandomForestClassifier":
            self.model = RandomForestClassifier()
        elif self.model_name == "ElasticNet":
            self.model = linear_model.ElasticNet()
        elif self.model_name == "LogisticRegression":
            self.model = LogisticRegression()
        elif self.model_name == "SVC":
            if self.svc_max_iter == -1:
                self.model = SVC(probability=True)
            else:
                self.model = SVC(probability=True, max_iter=self.svc_max_iter)
        else:
            TypeError(f"{self.model_name} not support!")
        self.final_model = None 
        if self.final_model_name == "XGBClassifier":
            self.final_model = XGBClassifier()
        elif self.final_model_name == "RandomForestClassifier":
            self.final_model = RandomForestClassifier()
        elif self.final_model_name == "ElasticNet":
            self.final_model = linear_model.ElasticNet()
        elif self.final_model_name == "LogisticRegression":
            self.final_model = LogisticRegression()
        elif self.final_model_name == "SVC":
            self.final_model = SVC(probability=True, max_iter=self.svc_max_iter)
        else:
            TypeError(f"{self.final_model_name} not support!")
 
    def fit(self, input_features, labels, **kwargs):
        if input_features.ndim == 2:
            lr_params = LogisticRegression().get_params().keys()
            lr_param_dict = {}
            for key, value in self.param_grid.items():
                if value is not None and key in lr_params:
                    lr_param_dict[key] = value
            if lr_params:
                lr_model = LogisticRegression(**lr_param_dict)
            else:
                lr_model = LogisticRegression()
            # 500 ORF segments, each 100 segments as one group, total 5 groups
            for i in range(5):
                lr_model.fit(input_features[i,:].reshape(-1,self.split_num), labels)
                print(f"train LR: NO.{i} training completed")
            self.final_model, result = lr_model, {}
            return result
        elif input_features.ndim == 3: 
            split_num_dim = input_features.shape[1]
            train_features = input_features[:,:int(split_num_dim/2),:]
            test_features = input_features[:,int(split_num_dim/2):,:]
            # train
            labels_list = []
            for label in labels:
                labels_list.extend([label] * train_features.shape[1])
            train_labels = np.array(labels_list)
            train_features = train_features.reshape(-1, train_features.shape[2])
            xgboost_params = XGBClassifier().get_params().keys()
            xgb_param_dict = {}
            for key, value in self.param_grid.items():
                if value is not None and key in xgboost_params:
                    xgb_param_dict[key] = value
            if xgb_param_dict:
                xgb_model = XGBClassifier(**xgb_param_dict)
            else:
                xgb_model = XGBClassifier()
            xgb_model.fit(train_features, train_labels)   
            # predict
            test_prob_all = []
            for i in range(0,int(split_num_dim/2), self.split_num):
                test_prob = xgb_model.predict_proba(test_features[:,i:i+self.split_num,:].reshape(-1, test_features.shape[2])).tolist()
                test_prob_all.append(test_prob)
            test_prob_all = np.array(test_prob_all)
            test_result_dict = {"train_prob0": test_prob_all[:,:,0], "train_prob1": test_prob_all[:,:,1],"label": labels}
            self.model, result = xgb_model, test_result_dict
            return result
        else:
            raise ValueError("input_features shape error!")
        
    
    def fit_simple(self, input_features, labels, **kwargs):
        sample_num_dim, split_num_dim, feature_dim = input_features.shape
        train_labels = np.array(labels)
        model_params = self.model.get_params().keys()
        model_param_dict = {}
        for key, value in self.param_grid.items():
            if value is not None and key in model_params:
                model_param_dict[key] = value
        if model_param_dict:
            model.set_params(**model_param_dict)
        else:
            model = self.model
        if self.method == "mean":
            train_features = np.mean(input_features, axis=1)
        elif self.method == "max":
            train_features = np.max(input_features, axis=1)
        elif self.method == "min":
            train_features = np.min(input_features, axis=1)
        elif self.method == "voting":
            train_features = input_features.reshape(-1, input_features.shape[2])
            labels_list = []
            for label in train_labels:
                labels_list.extend([label] * input_features.shape[1])
            train_labels = np.array(labels_list)
        elif self.method == "pca":
            train_features = self.pca.fit_transform(input_features.transpose(0,2,1).reshape(-1, split_num_dim)).reshape(sample_num_dim, feature_dim)
        elif self.method == "kpca":
            # choice_sample_num = 20
            # sample_indices = np.random.choice(input_features.shape[0], size=choice_sample_num, replace=False)
            # sampled_input_features = input_features[sample_indices]
            # train_features = self.kpca.fit_transform(sampled_input_features.transpose(0,2,1).reshape(-1, split_num_dim)).reshape(choice_sample_num, feature_dim)
            # train_labels = train_labels[sample_indices]
            
            train_features = self.kpca.fit_transform(input_features.transpose(0,2,1).reshape(-1, split_num_dim)).reshape(sample_num_dim, feature_dim)
        else:
            raise ValueError(f"method {self.method} not support!")
        model.fit(train_features, train_labels)   
        self.model, result = model, None
        return result


    def predict_prob(self, input_features, labels, names, **kwargs):
        # first classifier
        split_num_dim = input_features.shape[1]
        input_features = input_features.reshape(-1, input_features.shape[2])
        prob1 = self.model.predict_proba(input_features)[:,1]
        # inter_prob1 represents the results of the 10 groups of experiments for 18 samples
        inter_prob1 = []
        for index in range(0, split_num_dim, self.split_num):
            # output represents the results of the i-th group of experiments for 18 samples
            output = []
            for i in range(index, len(prob1), split_num_dim):
                output.extend(prob1[i:i+self.split_num])
            inter_prob1.append(output)
        # second classifier
        lr_input = prob1.reshape(-1, split_num_dim)
        pred_list = []
        prob_list = []
        for i in range(0, split_num_dim, self.split_num):
            pred_list.append(self.final_model.predict(lr_input[:,i:i+self.split_num]).tolist())
            prob_list.append(self.final_model.predict_proba(lr_input[:,i:i+self.split_num]).tolist())
        pred_array = np.array(pred_list)
        prob_array = np.array(prob_list)
        metrics_list = []
        # for i in range(10):
        #     metrics_list.append(self.compute_metrics(pred_array[i, :], labels, prob_array[i, :, 1]))
        return metrics_list, np.array(inter_prob1), labels, pred_array, prob_array, names
   
   
    def predict_prob_simple(self, input_features, labels, names, **kwargs):
        sample_num_dim, split_num_dim, feature_dim = input_features.shape
        pred_list = []
        prob_list = []
        metrics_list = []
        
        for i in range(0, split_num_dim, self.split_num):
            split_features = input_features[:,i:i+self.split_num,:]
            if self.method == "mean":
                test_features = np.mean(split_features, axis=1)
            elif self.method == "max":
                test_features = np.max(split_features, axis=1)
            elif self.method == "min":
                test_features = np.min(split_features, axis=1)
            elif self.method == "voting":
                # (128, 100, 132) -> (128*100, 132)
                test_features = split_features.reshape(-1, feature_dim)
            elif self.method == "pca":
                test_features = self.pca.fit_transform(split_features.transpose(0,2,1).reshape(-1, self.split_num)).reshape(sample_num_dim, feature_dim)
            elif self.method == "kpca":
                test_features = self.kpca.fit_transform(split_features.transpose(0,2,1).reshape(-1, self.split_num)).reshape(sample_num_dim, feature_dim)
            else:
                raise ValueError(f"method {self.method} not support!")
            if self.method == "voting":
                all_pred_result = self.model.predict(test_features)
                all_prob_result = self.model.predict_proba(test_features)
                # result = [Counter(result[i:i+self.split_num]).most_common(1)[0][0] for i in range(0, len(result), self.split_num)]
                group_pred_list = []
                group_prob_list = []
                for i in range(0, len(all_pred_result), self.split_num):
                    sample_pred = all_pred_result[i:i+self.split_num].tolist()
                    sample_prob = all_prob_result[i:i+self.split_num].tolist()
                    if sample_pred.count(1) >= int(len(sample_pred)/2):
                        group_pred_list.append(1)
                    else:
                        group_pred_list.append(0)
                    group_prob_list.append([sum([row[0] for row in sample_prob])/len(sample_prob), sum([row[1] for row in sample_prob])/len(sample_prob)])
                pred_list.append(group_pred_list)
                prob_list.append(group_prob_list)
            else:
                pred_list.append(self.model.predict(test_features).tolist())
                prob_list.append(self.model.predict_proba(test_features).tolist())
            metrics_list.append(self.compute_metrics(pred_list[-1], labels, [row[1] for row in prob_list[-1]]))
        pred_array = np.array(pred_list)
        prob_array = np.array(prob_list)
        return metrics_list, None, labels, pred_array, prob_array, names
        

    @staticmethod
    def compute_metrics(predictions, labels, probability) -> dict:
        """Computes Matthews correlation coefficient (MCC score) for binary classification"""
        tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
        spe = tn/(tn+fp)
        recall = recall_score(labels, predictions)
        gep = (recall*spe)**0.5
        try:
            auc = roc_auc_score(labels, probability) ## y_true=ground_truth
        except:
            auc = -1
        r = {
            'mcc_score': matthews_corrcoef(labels, predictions),
            'f1_score': f1_score(labels, predictions),
            'accuracy_score': accuracy_score(labels, predictions),
            'recall_score': recall_score(labels, predictions),
            'precision_score': precision_score(labels, predictions),
            'spe': spe,
            'gep': gep,
            'auc': auc,
            'tn': tn,
            'fp': fp,
            'fn': fn,
            'tp': tp,
        }
        return r


class MLTrainer:
    def __init__(
            self,
            model: MLClassModel,
            train_dataset: Optional[Dataset] = None,
            data_collator: Optional[DataCollator] = None,
            seed: int = 42,
            overwrite_output_dir: bool = False,
            resume_from_checkpoint: Optional[str] = None,
            output_dir: Optional[str] = None,
            logging_dir: Optional[str] = None,
            save_model: Optional[bool] = False,
    ):
        self.model: MLClassModel = model
        self.train_dataset: Dataset = train_dataset
        self.model_name = self.model.model_name
        self.data_collator = data_collator if data_collator is not None else self._data_collator
        self.seed = seed

        self.overwrite_output_dir: bool = overwrite_output_dir
        self.resume_from_checkpoint: Optional[str] = resume_from_checkpoint
        self.output_dir: Optional[str] = output_dir
        self.logging_dir: Optional[str] = logging_dir
        self.save_model: Optional[bool] = save_model if self.output_dir is not None else False

        self.init_trainer()

    def set_seed(self, seed: Optional[int] = None):
        if seed is None:
            seed = self.seed
        set_seed(seed=seed)

    def init_trainer(self):
        self.set_seed(self.seed)

    def train(self, train_dataset: Optional[Dataset] = None, resume_from_checkpoint: Optional[str] = None,**kwargs):
        if train_dataset is None:
            train_dataset = self.train_dataset
        input_features, labels = self.data_collator(train_dataset)

        final_result = None
        if self.model.method == 'stacked':
            result = self.model.fit(input_features, labels, **kwargs)
            final_result = self.model.fit(result["train_prob1"], result["label"])
        elif self.model.method in ['max', 'min', 'mean','voting', 'pca', 'kpca']:
            result = self.model.fit_simple(input_features, labels, **kwargs)
        else:
            raise ValueError(f"method {self.model.method} not support!")
        return result, final_result
       
    def test(self, x_test: Dataset):
        input_features, labels, names = self.data_collator(x_test)
        if self.model.method == 'stacked':
            return self.model.predict_prob(input_features, labels, names)
        elif self.model.method in ['max', 'min', 'mean','voting', 'pca', 'kpca']:
            return self.model.predict_prob_simple(input_features, labels, names)
        else:
            raise ValueError(f"method {self.model.method} not support!")

        
    @staticmethod
    def _data_collator(dataset: Dataset):
        loader = DataLoader(dataset, batch_size=len(dataset))
        # loader = DataLoader(dataset, batch_size=2)
        if dataset.return_name:
            input_features, labels, names = next(iter(loader))
            input_features, labels = input_features.numpy(), labels.numpy()
            return input_features, labels, names
        else:
            input_features, labels = next(iter(loader))
            input_features, labels = input_features.numpy(), labels.numpy()
            return input_features, labels
