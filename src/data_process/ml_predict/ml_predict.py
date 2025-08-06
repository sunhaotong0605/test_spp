import json
import os
import sys
import shutil

root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, root_path)

import hydra
import joblib
import pandas as pd
from omegaconf import OmegaConf, DictConfig
from transformers import set_seed
import tqdm
import pickle
from matplotlib import pyplot as plt
from sklearn.model_selection import KFold
import numpy as np
import openpyxl

from src.self_logger import logger, init_logger
from src.utils.train import process_config, print_config
from src.datasets.ml_dataset import ProbioticTestDataset
from src.models.stacked_aggregation_classifier import MLClassModel, MLTrainer

def ml_predict(config: DictConfig):
    set_seed(config.ml_predict.train.seed)

    data_config = config.ml_predict.dataset
    train_dataset, test_dataset, eval_dataset = None, None, None
    
    model = MLClassModel(**config.ml_predict.model, split_num=config.ml_predict.dataset.split_num)
    trainer = MLTrainer(
        model=model,
        overwrite_output_dir=config.ml_predict.train.overwrite_output_dir,
        output_dir=config.ml_predict.train.output_dir,
        logging_dir=config.ml_predict.train.logging_dir,
        resume_from_checkpoint=config.ml_predict.train.resume_from_checkpoint,
    )

    if config.ml_predict.train.overwrite_output_dir:
        shutil.rmtree(config.ml_predict.train.output_dir, ignore_errors=True)

    if config.ml_predict.train.trained_model_path is not None:
        trainer.model.model = joblib.load(os.path.join(config.ml_predict.train.trained_model_path, "model.pkl"))
        trainer.model.final_model = joblib.load(os.path.join(config.ml_predict.train.trained_model_path, "final_model.pkl"))
    else:
        raise ValueError("trained_model_path is None, please set it in the config file")

    if config.ml_predict.dataset.test_split is not None:
        data_config.test_split = str(data_config.test_split)
        test_path = os.path.join(data_config.dest_path, data_config.dataset_name, data_config.test_split+'.txt')
        assert os.path.exists(test_path), f"test path: {test_path} not exists"
        test_dataset = ProbioticTestDataset(test_path, return_name=True, **data_config)
    else:
        raise ValueError("test_split is None, please set it in the config file")
    if test_dataset is not None and config.ml_predict.train.do_predict:
        # logger.info("Start predict ...")
        test_metrics, inter_prob1, ground_truth, test_predict, test_prob, test_names = trainer.test(test_dataset)
        sample_labels = []
        sample_prob = []
        for sample_idx in range(test_predict.shape[1]):
            sample_predictions = test_predict[:, sample_idx]
            if np.sum(sample_predictions == 0) > 5:
                sample_labels.append('Non-probiotics')
            else:
                sample_labels.append('Probiotics')
            sample_probability = test_prob[:, sample_idx, :]
            average_probability = np.mean(sample_probability, axis=0)
            if sample_labels[-1] == 'Non-probiotics':
                sample_prob.append(average_probability[0])
            else:
                sample_prob.append(average_probability[1])
        output_file = os.path.join(config.ml_predict.train.output_dir, f"{config.extract_llmr.model._name_}_Predict_Output.txt")
        with open(output_file, 'w') as f:
            f.write("name\tlabel\tscore\n")
            for i in range(len(test_names)):
                f.write(f"{test_names[i]}\t{sample_labels[i]}\t{sample_prob[i]:.4f}\n")

@hydra.main(config_path="configs", config_name="config.yaml", version_base=None)
def main(config: OmegaConf):
    # check if the config is valid
    config = process_config(config)

    os.makedirs(config.ml_predict.dataset.dest_path, exist_ok=True)
    os.makedirs(config.ml_predict.train.output_dir, exist_ok=True)
    os.makedirs(config.ml_predict.train.logging_dir, exist_ok=True)
    if config.extract_llmr.model._name_ == "NTForClassifier":
        config.ml_predict.train.trained_model_path = os.path.join(config.ml_predict.train.trained_model_path, 'NT_50M')
    elif config.extract_llmr.model._name_ == "EvoForClassifier":
        config.ml_predict.train.trained_model_path = os.path.join(config.ml_predict.train.trained_model_path, 'EVO_7B')
    else:
        raise ValueError(f"Unknown model name: {config.extract_llmr.model._name_}")
    # print_config(config, resolve=True, save_dir=config.enhance_rep.train.logging_dir, prefix="ml_predict")
   
    init_logger(svr_name="ml_predict", log_path=config.enhance_rep.train.logging_dir)
    # logger.info("start ml_predict...")
    ml_predict(config)
    logger.info("Prediction completed")

if __name__ == '__main__':
    main()
