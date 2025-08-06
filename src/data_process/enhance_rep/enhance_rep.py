import glob
import os
import pickle
import sys

import torch, gc
import tqdm
import hydra
import pandas as pd
import numpy as np
from omegaconf import DictConfig, OmegaConf
from transformers import TrainingArguments, Trainer, DataCollatorWithPadding, set_seed

from src import utils
from src.utils import registry
from src.utils.train import print_config, process_config
from src.self_logger import logger, init_logger
from src.datasets.probiotics_dataset import  ProbioticSplitEnhanceRepresentationDataset
from src.datasets.probiotics_dataset import ProbioticsDataProcess


def enhance_rep(config: DictConfig):
    set_seed(config.enhance_rep.train.seed)

    model = utils.config.instantiate(registry.model, config.enhance_rep.model)
    # logger.info(model)

    # Load the tokenizer and dataset
    dataset = utils.config.instantiate(registry.dataset, config.enhance_rep.dataset, partial=True)
    # tokenizer = model.embedding.get_tokenizer
    train_dataset, test_dataset, val_dataset = None, None, None

    # Set up training arguments
    training_args = TrainingArguments(
        label_names=["labels"],
        remove_unused_columns=False, 
        save_total_limit=2,
        load_best_model_at_end=True,
        **config.enhance_rep.train,
    )

    # Set up Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if config.enhance_rep.train.do_train else None,
        eval_dataset=val_dataset if config.enhance_rep.train.do_eval else None,
        compute_metrics=utils.util.compute_metrics,
    )

    # Train model
    checkpoint = None
    if config.enhance_rep.train.resume_from_checkpoint is not None:
        checkpoint = config.enhance_rep.train.resume_from_checkpoint
     
    # Prediction
    with torch.no_grad():
        if config.enhance_rep.dataset.test_split is not None and config.enhance_rep.train.do_predict:
            sample_dataset = dataset(
                dest_path=config.enhance_rep.dataset.llm_rep_path,
                _dest_path=config.enhance_rep.dataset.ef_path,
                split=config.enhance_rep.dataset.test_split,
            )

            for sample in tqdm.tqdm(sample_dataset):
                gc.collect()
                torch.cuda.empty_cache()
 
                test_dataset = ProbioticSplitEnhanceRepresentationDataset(
                    seqs_labels=sample['seqs_labels'],
                    manual_feature=sample['manual_feature'],
                    embedding=sample['embedding'],
                )
                pickles_dir = os.path.join(config.enhance_rep.train.output_dir, "pickles")
                os.makedirs(pickles_dir, exist_ok=True)

                # For probiotic sample predict
                try:
                    trainer.data_collator = test_dataset.data_collator
                except:
                    pass

                prediction_output = trainer.predict(test_dataset)
                if isinstance(prediction_output.predictions, tuple):
                    # output tuple with SequenceClassifierOutput
                    predictions, embeddings = prediction_output.predictions
                    # after cross attention, the dimension will increase, need to squeeze to reduce the dimension
                    embeddings = np.squeeze(embeddings)
                    # embeddings = embeddings.tolist()
                else:
                    predictions, embeddings = prediction_output.predictions, None
                    embeddings = [None for _ in range(len(predictions))]

                predict_result = {
                    "sample_name": sample['sample_name'],
                    "seqs_paths": sample['seqs_paths'],
                    "seqs_labels": sample['seqs_labels'],
                    "model_predict": {
                        "embedding": embeddings,
                    }
                }
                output_pkl_path = os.path.join(pickles_dir, f"{sample['sample_name']}.pkl")
                with open(output_pkl_path, "wb") as f:
                    pickle.dump(predict_result, f)
                # logger.info(f"{sample['sample_name']} predict result saved to {output_pkl_path}")
        else:
            logger.info(f"not valid data split name: {config.enhance_rep.dataset.test_split}")
        
        # logger.info("Max GPU memory allocated:", torch.cuda.max_memory_allocated() / (1024 * 1024), "MB")


@hydra.main(config_path="configs", config_name="config.yaml", version_base=None)
def main(config: OmegaConf):
    # check if the config is valid
    config = process_config(config)
    # set wandb_mode disabled
    os.environ["WANDB_MODE"] = "disabled"

    os.makedirs(config.enhance_rep.dataset.llm_rep_path, exist_ok=True)
    os.makedirs(config.enhance_rep.dataset.ef_path, exist_ok=True)
    os.makedirs(config.enhance_rep.train.output_dir, exist_ok=True)
    os.makedirs(config.enhance_rep.train.logging_dir, exist_ok=True)
    if config.extract_llmr.model._name_ == "NTForClassifier":
        config.enhance_rep.train.resume_from_checkpoint = os.path.join(config.enhance_rep.train.resume_from_checkpoint, 'NT_50M')
    elif config.extract_llmr.model._name_ == "EvoForClassifier":
        config.enhance_rep.train.resume_from_checkpoint = os.path.join(config.enhance_rep.train.resume_from_checkpoint, 'EVO_7B')
    else:
        raise ValueError(f"Unknown model name: {config.extract_llmr.model._name_}")
    # print_config(config, resolve=True, save_dir=config.enhance_rep.train.logging_dir, prefix="enhance_rep")
   
    init_logger(svr_name="enhance_rep", log_path=config.enhance_rep.train.logging_dir)
    # logger.info("start enhance_rep...")
    enhance_rep(config)

    pbt_dp = ProbioticsDataProcess()
    pbt_dp.probiotics_get_pickles_txt(dir=config.enhance_rep.train.output_dir)
    logger.info("Representation enhancement completed")

if __name__ == '__main__':
    main()