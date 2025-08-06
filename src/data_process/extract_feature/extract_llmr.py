import glob
import os
import pickle
import sys

import torch
import tqdm
import hydra
import pandas as pd
from omegaconf import DictConfig, OmegaConf
from transformers import TrainingArguments, Trainer, DataCollatorWithPadding, set_seed
import numpy as np


from src import utils
from src.utils import registry
from src.utils.train import print_config, process_config
from src.self_logger import logger, init_logger
from src.datasets.probiotics_dataset import ProbioticsDataProcess

def extract_llmr(config: DictConfig):
    set_seed(config.extract_llmr.train.seed)

    model = utils.config.instantiate(registry.model, config.extract_llmr.model)
    # logger.info(model)

    # Load the tokenizer and dataset
    dataset = utils.config.instantiate(registry.dataset, config.extract_llmr.dataset, partial=True)
    try:
        tokenizer = model.embedding.get_tokenizer
    except:
        tokenizer = None
    train_dataset, test_dataset, val_dataset = None, None, None

    # Set up training arguments
    training_args = TrainingArguments(
        label_names=["labels"],
        remove_unused_columns=False,
        save_total_limit=2,
        load_best_model_at_end=True,
        **config.extract_llmr.train,
    )

    # Set up Trainer
    data_collator = DataCollatorWithPadding(tokenizer)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if config.extract_llmr.train.do_train else None,
        eval_dataset=val_dataset if config.extract_llmr.train.do_eval else None,
        compute_metrics=utils.util.compute_metrics,
        data_collator=data_collator,
    )

    # Train model
    checkpoint = None
    if config.extract_llmr.train.resume_from_checkpoint is not None:
        checkpoint = config.extract_llmr.train.resume_from_checkpoint
        
    
    # Prediction
    with torch.no_grad():
        config.extract_llmr.dataset.dataset_name = str(config.extract_llmr.dataset.dataset_name)
        if config.extract_llmr.dataset.test_split is not None and config.extract_llmr.train.do_predict:
            test_dataset_path = os.path.join(config.extract_llmr.dataset.dest_path, config.extract_llmr.dataset.dataset_name)
            assert os.path.exists(test_dataset_path), f"test dataset not found: {test_dataset_path}"
            # logger.info(f"start predict from: {test_dataset_path}")
            test_splits = glob.glob(os.path.join(test_dataset_path, config.extract_llmr.dataset.test_split + ".txt"))
            test_splits = [os.path.basename(name).split(".txt")[0] for name in test_splits if ".txt" in name]

            # logger.info(f"predict data nums: {len(test_splits)}, which including:\n{test_splits}")


            pickles_dir = os.path.join(config.extract_llmr.train.output_dir, "pickles")
            os.makedirs(pickles_dir, exist_ok=True)


            for test_split in tqdm.tqdm(test_splits, desc=f"predict sample"):
                torch.cuda.empty_cache()
                
                output_pkl_path = os.path.join(pickles_dir, f'{test_split}.pkl')

                test_dataset = dataset(
                    tokenizer=tokenizer,
                    split=test_split,
                )
                # For probiotic sample predict
                try:
                    trainer.data_collator = test_dataset.data_collator
                except:
                    pass

                prediction_output = trainer.predict(test_dataset)
                if isinstance(prediction_output.predictions, tuple):
                    # output tuple with SequenceClassifierOutput
                    predictions, embeddings = prediction_output.predictions
                    embeddings = embeddings.tolist()
                else:
                    predictions, embeddings = prediction_output.predictions, None
                    embeddings = [None for _ in range(len(predictions))]

                predict_result = {
                    "sample_name": test_split,
                    "seqs_paths": test_dataset.sequences,
                    "seqs_labels": prediction_output.label_ids,
                    "model_predict": {
                        "embedding": embeddings,
                    }
                }
                with open(output_pkl_path, "wb") as f:
                    pickle.dump(predict_result, f)
                # logger.info(f"predict pkl result: {output_pkl_path}")
        else:
            logger.info(f"not valid data split name: {config.dataset.test_split}")
        
        # logger.info("Max GPU memory allocated:", torch.cuda.max_memory_allocated() / (1024 * 1024), "MB")
        # logger.info("Finished!")


@hydra.main(config_path="configs", config_name="config.yaml", version_base=None)
def main(config: OmegaConf):
    # check if the config is valid
    config = process_config(config)
    # set wandb_mode disabled
    os.environ["WANDB_MODE"] = "disabled"

    os.makedirs(config.extract_llmr.dataset.dest_path, exist_ok=True)
    os.makedirs(config.extract_llmr.train.output_dir, exist_ok=True)
    os.makedirs(config.extract_llmr.train.logging_dir, exist_ok=True)
    # print_config(config, resolve=True, save_dir=config.extract_llmr.train.logging_dir, prefix="extract_llmr")
    
    pbt_dp = ProbioticsDataProcess()
    short_genomes = pbt_dp.probiotics_check_sample_len(
        data_path=config.cut_seq.data_path,
        min_len=1000000, 
        max_len=10000000
    )
    pbt_dp.probiotics_single_sample_data(
        num_bound=(1000,12000), 
        len_bound=None, 
        dest_path= config.extract_llmr.dataset.pkl_path, 
        save_path=config.extract_llmr.dataset.dest_path,
        name="*",
        short_genomes=short_genomes
    )

    init_logger(svr_name="extract_llmr", log_path=config.extract_llmr.train.logging_dir)
    # logger.info("start extract_llmr...")
    extract_llmr(config)

    pbt_dp.probiotics_get_pickles_txt(dir=config.extract_llmr.train.output_dir)
    logger.info("Foundation model representation generation completed")

if __name__ == '__main__':
    # torch.cuda.set_device(3)
    main()