#!/usr/bin/env python
import os
import pickle
import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

import hydra
from omegaconf import DictConfig, OmegaConf
from Bio import SeqIO

from src.utils.train import print_config, process_config
from src.self_logger import logger, init_logger 


def cut_sequence(sequence, cut_length, overlap_ratio):
    overlap = int(cut_length * overlap_ratio)
    for i in range(0, len(sequence), cut_length - overlap):
        if i + cut_length <= len(sequence):
            yield (sequence[i:i + cut_length], i, i + cut_length - 1)
        else:
            continue

def get_cut_sequence(sequence, split_length, overlap_ratio):
    split_list = []
    if len(sequence) <= split_length:
        split_list.append((sequence, 0, len(sequence) - 1))
    else:
        for (seq, start, end) in cut_sequence(sequence, split_length, overlap_ratio):
            split_list.append((seq, start, end))
    return split_list

def process_file(file_path, save_path, split_length, overlap_ratio, **kwargs):
    cut_dict = {}
    # set label to 1
    cut_dict['label'] = 1

    for record in SeqIO.parse(file_path, "fasta"):
        cut_dict['name'] = record.id
        # Convert sequence to uppercase
        sequence = str(record.seq.upper())
        split_list = get_cut_sequence(sequence, split_length, overlap_ratio)
        for item in split_list:
            cut_dict['seq'] = item[0]
            cut_dict['start'] = item[1]
            cut_dict['end'] = item[2]
            with open(os.path.join(save_path, f"{cut_dict['name']}_{cut_dict['start']}_{cut_dict['end']}.pkl"), 'wb') as file:
                pickle.dump(cut_dict, file)

def cut_seq(config: DictConfig):
    with ThreadPoolExecutor() as executor:
        futures = []
        # if input is a dir
        if os.path.isdir(config.cut_seq.data_path):
            for file_name in os.listdir(config.cut_seq.data_path):
                if file_name.endswith('.fna') or file_name.endswith('.fasta'):
                    file_path = os.path.join(config.cut_seq.data_path, file_name)
                    suffix = '.fna' if file_name.endswith('.fna') else '.fasta'
                    save_path = os.path.join(config.cut_seq.output_path, file_name.split(suffix)[0])
                    os.makedirs(save_path, exist_ok=True)
                    futures.append(executor.submit(process_file, file_path, save_path, **config.cut_seq))
        # if input is a single file
        else:
            futures.append(executor.submit(process_file, config.cut_seq.data_path, config.cut_seq.output_path, **config.cut_seq))

        for future in tqdm.tqdm(as_completed(futures), total=len(futures)):
            future.result()

    # print(f'Cut sequences saved to {config.cut_seq.output_path}')


@hydra.main(config_path="configs", config_name="config.yaml", version_base=None)
def main(config: OmegaConf):
    # check if the config is valid
    config = process_config(config)
    
    if config.cut_seq.data_path is None:
        raise ValueError("The 'input_path' parameter in config must be specified.")
    if config.cut_seq.output_dir is None:
        raise ValueError("The 'output_path' parameter in config must be specified.")

    # if finished, skip
    if os.path.exists(config.cut_seq.output_path) and len(os.listdir(config.cut_seq.output_path)) > 1:
        logger.info("The 'sequence segmentation' process has already been completed, skipping...")
        return
    
    os.makedirs(config.cut_seq.output_path, exist_ok=True)
    print_config(config, resolve=True, save_dir=os.path.join(config.cut_seq.output_path, 'logs'), prefix="cut_seq")

    # init logger
    init_logger(svr_name="cut_seq", log_path=os.path.join(config.cut_seq.output_path, 'logs'))
    # logger.info("Start cutting seq...")
    cut_seq(config)
    logger.info("Sequence segmentation completed​​​")


if __name__ == '__main__':
    main()