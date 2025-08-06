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
    fna_row_list = []
    for record in SeqIO.parse(file_path, "fasta"):
        sequence = str(record.seq.upper())
        split_list = get_cut_sequence(sequence, split_length, overlap_ratio)
        for item in split_list:
            fna_row_list.append(f'>{record.id}_{item[1]}_{item[2]}')
            fna_row_list.append(item[0])
    with open(os.path.join(save_path, os.path.basename(file_path).replace(".fasta", ".fna")), 'w') as file:
        for row in fna_row_list:
            file.write(row + '\n')

def cut_seq_to_fna(config: DictConfig):
    with ThreadPoolExecutor() as executor:
        futures = []
        # if input is a dir
        if os.path.isdir(config.cut_seq_to_fna.data_path):
            for file_name in os.listdir(config.cut_seq_to_fna.data_path):
                if file_name.endswith('.fna') or file_name.endswith('.fasta'):
                    file_path = os.path.join(config.cut_seq_to_fna.data_path, file_name)
                    futures.append(executor.submit(process_file, file_path, config.cut_seq_to_fna.output_path, **config.cut_seq_to_fna))
        # if input is a single file
        else:
            futures.append(executor.submit(process_file, config.cut_seq_to_fna.data_path, config.cut_seq_to_fna.output_path, **config.cut_seq_to_fna))
    for future in tqdm.tqdm(as_completed(futures), total=len(futures)):
        future.result()

    # print(f'Cut sequences to fna saved to {config.cut_seq_to_fna.output_path}')


@hydra.main(config_path="configs", config_name="config.yaml", version_base=None)
def main(config: OmegaConf):
    # check if the config is valid
    config = process_config(config)
    
    if config.cut_seq_to_fna.data_path is None:
        raise ValueError("The 'input_path' parameter in config must be specified.")
    if config.cut_seq_to_fna.output_dir is None:
        raise ValueError("The 'output_path' parameter in config must be specified.")

    # if finished, skip
    if os.path.exists(config.cut_seq_to_fna.output_path) and len(os.listdir(config.cut_seq_to_fna.output_path)) > 1:
        # logger.info("The 'cut_seq_to_fna' process has already been completed, skipping...")
        return
    
    os.makedirs(config.cut_seq_to_fna.output_path, exist_ok=True)
    # print_config(config, resolve=True, save_dir=os.path.join(config.cut_seq_to_fna.output_path, 'logs'), prefix="cut_seq_to_fna")

    # init logger
    init_logger(svr_name="cut_seq_to_fna", log_path=os.path.join(config.cut_seq_to_fna.output_path, 'logs'))
    # logger.info("start cut seq to fna...")
    cut_seq_to_fna(config)
    # logger.info("cut seq to fna finished!")


if __name__ == '__main__':
    main()