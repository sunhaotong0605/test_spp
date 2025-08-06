#!/usr/bin/env python
import os
import pickle
import tqdm
from itertools import product

from Bio import SeqIO
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed

import hydra
from omegaconf import DictConfig, OmegaConf
from Bio import SeqIO

from src.utils.train import print_config, process_config
from src.self_logger import logger, init_logger
from src.datasets.probiotics_dataset import ProbioticsDataProcess

def generate_all_kmers(k):
    nucleotides = ["A", "G", "C", "T"]
    return ["".join(elt) for elt in product(nucleotides, repeat=k)]

def generate_kmer_frequencies(dna_sequence, k):
    kmers = generate_all_kmers(k)
    kmers_dict = {i: 0 for i in kmers}
    dna_sequence = dna_sequence.split("N")
    for dna_sequence_split in dna_sequence:
        for i in range(len(dna_sequence_split) - k + 1):
            kmers_dict[dna_sequence_split[i:i+k]] += 1
    return kmers_dict

def normalize_kmer_frequencies(dna_sequence, k):
    kmers_freq = []
    for j in range(k):
        kmers_dict = generate_kmer_frequencies(dna_sequence, j+1)
        _kmers_freq = list(kmers_dict.values())
        if sum(_kmers_freq) != 0:
            _kmers_freq = [i/sum(_kmers_freq) for i in _kmers_freq]
        kmers_freq.extend(_kmers_freq)
    return kmers_freq

def generate_all_gaps(g):
    nucleotides = ["A", "G", "C", "T"]
    return [nucleotides[i]+"."*g+nucleotides[j] for i in range(len(nucleotides)) for j in range(len(nucleotides))]

def generate_ggap_frequencies(dna_sequence, g):
    ggaps = generate_all_gaps(g)
    ggaps_dict = {i: 0 for i in ggaps}
    dna_sequence = dna_sequence.split("N")
    for dna_sequence_split in dna_sequence:
        for i in range(len(dna_sequence_split) - g - 1):
            ggaps_dict[dna_sequence_split[i]+"."*g+dna_sequence_split[i+g]] += 1
    return ggaps_dict

def normalize_ggap_frequencies(dna_sequence, g):
    ggaps_freq = []
    for j in range(g):
        ggaps_dict = generate_ggap_frequencies(dna_sequence, j+1)
        _ggaps_freq = list(ggaps_dict.values())
        if sum(_ggaps_freq) != 0:
            _ggaps_freq = [i/sum(_ggaps_freq) for i in _ggaps_freq]
        ggaps_freq.extend(_ggaps_freq)
    return ggaps_freq

def process_file(file_path, config, k):
    file_name = os.path.basename(file_path)
    suffix = '.fna'
    ORF_dict = {}
    ORF_dict['model_predict'] = {}
    ORF_dict['model_predict']['embedding'] = []
    seqs_num = 0

    ORF_dict['sample_name'] = file_name.replace(suffix, '')
    ORF_dict['seqs_paths'] = []
    for record in SeqIO.parse(file_path, "fasta"):
        ORF_dict['seqs_paths'].append(os.path.join(os.path.normpath(config.cut_seq.output_path), file_name.replace(suffix, ''), f"{record.id}.pkl"))
        ef_list = normalize_kmer_frequencies(str(record.seq), k)
        ef_list.extend(normalize_ggap_frequencies(str(record.seq), k))
        ORF_dict['model_predict']['embedding'].append(ef_list)
        seqs_num += 1
    # set seqs_labels to 1
    ORF_dict['seqs_labels'] = np.array([1]*seqs_num)
    output_file_path = os.path.join(config.extract_ef.output_path, file_name.replace(suffix, '.pkl'))
    with open(output_file_path, 'wb') as file:
        pickle.dump(ORF_dict, file)
    return file_name

def extract_ef(config: DictConfig):
    for k in config.extract_ef.k:
        file_paths = [os.path.join(config.extract_ef.data_path, file_name) for file_name in os.listdir(config.extract_ef.data_path) if file_name.endswith('.fna')]
        with ThreadPoolExecutor() as executor:
            futures = {executor.submit(process_file, file_path, config, k): file_path for file_path in file_paths}
            for future in tqdm.tqdm(as_completed(futures), total=len(futures)):
                try:
                    file_name = future.result()
                    # logger.info(f"Processed {file_name} with k={k}")
                except Exception as e:
                    logger.error(f"Error processing file: {futures[future]} - {e}")
        # logger.info(f'Extracted features for {len(file_paths)} files with k={k} and saved to {config.extract_ef.output_path}')

@hydra.main(config_path="configs", config_name="config.yaml", version_base=None)
def main(config: OmegaConf):
    # check if the config is valid
    config = process_config(config)
    
    if config.extract_ef.output_dir is None:
        raise ValueError("The 'output_path' parameter in config must be specified.")

    # if finished, skip
    if os.path.exists(config.extract_ef.output_path) and len(os.listdir(config.extract_ef.output_path)) > 1:
        logger.info("The 'engineered feature extraction' process has already been completed, skipping...")
        return
    
    os.makedirs(config.extract_ef.output_path, exist_ok=True)
    # print_config(config, resolve=True, save_dir=os.path.join(config.extract_ef.output_path, 'logs'), prefix="extract_ef")

    # init logger
    init_logger(svr_name="extract_ef", log_path=os.path.join(config.extract_ef.output_path, 'logs'))
    # logger.info("start extract_ef...")
    extract_ef(config)


    pbt_dp = ProbioticsDataProcess()
    pbt_dp.probiotics_get_all_txt(pkl_path=config.extract_ef.output_path)
    logger.info("Engineered feature extraction completed")

if __name__ == '__main__':
    main()