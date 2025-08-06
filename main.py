import os
import sys

import hydra
from omegaconf import DictConfig, OmegaConf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.data_process.cut_seq import cut_seq
from src.data_process.extract_feature import cut_seq_to_fna
from src.data_process.extract_feature import extract_ef
from src.data_process.extract_feature import extract_llmr
from src.data_process.enhance_rep import enhance_rep
from src.data_process.ml_predict import ml_predict

@hydra.main(config_path="configs", config_name="config.yaml", version_base=None)
def main(config: OmegaConf):
    if config.run_cut_seq:
        cut_seq.main(config)
    if config.run_cut_seq_to_fna:
        cut_seq_to_fna.main(config)
    if config.run_extract_ef:
        extract_ef.main(config)
    if config.run_extract_llmr:
        extract_llmr.main(config)
    if config.run_enhance_rep:
        enhance_rep.main(config)
    if config.run_ml_predict:
        ml_predict.main(config)
 
if __name__ == '__main__':
    main()