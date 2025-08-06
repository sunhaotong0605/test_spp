import glob
import os.path
import pickle
import random
from multiprocessing import Pool
from typing import List, Optional, Dict, Any, Tuple

import numpy as np
import pandas as pd
import torch
import tqdm
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase, DataCollatorWithPadding
from Bio import SeqIO
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import KernelPCA

from src.self_logger import logger

class ProbioticDataset(Dataset):
    def __init__(
            self,
            dest_path: str,
            tokenizer: PreTrainedTokenizerBase,
            dataset_name: str = "FB",
            split: str = "train",
            max_length: int = 512,
            seed: int = 42,
            **kwargs
    ):
        super(ProbioticDataset, self).__init__()
        np.random.seed(seed)
        random.seed(seed)
        dataset_name = str(dataset_name)
        
        data_path = os.path.join(dest_path, dataset_name, split + ".txt")
        assert os.path.exists(data_path), f"Data path {data_path} does not exist."

        with open(data_path, "r") as f:
            data = f.read().split("\n")
        if data[-1] == "":
            data = data[:-1]

        # sample.txt
        self.data_path = data_path
        # pkl path list
        self.sequences = data
        self.max_length = max_length
        self.labels = None
        self.tokenizer: PreTrainedTokenizerBase = tokenizer
        self._data_collator = DataCollatorWithPadding(self.tokenizer)

    def __len__(self): 
        return len(self.sequences)

    def __getitem__(self, idx):
        with open(self.sequences[idx], "rb") as f:
            data = pickle.load(f)
        try:
            seq = data["Seq"][:self.max_length]
            label = data["Label"]
        except:
            seq = data["seq"][:self.max_length]
            label = data["label"]
        return {    
            'labels': label,
            "input_ids": self.tokenizer(seq)["input_ids"],
        }

    def data_collator(self, features: List[Dict[str, Any]]):
        input_ids, batch_label, attention_mask = [], [], []
        max_length = max([len(s['input_ids']) for s in features])

        for sample in features:
            input_id = np.zeros(max_length, dtype=np.float32)
            sample_id = sample['input_ids']
            mask = [1] * len(sample_id) + [0] * (max_length - len(sample_id))
            attention_mask.append(mask)
            input_id[:len(sample_id)] = sample_id
            input_ids.append(input_id)
            batch_label.append(sample['labels'])
        input_ids, batch_label, attention_mask = np.array(input_ids), np.array(batch_label), np.array(attention_mask)
        input_ids = torch.tensor(input_ids).long()
        batch_label = torch.tensor(batch_label).long()
        attention_mask = torch.tensor(attention_mask).long()

        return {"input_ids": input_ids, "labels": batch_label, "attention_mask": attention_mask}

class ProbioticEnhanceRepresentationDataset(Dataset):
    def __init__(
            self,
            dest_path: str,
            _dest_path: str,
            dataset_name: str = "",
            split: str = "train",
            seed: int = 42,
            only_features: bool = False,
            **kwargs
    ):
        super(ProbioticEnhanceRepresentationDataset, self).__init__()
        np.random.seed(seed)
        random.seed(seed)
        split = str(split)

        embedding_txt_path = os.path.join(dest_path, dataset_name, split + ".txt")
        assert os.path.exists(embedding_txt_path), f"Data path {embedding_txt_path} does not exist."
        with open(embedding_txt_path, "r") as f:
            embedding_paths = f.read().split("\n")
        if embedding_paths[-1] == "":
            embedding_paths = embedding_paths[:-1]

        if os.path.exists(os.path.join(_dest_path, dataset_name, split + ".txt")) or os.path.exists(os.path.join(_dest_path, dataset_name, "all.txt")):
            if os.path.exists(os.path.join(_dest_path, dataset_name, split + ".txt")):
                manual_feature_txt_path = os.path.join(_dest_path, dataset_name, split + ".txt")
            else:
                manual_feature_txt_path = os.path.join(_dest_path, dataset_name, "all.txt")
            with open(manual_feature_txt_path, "r") as f:
                manual_feature_paths = f.read().split("\n")

        self.embedding_paths = embedding_paths
        self.manual_feature_paths = manual_feature_paths
        self.dest_path = dest_path
        self._dest_path = _dest_path
        self.only_features = only_features

    def __len__(self):
        return len(self.embedding_paths)

    def __getitem__(self, idx):
        with open(self.embedding_paths[idx], "rb") as f:
            embedding_dict = pickle.load(f)

        for path in self.manual_feature_paths:
            if path.endswith(self.embedding_paths[idx].split("/")[-1]):
                with open(path, "rb") as f:
                    manual_feature_dict = pickle.load(f)
                    break

        sample_name = embedding_dict['sample_name']
        seqs_paths = embedding_dict['seqs_paths']
        seqs_labels = embedding_dict['seqs_labels'][0] if isinstance(embedding_dict['seqs_labels'], list) else embedding_dict['seqs_labels']
        embedding = embedding_dict['model_predict']['embedding']
        manual_feature = []
        # Some orf fragments are filtered, so we need to align them here
        for path in seqs_paths:
            try:
                index = manual_feature_dict['seqs_paths'].index(path)
                manual_feature.append(manual_feature_dict['model_predict']['embedding'][index])
            except:
                pass
        if len(manual_feature) != len(embedding):
            raise ValueError(f"manual_feature and embedding length not equal, manual_feature: {len(manual_feature)}, embedding: {len(embedding)}")
        if self.only_features:
            return {
                "seqs_labels": seqs_labels,
                'manual_feature': manual_feature,
                "embedding": embedding,
            }
        return {
            "sample_name": sample_name,
            'seqs_paths': seqs_paths,
            "seqs_labels": seqs_labels,
            'manual_feature': manual_feature,
            "embedding": embedding,
        }
    def data_collator(self, features: List[Dict[str, Any]]):
        # if there are other custom inputs besides input_ids, token_type_ids, attention_mask, labels, you need to customize data_collator
        max_length = max([len(s['embedding']) for s in features])
        feature_dim = len(features[0]['embedding'][1])

        batch_label, manual_feature, embedding = [], [], []

        for sample in features:
            # manual_feature
            sample_manual_feature = sample['manual_feature']
            max_manual_feature = np.zeros((max_length, feature_dim), dtype=np.float32)
            max_manual_feature[:len(sample_manual_feature), :] = sample_manual_feature
            manual_feature.append(max_manual_feature)

            # embedding
            sample_embedding = sample['embedding']
            max_embedding = np.zeros((max_length, feature_dim), dtype=np.float32)
            max_embedding[:len(sample_embedding), :] = sample_embedding
            embedding.append(max_embedding)

            # labels
            batch_label.append(sample['seqs_labels'])


        batch_label, manual_feature, embedding = np.array(batch_label), np.array(manual_feature), np.array(embedding)
        batch_label = torch.tensor(batch_label).long()
        manual_feature = torch.tensor(manual_feature)
        embedding = torch.tensor(embedding)

        return {"seqs_labels": batch_label, "manual_feature": manual_feature, "embedding": embedding}

            

class ProbioticSplitEnhanceRepresentationDataset(Dataset):
    def __init__(
            self,
            seqs_labels: int,
            manual_feature,
            embedding,
            seed: int = 42,
            **kwargs
    ):
        super(ProbioticSplitEnhanceRepresentationDataset, self).__init__()
        np.random.seed(seed)
        random.seed(seed)   

        self.seqs_labels = seqs_labels
        self.manual_feature = manual_feature
        self.embedding = embedding
        self.pca = KernelPCA(n_components=132, kernel='rbf')
 
        # LLM representation and engineering feature global normalization
        embed_min = np.min(self.embedding)
        embed_max = np.max(self.embedding)
        self.embedding = (self.embedding - embed_min) / (embed_max - embed_min)
        mf_min = np.min(self.manual_feature)
        mf_max = np.max(self.manual_feature)
        self.manual_feature = (self.manual_feature - mf_min) / (mf_max - mf_min)
        
        # LLM representation dimensionality reduction
        self.embedding = self.pca.fit_transform(self.embedding)

        # LLM representation global normalization
        embed_min = np.min(self.embedding)
        embed_max = np.max(self.embedding)
        self.embedding = (self.embedding - embed_min) / (embed_max - embed_min)

    def __len__(self):
        return len(self.embedding)

    def __getitem__(self, idx):
        # transform to tensor and add a dimension at 0
        embedding = torch.tensor(self.embedding[idx]).unsqueeze(0)
        manual_feature = torch.tensor(self.manual_feature[idx]).unsqueeze(0)

        return {    
            'labels': self.seqs_labels[0],
            "embedding": embedding,
            "manual_feature": manual_feature,
        }

    
class ProbioticsDataProcess:
    def __init__(self, seed=42):
        self.seed = seed
        self._set_random_seed(self.seed)

    @staticmethod
    def _set_random_seed(seed):
        random.seed(seed)
        np.random.seed(seed)

    def probiotics_check_sample_len(
            self,
            data_path: str,
            min_len: int = 100,
            max_len: int = 2500
    ):
        short_genomes = []
        for file_name in tqdm.tqdm(os.listdir(data_path)):
            if file_name.endswith('.fna') or file_name.endswith('.fasta'):
                file_path = os.path.join(data_path, file_name)
                total_length = 0
                for record in SeqIO.parse(file_path, "fasta"):
                    total_length += len(record.seq)
                if total_length < min_len or total_length > max_len:
                    short_genomes.append(file_name)
        return short_genomes


    def probiotics_single_sample_data(
            self,
            dest_path: str,
            save_path: str,
            sample_num: int = -1,
            name: str = "*",
            num_bound: Optional[Tuple] = (3000, 7000),
            len_bound: Optional[Tuple] = (100, 2500),
            seed: int = 42,
            short_genomes: List[str] = []

    ):

        random.seed(seed)
        assert os.path.exists(dest_path), f"{dest_path} is not exits"
        os.makedirs(save_path, exist_ok=True)

        file_dirs = glob.glob(os.path.join(dest_path, name))
        if sample_num != -1:
            random.shuffle(file_dirs)
            count = 0
        for file_dir in tqdm.tqdm(file_dirs):
            if os.path.basename(file_dir) in short_genomes:
                continue
            pkl_path = glob.glob(os.path.join(f"{file_dir}", "*.pkl"))
            # filter sample by orf nums
            if num_bound is not None:
                if len(pkl_path) < num_bound[0] or len(pkl_path) > num_bound[1]:
                    continue
            if len_bound is not None:
                pool = Pool(10)  # use multi-process to speed up
                valid_pkl = []
                for pkl in pkl_path:
                    valid_pkl.append(pool.apply_async(self.filter_pkl, args=(pkl, len_bound[0], len_bound[1],)))
                pool.close()
                pool.join()
                pkl_path = [pkl.get() for pkl in valid_pkl if pkl.get() is not None]
            else:
                pkl_path = pkl_path
            if sample_num != -1:
                count = count + 1
                if count == sample_num + 1:
                    break
            save_name = os.path.basename(file_dir).rstrip("/") + ".txt"
            with open(os.path.join(save_path, save_name), "w") as f:
                for pkl in pkl_path:
                    f.write(pkl + "\n")

    def probiotics_get_pickles_txt(
            self,
            dir: str,
    ):
        with open(dir+'/pickles.txt', "w") as f:
            for path in glob.glob(f"{dir}/pickles/*.pkl"):
                f.write(path + "\n")
        # logger.info(f"All results saved in {dir+'/pickles.txt'}")
    
    def probiotics_get_all_txt(
            self,
            pkl_path: str,
    ):
        with open(pkl_path + '/all.txt', "w") as f:
            for path in glob.glob(f"{pkl_path}/*.pkl"):
                f.write(path + "\n")
        # logger.info(f"All results saved in {pkl_path}/all.txt")


if __name__ == '__main__':
    pbt_dp = ProbioticsDataProcess()
