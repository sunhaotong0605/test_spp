import pickle

import numpy as np
from torch.utils.data import Dataset
import torch

from src.self_logger import logger

class BaseDataset(Dataset):

    def __init__(self, data_path):
        logger.info(f"loading data from {data_path}")
        self.data_path = data_path
        self.datas, self.labels ,self.names= [], [],[]
        self._read_data()
        self.len = len(self.datas)


    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        return_list=[torch.tensor(self.datas[idx])]
        try:
            label=self.labels[idx] if self.labels is not None else None
        except:
            label=self.labels if self.labels is not None else None
        return_list.append(label)
        if self.names:
            return_list.append(self.names[idx])
        return return_list

    def _read_data(self):
        raise NotImplementedError


class ProbioticDataset(BaseDataset):
    def __init__(
            self, 
            data_path,
            return_name=False, 
            **kwargs
            ):
        self.return_name=return_name
        super().__init__(data_path)

    def _read_data(self):
        datas, labels, names = [], [],[]
        with open(self.data_path, 'rb') as file:
            pkl_data = pickle.load(file)

        try:
            names = [x.split("/")[-1].rstrip(".pkl") for x in pkl_data['seqs_paths']]
        except:
            names = [x.split("/")[-1].rstrip(".pkl") for x in pkl_data['seqs_paths'][0]]

        labels = pkl_data['seqs_labels']
        datas = pkl_data['model_predict']['embedding']

        if self.return_name:
            self.datas, self.labels,self.names= datas, labels,names
        else:
            self.datas, self.labels = datas, labels

        
class ProbioticKFTrainDataset(Dataset):
    def __init__(
            self, 
            seq_paths, 
            max_length=1000,
            return_name = False,
            split_num = -1,
            seed=42,
            **kwargs):
        self.max_length = max_length
        self.return_name = return_name
        self.length_list = []
        self.seq_paths = seq_paths
        self.seed = seed
        self.split_num = split_num
        super().__init__()

    def __len__(self):
        return len(self.seq_paths)

    def __getitem__(self, idx):
        with open(self.seq_paths[idx].strip(), "rb") as f:
            data = pickle.load(f)
        embedding = data["model_predict"]["embedding"]
 
        if self.split_num != -1:
            # if is a split number ablation experiment
            # randomly select self.split_num elements from embedding, do this 10 times
            temp_array = np.array([])
            embedding = np.array(embedding, dtype=np.float32)
            for i in range(10):
                indices = np.random.choice(embedding.shape[0], self.split_num, replace=False)
                temp = embedding[indices]
                if i == 0:
                    temp_array = temp
                else:
                    temp_array = np.concatenate((temp_array, temp), axis=0)
            embedding = temp_array
        else:
            # if not a split number ablation experiment
            # randomly select self.max_length elements from embedding
            if len(embedding) > self.max_length:
                embedding = np.array(embedding, dtype=np.float32)
                indices = np.random.choice(embedding.shape[0], self.max_length, replace=False)
                embedding = embedding[indices]
                self.length_list.append(self.max_length)

        self.embedding = embedding
        self.labels = data["seqs_labels"] if isinstance(data["seqs_labels"], (int, np.integer)) else data["seqs_labels"][0]

        if self.return_name:
            self.names = data['sample_name']
            return [self.embedding,self.labels,self.names]
        else:
            return [self.embedding,self.labels]
 
class ProbioticKFEvalDataset(Dataset):
    def __init__(
            self,  
            seq_paths, 
            max_length=1000,
            return_name = False,
            split_num = -1,
            seed=42,
            **kwargs):
        self.max_length = max_length
        self.return_name = return_name
        self.length_list = []
        self.seq_paths = seq_paths
        self.split_num = split_num
        np.random.seed(seed)
        super().__init__()

    def __len__(self):
        return len(self.seq_paths)

    def __getitem__(self, idx):
        with open(self.seq_paths[idx].strip(), "rb") as f:
            data = pickle.load(f)
        embedding = data["model_predict"]["embedding"]

        split_num = 100 if self.split_num == -1 else self.split_num
        # random select 100 elements from embedding, do this 10 times
        temp_array = np.array([])
        embedding = np.array(embedding, dtype=np.float32)
        for i in range(10):
            indices = np.random.choice(embedding.shape[0], split_num, replace=False)
            temp = embedding[indices]
            if i == 0:
                temp_array = temp
            else:
                temp_array = np.concatenate((temp_array, temp), axis=0)
        embedding = temp_array

        self.embedding = embedding
        self.labels = data["seqs_labels"] if isinstance(data["seqs_labels"], (int, np.integer)) else data["seqs_labels"][0]

        if self.return_name:
            self.names = data['sample_name']
            return [self.embedding,self.labels,self.names]
        else:
            return [self.embedding,self.labels]



class ProbioticTestDataset(Dataset):
    def __init__(
            self, 
            data_path, 
            max_length=1000,
            split_num = -1,
            return_name = False,
            **kwargs):
        self.max_length = max_length
        self.return_name = return_name
        self.length_list = []
        self.split_num = split_num
        super().__init__()
        with open(data_path, 'r') as file:
            self.sequences = [x.rstrip() for x in file.readlines()]

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        with open(self.sequences[idx].strip(), "rb") as f:
            data = pickle.load(f)
        embedding = data["model_predict"]["embedding"]

        split_num = 100 if self.split_num == -1 else self.split_num
        # randomly select 100 elements from embedding, do this 10 times
        temp_array = np.array([])
        embedding = np.array(embedding, dtype=np.float32)
        for i in range(10):
            indices = np.random.choice(embedding.shape[0], split_num, replace=False)
            temp = embedding[indices]
            if i == 0:
                temp_array = temp
            else:
                temp_array = np.concatenate((temp_array, temp), axis=0)
        embedding = temp_array

        self.embedding = embedding
        self.labels = data["seqs_labels"] if isinstance(data["seqs_labels"], (int, np.integer)) else data["seqs_labels"][0]
        
        if self.return_name:
            self.names = data['sample_name']
            return [self.embedding,self.labels,self.names]
        else:
            return [self.embedding,self.labels]
        