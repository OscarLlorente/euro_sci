# machine learning libraries
import torch
import numpy as np
import pandas as pd
from transformers import AutoTokenizer
from torch.utils.data import Dataset, DataLoader
from sklearn.feature_extraction.text import TfidfVectorizer

# other libraries
import os
import pickle
import random
from abc import ABC, abstractmethod
from typing import Literal


class BaseDataset(Dataset, ABC):
    
    def __init__(self, df_path: str) -> None:
        
            
        df = pd.read_csv(df_path, usecols=['Title', 'Review Text', 'Rating'], 
                            dtype={'Title': 'string', 'Review Text': 'string', 'Rating': int})
        df = df.dropna().reset_index()
        self.df = df

    def __len__(self) -> int:
        return self.df.shape[0]

    @abstractmethod
    def __getitem__(self, index: int) -> any:
        pass
    
class TfIdfDataset(BaseDataset):
    
    def __init__(self, df_path: str) -> None:
        
        super().__init__(df_path)
        
        self.texts = [preprocess(self.df['Title'][index] + self.df['Review Text'][index]) for index in range(len(self))]
        tfidf = TfidfVectorizer(max_features=5000)
        self.x = (tfidf.fit_transform(self.texts)).toarray()
        
    # overriding method
    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        tf_idf_vector = self.x[index, :]
        label = self.df['Rating'][index] - 1
        return tf_idf_vector, label


class BertDataset(BaseDataset):

    def __init__(self, df_path: str) -> None:
            
        super().__init__(df_path)
        
        df = pd.read_csv(df_path, usecols=['Title', 'Review Text', 'Rating'], 
                            dtype={'Title': 'string', 'Review Text': 'string', 'Rating': int})
        df = df.dropna().reset_index()
        self.df = df

        tokenizer = AutoTokenizer.from_pretrained('activebus/BERT-XD_Review')

        self.texts = [
            tokenizer(self.df['Title'][index] + self.df['Review Text'][index], padding='max_length', 
                        max_length = 512, 
                        truncation=True,return_tensors="pt") for index in range(len(self))]

    # overriding method
    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, int]:
        
        input_ids = self.texts[index]['input_ids'][0]
        attention_mask = self.texts[index]['attention_mask'][0]
        label = self.df['Rating'][index] - 1
        return input_ids, attention_mask, label


def load_base_dataset(
    df_path: str,
    save_path: str, 
    dataset_type: Literal['tfidf', 'bert'],
    split_sizes: tuple[float, float, float],
    batch_size: int = 200,
    shuffle: bool = True
) -> tuple[DataLoader, DataLoader, DataLoader]:
    
    # load dataset if save path exists
    if os.path.exists(save_path):
        full_dataset = pickle.load(open({save_path}, 'rb'))
    
    else:
        # define the full dataset
        if dataset_type == 'tfidf':
            full_dataset = TfIdfDataset(df_path)
        
        elif dataset_type == 'bert':
            full_dataset = BertDataset(df_path)
            
        torch.save(full_dataset, save_path)
    
    # split dataset into train, val and test
    train_size = int(split_sizes[0] * len(full_dataset))
    val_size = int(split_sizes[1] * len(full_dataset))
    test_size = len(full_dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = \
        torch.utils.data.random_split(full_dataset, [train_size, val_size, test_size])
    
    # define dataloaders
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=shuffle)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle)
    
    return train_dataloader, val_dataloader, test_dataloader


def preprocess(text: str) -> list[str]:
    pass


def accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """
    This method computes accuracy from logits and labels
    
    Parameters
    ----------
    logits : torch.Tensor
        batch of logits. Dimensions: [batch, number of classes]
    labels : torch.Tensor
        batch of labels. Dimensions: [batch]
        
    Returns
    -------
    float
        accuracy of predictions
    """

    # compute predictions
    predictions = logits.argmax(1).type_as(labels)

    # compute accuracy from predictions
    result = float(predictions.eq(labels).float().mean().cpu().detach().numpy())

    return result


def set_seed(seed: int) -> None:
    """
    This function sets a seed and ensure a deterministic behavior
    Parameters
    ----------
    seed : int
    Returns
    -------
    None
    """

    # set seed in numpy and random
    np.random.seed(seed)
    random.seed(seed)

    # set seed and deterministic algorithms for torch
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)

    # Ensure all operations are deterministic on GPU
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.determinstic = True
    torch.backends.cudnn.benchmark = False

    # for deterministic behavior on cuda >= 10.2
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    return None
