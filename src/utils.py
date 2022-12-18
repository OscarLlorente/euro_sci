# deep learning libraries
import torch
import numpy as np
import pandas as pd
from transformers import BertTokenizer
from torch.utils.data import Dataset, DataLoader

# other libraries
import os
import pickle
import random


class BaseDataset(Dataset):

    def __init__(self, save_path: str, df_path: str) -> None:
        
        if not os.path.exists(save_path):
            # codes = pd.read_excel(codes_path, usecols=['code'], dtype={'code': 'string'})
            # codes = codes.dropna().reset_index()
            # codes = codes['code'].unique().tolist()
            # codes = dict(zip(codes, range(len(codes))))
            # self.codes = codes

            # projects = pd.read_excel(projects_path, usecols=['title', 'summary', 'euroSciVocCode'], 
            #                         dtype={'title': 'string', 'summary': 'string','euroSciVocCode': 'string'})
            # projects = projects.dropna().reset_index()
            # projects['euroSciVocCode'] = projects['euroSciVocCode'].apply(lambda code: code[1:-1].split(',')[0])
            # self.projects = projects
            
            df = pd.read_csv(df_path, usecols=['Title', 'Review Text', 'Rating'], 
                             dtype={'Title': 'string', 'Review Text': 'string','Rating': int})
            df = df.dropna().reset_index()
            self.df = df

            tokenizer = BertTokenizer.from_pretrained('bert-base-cased')

            self.texts = [
                tokenizer(self.df['Title'][index] + self.df['Review Text'][index], padding='max_length', 
                          max_length = 512, 
                          truncation=True,return_tensors="pt") for index in range(len(self))]
            
            os.makedirs(save_path)
            pickle.dump(self.texts, open(f'{save_path}/texts', 'wb'))
            pickle.dump(self.df, open(f'{save_path}/df', 'wb'))
            # pickle.dump(self.projects, open(f'{save_path}/projects', 'wb'))
            
        else:
            self.texts = pickle.load(open(f'{save_path}/texts', 'rb'))
            self.df = pickle.load(open(f'{save_path}/df', 'rb'))
            # self.projects = pickle.load(open(f'{save_path}/projects', 'rb'))

    def __len__(self) -> int:
        return self.df.shape[0]

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, int]:
        
        input_ids = self.texts[index]['input_ids'][0]
        attention_mask = self.texts[index]['attention_mask'][0]
        label = self.df['Rating'][index] - 1
        return input_ids, attention_mask, label

def load_base_dataset(
    save_path: str,
    projects_path: str, 
    codes_path: str, 
    split_sizes: tuple[float, float, float],
    batch_size: int = 200,
    shuffle: bool = True
) -> tuple[DataLoader, DataLoader, DataLoader]:
    
    # define the full dataset
    full_dataset = BaseDataset(save_path, projects_path)
    
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
