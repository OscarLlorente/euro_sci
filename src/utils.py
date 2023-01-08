#==============================================================#
#   Project - Natural Language Processing                      #
#   Author: María Sauras & Oscar Llorente                      #
#   Date: 19/12/2022                                           #                                    
#==============================================================#

# machine learning and NLP libraries
import torch
import torchtext
import numpy as np
import pandas as pd
from transformers import AutoTokenizer
from torch.utils.data import Dataset, DataLoader
from scipy.sparse import csr_matrix
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn import svm
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import balanced_accuracy_score, classification_report

# other libraries
import os
import random
import re
from typing import List, Tuple, Union, Literal

    
def extract_data(
    projects_path: str, 
    codes_path: str, 
    level: int
) -> Tuple[List[str], np.ndarray, np.ndarray, np.ndarray]:
    """
    This function extract the data from xlsx files

    Parameters
    ----------
    projects_path : str
        path of the projects.xlsx file
    codes_path : str
        path of the SciVocCodes.xlsx file
    level : int
        level for classification

    Returns
    -------
    List[str]
        corpus of texts
    np.ndarray
        labels. Dimensions: [number of documents]
    np.ndarray
        costs of the projects. Dimensions: [number of documents]
    np.ndarray
        contributions of the EU in the projects. Dimensions: [number of documents]
    """
    
    # extract codes dictionary
    codes = pd.read_excel(codes_path, usecols=['code', 'full_code'], dtype={'code': 'string', 'full_code': 'string'})
    codes = codes.dropna().reset_index()
    codes['full_code'] = codes['full_code'].apply(
        lambda x: x[1:].split('/')[level] if len(x[1:].split('/')) > level else np.nan)
    codes = codes.dropna().reset_index()
    full_codes = codes['full_code'].tolist()
    codes = codes['code'].tolist()
    codes = dict(zip(codes, full_codes))
    
    # create variable to get correct classes
    unique_codes = np.sort(np.unique(full_codes).astype(int)).astype(str)
    classes_dict = dict(zip(unique_codes, np.array(list(range(len(unique_codes)))).astype(str)))
    
    # extract projects text, variables and codes
    projects = pd.read_excel(projects_path, usecols=['title', 'summary', 'euroSciVocCode', 'totalCost', 
                                                     'ecMaxContribution'], 
                                        dtype={'title': 'string', 'summary': 'string','euroSciVocCode': 'string', 
                                               'totalCost': float, 'ecMaxContribution': float})
    projects = projects.dropna().reset_index()
    # projects['euroSciVocCode'] = \
    #     projects['euroSciVocCode'].apply(lambda code: classes_dict[codes[code[1:-1].split(',')[0]]])
    projects['euroSciVocCode'] = projects['euroSciVocCode'].apply(
        lambda code: 
            classes_dict[codes[code[1:-1].split(',')[0]]] if code[1:-1].split(',')[0] in set(codes.keys()) else np.nan)
    projects = projects.dropna().reset_index()
    
    # concat title and summary
    projects['summary'] = projects['title'] + projects['summary']
    
    # get final variables to return them
    corpus = projects['summary']
    labels = projects['euroSciVocCode'].to_numpy()
    costs = projects['totalCost'].to_numpy()
    contributions = projects['ecMaxContribution'].to_numpy()

    return corpus, labels, costs, contributions
    
    
def preprocessing(texts: List[str]) -> List[str]:
    """
    This function computes a preprocessing pipeline

    Parameters
    ----------
    texts : List[str]
        list of documents

    Returns
    -------
    List[str]
        list of preprocessed documents
    """
    
    # set needed objects
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    
    # tokenize
    texts = [word_tokenize(text) for text in texts]
    
    # eliminate non char tokens and remove stop-words
    texts = [re.sub("[^A-Za-z']+", ' ', ' '.join(text)).lower() for text in texts]
    texts = [[word for word in text.split(' ') if word not in stop_words] for text in texts]
    
    # lemmantize and return list of strings
    texts = [[lemmatizer.lemmatize(word) for word in text] for text in texts]
    texts = [' '.join(text) for text in texts]
    
    return texts


class BasicModel:
    """
    This class is a wrapper for the classification model
    
    Attributes
    ----------
    self.x_train : np.ndarray
        texts for training. Dimensions: [number of documents, vectors size]
    self.x_test : np.ndarray
        texts for testing. Dimensions: [number of documents, vectors size]
    self.y_train : np.ndarray
        labels for training. Dimensions: [number of documents]
    self.y_test : np.ndarray
        labels for testing. Dimensions: [number of documents]
    self.y_pred : np.ndarray
        predicted labels. Dimensions: [number of documents]
    
    Methods
    -------
    accuracy -> float
    classification_report -> float
    """
    
    def __init__(self, corpus: Union[np.ndarray, csr_matrix], labels: np.ndarray, test_size: float = 0.1) -> None:
        """
        This method is the contructor for BasicModel class

        Parameters
        ----------
        corpus : Union[np.ndarray, csr_matrix]
            corpus of the texts. Dimensions: [number of documents, vectors size]
        labels : np.ndarray
            array with the labels. Dimensions: [number of documents]
        """
        
        self.x_train, self.x_test, self.y_train, self.y_test = \
            train_test_split(corpus, labels, test_size=test_size, random_state=42)
            
        # train model
        grid_param = {'kernel':('linear', 'rbf'), 'C':[0.1 ,1, 10]}
        svc = svm.SVC(class_weight='balanced')
        model_grid = GridSearchCV(svc, grid_param, refit=True, cv = 2, n_jobs=-1)
        model_grid.fit(self.x_train, self.y_train)
        
        # get predictions
        self.y_pred = model_grid.predict(self.x_test)
    
    @property
    def accuracy(self) -> float:
        """
        This method computes the balanced accuracy of the model

        Returns
        -------
        float
            balanced accuracy
        """
        
        return balanced_accuracy_score(self.y_test, self.y_pred)
    
    @property
    def classification_report(self) -> str:
        """
        This method computes the classification report of the model

        Returns
        -------
        str
            The classification report of the model
        """
        
        return classification_report(self.y_test, self.y_pred)
    

class BertDataset(Dataset):

    def __init__(self, projects_path: str, codes_path: str) -> None:
        
        # extract texts and labels
        self.texts, self.labels, _, _ = extract_data(projects_path, codes_path)

        # define tokenizer
        tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

        # get tensors with tokenizer
        self.texts = [tokenizer(text, padding='max_length',  max_length = 200, truncation=True, 
                                return_tensors="pt") for text in self.texts]
        
    def __len__(self) -> int:
        return len(self.texts)

    # overriding method
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        
        # get data from index
        input_ids = self.texts[index]['input_ids'][0]
        attention_mask = self.texts[index]['attention_mask'][0]
        label = int(self.labels[index])
        
        return input_ids, attention_mask, label
    
    
class EmebeddingsDataset(Dataset):
    
    def __init__(self, projects_path: str, codes_path: str) -> None:
        
        self.texts, self.labels, _, _ = extract_data(projects_path, codes_path)
        self.glove = torchtext.vocab.GloVe(name='6B', dim=300)
        
    def __len__(self) -> int:
        return len(self.texts)
    
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int, int]:
        
        if self.texts[index] == None:
            embeddings = torch.zeros(200, 300)
        else:
            embeddings = self.glove.get_vecs_by_tokens(self.texts[index].split(' '), lower_case_backup=True)
        if embeddings.shape[0] > 200:
            embeddings = embeddings[:200]
        else:
            embeddings = torch.cat((torch.zeros(200-embeddings.shape[0], 300), embeddings), dim=0)
            
        return embeddings, 0, int(self.labels[index])


def load_base_dataset(
    projects_path: str,
    costs_path: str,
    save_path: str, 
    dataset_type: Literal['embeddings', 'bert'],
    split_sizes,
    batch_size: int = 200,
    shuffle: bool = True
):
    
    # load dataset if save path exists
    if os.path.exists(save_path):
        full_dataset = torch.load(save_path)
    
    else:
        # define the full dataset
        if dataset_type == 'embeddings':
            full_dataset = EmebeddingsDataset(projects_path, costs_path)
        
        elif dataset_type == 'bert':
            full_dataset = BertDataset(projects_path, costs_path)
            
        torch.save(full_dataset, save_path)
    
    # split dataset into train, val and test
    train_size = int(split_sizes[0] * len(full_dataset))
    val_size = int(split_sizes[1] * len(full_dataset))
    test_size = len(full_dataset) - train_size - val_size
    train_dataset, val_dataset = \
        torch.utils.data.random_split(full_dataset, [0.9, 0.1])
    
    # define dataloaders
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=shuffle)
    # test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle)
    
    return train_dataloader, val_dataloader


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
