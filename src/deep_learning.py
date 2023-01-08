#==============================================================#
#   Project - Natural Language Processing                      #
#   Author: María Sauras & Oscar Llorente                      #
#   Date: 19/12/2022                                           #                                    
#==============================================================#


# deep learning libraries
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from torchmetrics.classification import MulticlassAccuracy
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import balanced_accuracy_score, classification_report

# other libraries
import os
from typing import Literal
from tqdm.auto import tqdm

# own modules
from src.models import LSTMModel, BertClassifier
from src.utils import load_base_dataset

# set number of threads
torch.set_num_threads(8)

# set device
device = torch.device('cuda') if torch.cuda.is_available else torch.device('cpu')

# static variables
SAVE_PATH = 'models'


def main() -> None:
    # define variables
    exec_mode: Literal['train', 'test'] = 'test'
    
    # define hyperparameters
    model_type: Literal['embeddings', 'bert'] = 'bert'
    epochs = 50
    lr = 1e-1
    hidden_dims = (512, 128, 64)
    output_dim = 6
    dropout = 0.2
    
    # load data
    train_data, val_data = load_base_dataset('data/projects.xlsx', 'data/SciVocCodes.xlsx', f'data/{model_type}.pt', 
                                             model_type, (0.9, 0.1))
    
    # define name and writer
    name = f'mc_{model_type}_h_{hidden_dims}_d_{dropout}_l_{lr}'
    writer = SummaryWriter(f'runs/{name}')
    
    # define model
    if model_type == 'embeddings':
        model = LSTMModel(output_dim, dropout).to(device)
    elif model_type == 'bert':
        model = BertClassifier(hidden_dims, output_dim, dropout).to(device)
        model.bert.requires_grad_(False)
    
    # executing depending on exec_mode variable
    if exec_mode == 'train':
    
        # define optimizer
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 30, gamma=0.1)
        
        # compute class weights
        class_weights = compute_class_weight(
            'balanced', 
            classes=np.array(list(range(output_dim))), 
            y=np.array([train_data.dataset[i][2] for i in range(len(train_data.dataset))])
        )
        class_weights = torch.Tensor(class_weights).to(device)
        
        # define loss function
        loss = torch.nn.CrossEntropyLoss(class_weights)
        
        accuracy = MulticlassAccuracy(num_classes=output_dim).to(device)
        
        # define progress bar
        progress_bar = tqdm(range(epochs*(len(train_data) + len(val_data))))
        
        for epoch in range(epochs):
            # activate training mode
            model.train()
            
            # init vectors
            losses = []
            accuracies = []
            
            for input_ids, attention_mask, labels in train_data:
                # pass objects to correct device
                input_ids = input_ids.to(device)
                if model_type == 'bert':
                    attention_mask = attention_mask.to(device)
                labels = labels.to(device)
                
                # compute outputs and loss
                if model_type == 'bert':
                    outputs = model(input_ids, attention_mask)
                else:
                    outputs = model(input_ids)
                loss_value = loss(outputs, labels)
                
                # optimize model
                optimizer.zero_grad()
                loss_value.backward()
                optimizer.step()
                
                # compute loss and accuracy
                losses.append(loss_value.item())
                accuracies.append(accuracy(outputs, labels).item())
                
                # update progress bar
                progress_bar.update()
            
            # write on tensorboard
            writer.add_scalar('accuracy/train', np.mean(accuracies), epoch)
            writer.add_scalar('loss', np.mean(losses), epoch)
                
            # activate evaluation mode
            model.eval()
            with torch.no_grad():
                # init vectors
                accuracies = []
                
                # iterate over val data
                for input_ids, attention_mask, labels in val_data:
                    # pass objects to correct device
                    input_ids = input_ids.to(device)
                    if model_type == 'bert':
                        attention_mask = attention_mask.to(device)
                    labels = labels.to(device)
                    
                    # compute outputs
                    if model_type == 'bert':
                        outputs = model(input_ids, attention_mask)
                    else:
                        outputs = model(input_ids)
                    
                    # write accuracy
                    accuracies.append(accuracy(outputs, labels).item())
                    
                    # update progress bar
                    progress_bar.update()
                    
            scheduler.step()
            
            # write on tensorboard
            writer.add_scalar('accuracy/val', np.mean(accuracies), epoch)
        
        # create directory if it does not exist
        if not os.path.exists(f'{SAVE_PATH}'):
            os.makedirs(f'{SAVE_PATH}')
            
        # saving state dict
        torch.save(model.state_dict(), f'{SAVE_PATH}/{name}.pt')
            
    
    elif exec_mode == 'test':
        # load state dict
        model.load_state_dict(torch.load(f'{SAVE_PATH}/{name}.pt'))
        
        # activate evaluation mode
        model.eval()
        with torch.no_grad():
            # init vectors
            y_test = np.array([])
            y_pred = np.array([])
            
            # iterate over val data
            for input_ids, attention_mask, labels in val_data:
                # pass objects to correct device
                input_ids = input_ids.to(device)
                if model_type == 'bert':
                    attention_mask = attention_mask.to(device)
                labels = labels.to(device)
                
                # compute outputs
                if model_type == 'bert':
                    outputs = model(input_ids, attention_mask)
                else:
                    outputs = model(input_ids)
                
                # concatenate 
                y_test = np.concatenate((y_test, labels.detach().cpu().numpy()), axis=0)
                y_pred = np.concatenate((y_pred, torch.argmax(outputs, dim=1).detach().cpu().numpy()), axis=0)
            
        print(f'accuracy: {balanced_accuracy_score(y_test, y_pred)}')
        print(f'classification report: \n{classification_report(y_test, y_pred)}')


if __name__ == '__main__':
    main()