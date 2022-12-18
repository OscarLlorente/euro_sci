# deep learning libraries
import torch
import torchmetrics
import numpy as np
from torch.utils.tensorboard import SummaryWriter

# other libraries
import os
from typing import Literal
from tqdm.auto import tqdm

# own modules
from src.models import Mlp, BertClassifier
from src.utils import load_base_dataset, accuracy

# set device
device = torch.device('cuda') if torch.cuda.is_available else torch.device('cpu')

# static variables
SAVE_PATH = 'models'


def main() -> None:
    # define variables
    exec_mode: Literal['train', 'test'] = 'train'
    
    # define hyperparameters
    model_type: Literal['tfidf', 'bert'] = 'bert'
    epochs = 10
    lr = 1e-3
    hidden_dims = (512, 128, 64)
    output_dim = 5
    dropout = 0.1
    
    train_data, val_data, test_data = load_base_dataset('data/df.csv', f'data/{model_type}', model_type, 
                                                        (0.8, 0.1, 0.1))
    
    # define name and writer
    name = f'm_{model_type}_h_{hidden_dims}_d_{dropout}_l_{lr}'
    writer = SummaryWriter(f'runs/{name}')
    
    # define model
    if model_type == 'tfidf':
        model = Mlp(300, hidden_dims, output_dim, dropout).to(device)
    elif model_type == 'bert':
        model = BertClassifier(hidden_dims, output_dim, dropout).to(device)
        model.bert.requires_grad_(False)
    
    if exec_mode == 'train':
    
        # define optimizer
        optimizer = torch.optim.AdamW(model.mlp.parameters(), lr=lr)
        
        # define loss function
        loss = torch.nn.CrossEntropyLoss()
        
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
                attention_mask = attention_mask.to(device)
                labels = labels.to(device)
                
                # compute outputs and loss
                outputs = model(input_ids, attention_mask)
                loss_value = loss(outputs, labels)
                
                # optimize model
                optimizer.zero_grad()
                loss_value.backward()
                optimizer.step()
                
                # compute loss and accuracy
                losses.append(loss_value.item())
                accuracies.append(accuracy(outputs, labels))
                
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
                    attention_mask = attention_mask.to(device)
                    labels = labels.to(device)
                    
                    # compute outputs
                    outputs = model(input_ids, attention_mask)
                    
                    # write accuracy
                    accuracies.append(accuracy(outputs, labels))
                    
                    # update progress bar
                    progress_bar.update()
            
            # write on tensorboard
            writer.add_scalar('accuracy/val', np.mean(accuracies), epoch)
        
        # create directory if it does not exist
        if not os.path.exists(f'{SAVE_PATH}'):
            os.makedirs(f'{SAVE_PATH}')
            
        # saving state dict
        torch.save(model.state_dict(), f'{SAVE_PATH}/{name}.pt')
            
            
    elif exec_mode == 'test':
        pass


if __name__ == '__main__':
    main()