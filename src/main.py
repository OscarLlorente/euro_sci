# deep learning libraries
import torch
import torchmetrics
import numpy as np
from torch.utils.tensorboard import SummaryWriter

# other libraries
from tqdm.auto import tqdm

# own modules
from src.models import BertClassifier
from src.utils import load_base_dataset

# set device
device = torch.device('cuda') if torch.cuda.is_available else torch.device('cpu')


def main() -> None:
    # define variables
    epochs = 10
    lr = 1e-3
    
    train_data, val_data, test_data = load_base_dataset('data/tranformer_dataset', 'data/projects.xlsx', 
                                                        'data/SciVocCodes.xlsx', (0.8, 0.1, 0.1))
    
    writer = SummaryWriter('runs/model')
    
    # define model
    model = BertClassifier((1024, 512), 1010, 0.3).to(device)
    model.bert.requires_grad_(False)
    
    # define optimizer
    optimizer = torch.optim.AdamW(model.mlp.parameters(), lr=lr)
    
    # define loss function
    loss = torch.nn.CrossEntropyLoss()
    
    accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=1010).to(device)
    
    progress_bar = tqdm(range(epochs*len(train_data)))
    
    for epoch in range(epochs):
        # activate training mode
        model.train()
        
        # init vectors
        losses = []
        accuracies = []
        
        for input_ids, attention_mask, labels in train_data:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)
            
            outputs = model(input_ids, attention_mask)
            loss_value = loss(outputs, labels)
            
            optimizer.zero_grad()
            loss_value.backward()
            optimizer.step()
            
            losses.append(loss_value.item())
            accuracies.append(accuracy(outputs, labels).item())
            
            progress_bar.update()
            
        writer.add_scalar('loss', np.mean(losses), epoch)
        writer.add_scalar('loss/train', np.mean(accuracies), epoch)


if __name__ == '__main__':
    main()