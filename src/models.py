# deep learning libraries
import torch
from transformers import BertModel

class BertClassifier(torch.nn.Module):

    def __init__(self, hidden_dims: tuple[int, ...], output_dim: int, dropout: float) -> None:

        super().__init__()

        # define bert transformer
        self.bert = BertModel.from_pretrained('bert-base-cased')

        # define multi-layer-perceptron
        mlp_layers = [torch.nn.Linear(768, hidden_dims[0])]
        for i in range(1, len(hidden_dims)):
            mlp_layers.append(torch.nn.ReLU())
            mlp_layers.append(torch.nn.Dropout())
            mlp_layers.append(torch.nn.Linear(hidden_dims[i-1], hidden_dims[i]))
        mlp_layers.append(torch.nn.ReLU())
        mlp_layers.append(torch.nn.Dropout())
        mlp_layers.append(torch.nn.Linear(hidden_dims[-1], output_dim))
        self.mlp = torch.nn.Sequential(*mlp_layers)

    def forward(self, input_ids, attention_mask) -> torch.Tensor:
        
        # compute bert outputs
        _, pooled_output = self.bert(input_ids= input_ids, attention_mask=attention_mask,return_dict=False)

        # compute mlp outputs
        outputs = self.mlp(pooled_output)

        return outputs