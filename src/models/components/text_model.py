import torch

from torch import nn
from transformers import BertModel

class BertWrapper(nn.Module):
    def __init__(
        self, 
        pretrained_name: str,
    ):
        super().__init__()
        self.bert = BertModel.from_pretrained(pretrained_name)
        
    def forward(self, x: torch.Tensor):
        return self.bert(x)[0]