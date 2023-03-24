
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import torch.nn as nn
import torch
from transformers.activations import gelu

class MLP_head(nn.Module):
    """Roberta Head for masked language modeling."""

    def __init__(self):
        super().__init__()
        self.dense = nn.Linear(in_features=1024, out_features=1024, bias=True)
        self.layer_norm = nn.LayerNorm((1024,), eps=1e-05, elementwise_affine=True)

        self.decoder = nn.Linear(in_features=1024, out_features=1, bias=True)
        self.bias = nn.Parameter(torch.zeros(1))

        # Need a link between the two variables so that the bias is correctly resized with `resize_token_embeddings`
        self.decoder.bias = self.bias




    def forward(self, features, **kwargs):
        x = self.dense(features)
        x = gelu(x)
        x = self.layer_norm(x)

        # project back to size of vocabulary with bias
        x = self.decoder(x)

        return x

