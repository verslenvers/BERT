
import torch
from torch import nn


class LoRA(nn.Module):
    def __init__(self, W, W_bar, rank = 1):
        self.rank = rank # r << min(d, k)
        self.W_bar = W_bar

        ## TO DO: GAUSSIAN INITIALIZATION FOR A 
        # check torch.normal
        self.A = torch.zeros(rank, W_bar.shape[1]) # (r x k)
        self.B = torch.zeros(W_bar.shape[0], rank) # (d x r)
        self.W = W
    
    def forward(self, X):
        ## FIND ALPHA
        alpha = 0
        h = self.W@X + (alpha / self.rank)*self.B@self.A@X
