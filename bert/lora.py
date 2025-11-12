
import torch
from torch import nn
from math import sqrt


class LoRA(nn.Module):
    def __init__(self, W, W_bar, rank = 1):
        self.rank = rank # r << min(d, k)
        self.W_bar = W_bar
        self.alpha = self.rank
        self.A = nn.Parameter(torch.empty(self.rank, W_bar.shape[1])) # (r x k)
        self.B = nn.Parameter(torch.zeros(W_bar.shape[0], rank)) # (d x r)
        torch.nn.init.normal_(self.A, 0, 0.02) # Gaussian Initialization
        self.W = W
    
    def forward(self, X):
        return self.W@X + (self.alpha / self.rank)*self.B@self.A@X # setting alpha = rank cancels out scaling.
