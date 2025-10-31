import torch
from torch import nn
from math import sqrt

class Encoder(nn.Module):

    def __init__(self, model, batch_size, seq_len, pretrained_weights=0):
        super().__init__()
        self.pretrained_weights = pretrained_weights

        # self.N is batch_size
        self.N = batch_size
        self.seq_len = seq_len # POLISH
        # self.A is number of heads
        self.A = None
        # self.H is hidden size
        self.H = None
        # self.F is the size of the hidden layer in the feed-forward neural network
        self.F = 3072

        if model == "base":
            self.A = 12
            self.H = 768
           
        self.W = []
        # Initialize weights for multi-head attention
        if pretrained_weights == 0:
            for i in range(0, self.A):
                # d_model / h (or self.A) = 768 / 12 = 64 
                WQ = torch.zeros((self.H, self.H // self.A))
                WK = torch.zeros((self.H, self.H // self.A))
                WV = torch.zeros((self.H, self.H // self.A))

                nn.init.xavier_normal_(WQ)
                nn.init.xavier_normal_(WK)
                nn.init.xavier_normal_(WV)
                self.W.append([WQ, WK, WV])


            # dim(W_O) = h*d_v x d_model = 12*64 x 768 = 768 x 768
            self.WO = torch.zeros((self.H, self.H))
            nn.init.xavier_normal_(self.WO)

            # define dropout, linear layers, layernorm, and GELU
            self.linear1 = nn.Linear(self.H, self.F)
            self.gelu = nn.GELU()
            self.linear2 = nn.Linear(self.F, self.H)
            self.dropout = nn.Dropout(0.1)
            self.layernorm = nn.LayerNorm(self.H) # add normalized shape
        #else:
            #WQ = 

            
    def multi_head_self_attention(self, X: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # the output for each head
        Zs = []
        for head_i in self.W:
            WQ = head_i[0]
            WK = head_i[1]
            WV = head_i[2]
            
            # Q is (N x (self.H / self.A))
            Q = X@WQ
            K = X@WK
            V = X@WV


            scores = (Q@K.transpose(1, 2))/sqrt(self.H / self.A)
            scores_masked = scores.masked_fill(mask.view(10, 1, 1) == 0, float('-inf'))
            attn_weights = nn.functional.softmax(scores_masked, dim=-1)
            
            attn_weights = self.dropout(attn_weights)

            Z = attn_weights@V
            Zs.append(Z) 

        # Concat attention output from each head.
        O = torch.concat(Zs, dim=-1)@self.WO
        return O
    
    def feed_forward_network(self, X: torch.Tensor) -> torch.Tensor:
        X = self.linear1(X)
        X = self.gelu(X)
        X = self.linear2(X)
        return X
    
    def forward(self, X: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # X dimension is (N, self.H)
        X1 = self.multi_head_self_attention(X, mask) # (N, self.H)
        X = self.dropout(X1) + X
        X = self.layernorm(X)
        X2 = self.feed_forward_network(X) # (N, self.H)
        X = self.dropout(X2) + X
        X = self.layernorm(X)

        return X
    
    def get(self):
        return self.pretrained_weights

"""
Testing:
encoder = Encoder("base", 2, 10)
x = torch.randn(2, 10, 768)  # batch=2, seq_len=10
mask = torch.zeros(2, 10, 768)
out = encoder(x, mask)
"""
