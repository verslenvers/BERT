import torch
from torch import nn
from math import sqrt

class Encoder(nn.Module):

    def __init__(self, model, batch_size, seq_len, pretrained_weights=0):
        super().__init__()
        self.pretrained_weights = pretrained_weights

        # self.N is batch_size
        self.N = batch_size
        self.seq_len = seq_len 
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

        # Define dropout, linear layers, layernorm, and GELU           
        self.linear1 = nn.Linear(self.H, self.F)
        self.gelu = nn.GELU()
        self.linear2 = nn.Linear(self.F, self.H)
        self.layernorm_attention = nn.LayerNorm(self.H) 
        self.dropout = nn.Dropout(0.1)
        self.layernorm_output = nn.LayerNorm(self.H)

        # Initialize weights for multi-head attention
        if pretrained_weights == 0:
            for i in range(0, self.A):
                # d_model / h (or self.A) = 768 / 12 = 64 
                WQ = torch.zeros((self.H, self.H // self.A))
                WK = torch.zeros((self.H, self.H // self.A))
                WV = torch.zeros((self.H, self.H // self.A))

                bQ = torch.zeros(64,)
                bK = torch.zeros(64,)
                bV = torch.zeros(64,)

                nn.init.xavier_normal_(WQ)
                nn.init.xavier_normal_(WK)
                nn.init.xavier_normal_(WV)
                self.W.append([WQ, WK, WV, bQ, bK, bV])


            # dim(W_O) = h*d_v x d_model = 12*64 x 768 = 768 x 768
            self.WO = torch.zeros((self.H, self.H))
            self.bO = torch.zeros((self.H,))
            nn.init.xavier_normal_(self.WO)

        else:
           with torch.no_grad():
            self.WQ = pretrained_weights[0][1]
            self.bQ = pretrained_weights[1][1]
            self.WK = pretrained_weights[2][1]
            self.bK = pretrained_weights[3][1]
            self.WV = pretrained_weights[4][1]
            self.bV = pretrained_weights[5][1]
            self.WO = pretrained_weights[6][1]
            self.bO = pretrained_weights[7][1]
            self.layernorm_attention.weight.copy_(pretrained_weights[8][1])
            self.layernorm_attention.bias.copy_(pretrained_weights[9][1])
            self.linear1.weight.copy_(pretrained_weights[10][1])
            self.linear1.bias.copy_(pretrained_weights[11][1])
            self.linear2.weight.copy_(pretrained_weights[12][1])
            self.linear2.bias.copy_(pretrained_weights[13][1])
            self.layernorm_output.weight.copy_(pretrained_weights[14][1])
            self.layernorm_output.bias.copy_(pretrained_weights[15][1])

            
    def multi_head_self_attention(self, X: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if self.pretrained_weights == 0:
            # the output for each head
            Zs = []
            for head_i in self.W:
                WQ = head_i[0]
                WK = head_i[1]
                WV = head_i[2]
                bQ = head_i[3]
                bK = head_i[4]
                bV = head_i[5]
            
                # Q is (N x (self.H / self.A))
                Q = X@WQ + bQ
                K = X@WK + bK
                V = X@WV + bV


                scores = (Q@K.transpose(1, 2))/sqrt(self.H / self.A)
                scores_masked = scores.masked_fill(mask.view(10, 1, 1) == 0, float('-inf')) # adjust the ten so it's not hard-coded, should be seq_len. 
                attn_weights = nn.functional.softmax(scores_masked, dim=-1)
            
                attn_weights = self.dropout(attn_weights)

                Z = attn_weights@V
                Zs.append(Z) 

            # Concat attention output from each head.
            O = torch.concat(Zs, dim=2)@self.WO  + self.bO
            return O
        
        else:
            # (1, 10, 768) or (batch_size, seq_len, hidden_size)
            Q = X@self.WQ + self.bQ
            K = X@self.WK + self.bK
            V = X@self.WV + self.bV

            # Reshape to (batch, number of heads, seq_len, hidden_size // number of heads) ==> (N, 12, 10, 64)
            Q = Q.view(Q.shape[0], self.A, self.seq_len, self.H // self.A)
            K = K.view(K.shape[0], self.A, self.seq_len, self.H // self.A)
            V = V.view(V.shape[0], self.A, self.seq_len, self.H // self.A)
            
            # Tranpose so the last two dims of each tensor can match during matrix multiplication
            scores = (Q@K.transpose(2, 3))/sqrt(self.H / self.A) # (batch_size, num_heads, seq_len, seq_len)
            # Mask shape is (1, 10) ==> should be [batch_size, 1, 1, seq_len] # for brodcasting
            scores_masked = scores.masked_fill(mask.view(Q.shape[0], 1, 1, self.seq_len) == 0, float('-inf')) # (1, 1, 1, 10)

            attn_weights = nn.functional.softmax(scores_masked, dim=-1)
            attn_weights = self.dropout(attn_weights) # (batch_size, num_heads, seq_len, seq_len)

            # Concatenate weights together
            Z = attn_weights@V
            Z = Z.view(Z.shape[0], self.seq_len, self.H)
            O = Z@self.WO + self.bO

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
        X = self.layernorm_attention(X)
        X2 = self.feed_forward_network(X) # (N, self.H)
        X = self.dropout(X2) + X
        X = self.layernorm_output(X)

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
