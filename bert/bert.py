import torch
import re
from torch import nn
from encoder import Encoder
from transformers import BertTokenizer
from transformers import BertModel

bert = BertModel.from_pretrained('bert-base-uncased')
params = list(bert.named_parameters()) # needs to be converted into a list as it is an iterator, it would be exhausted in first run of the for loop otherwise.

word_embeddings = bert.embeddings.word_embeddings
positional_embeddings = bert.embeddings.position_embeddings
segment_embeddings = bert.embeddings.token_type_embeddings
trained_layer_norm = bert.embeddings.LayerNorm
encoder_weights = bert.parameters
batch_size = [16, 32]
num_epochs = [2, 3, 4]
learning_rate_adam = [5^(-5), 3^(-5), 2^(-5)]
encoder_regex = "^encoder\.layer\."
beta1=0.9
beta2=0.999

count = 0 
layer_weights_dict = {}
while count < 12:
    layer_weights_sublist = []
    regex_count = encoder_regex+str(count)+".*"
    for name, param in params:
        if re.match(regex_count, name):
            layer_weights_sublist.append([name, param])

    layer_weights_dict[str(count)] = layer_weights_sublist
    count += 1

# print(layer_weights_dict.keys())


# BERT (Bidirectional Encoder Representations from Transformers)
class BERT(nn.Module):
    def __init__(self, batch_size=2, seq_len=10):
        super().__init__()
        # the number of encoder blocks (BASE)
        self.L = 12
        self.seq_len = seq_len
        
        # Tokenizer & Embeddings
        self.bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.word_embeddings = word_embeddings
        self.positional_embeddings = positional_embeddings
        self.segment_embeddings = segment_embeddings
        self.embedding_layernorm = trained_layer_norm
        self.embedding_dropout = nn.Dropout(p=0.1)


        # Encoder blocks, 12 for BASE
        self.es = []
        # Create a list of weights per encoder, and then create an encoder based on them, i goes from 0 to 11

        for i in range(0, self.L):
            e = Encoder("base", batch_size, seq_len, layer_weights_dict[str(i)])
            self.es.append(e)

        # Input embeddings
        self.input_embedding = nn.Embedding(512, 768)

    def forward(self, X):
        ## Tokenize the Input
        batch = self.bert_tokenizer(X, max_length=10, truncation=True, padding="max_length") # normally max_length=512
        batch_ids = torch.tensor(batch["input_ids"])
        attention_mask = torch.tensor(batch["attention_mask"])
        token_type_ids = torch.tensor(batch["token_type_ids"])
        
        ## Embeddings layers
        token_embeddings = self.word_embeddings(batch_ids)
        segment_embeddings = self.segment_embeddings(token_type_ids)
        position_embeddings = self.positional_embeddings(torch.arange(self.seq_len).unsqueeze(0))
        embeddings_with_position = token_embeddings + segment_embeddings + position_embeddings 
        embedding_normalized = self.embedding_layernorm(embeddings_with_position)
        final_embedding = self.embedding_dropout(embedding_normalized)

        ## Apply encoders repeatedly to input.
        output = self.es[0](final_embedding, attention_mask)
        for e in self.es[1:]:
            output = e(output, attention_mask)

        # CONFIGURE OUTPUTS and OUTPUT LAYER
        # Normalization, Sigmoid + output
        return output
    

BERT_Model = BERT()
out = BERT_Model(["hello world"])