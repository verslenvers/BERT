import torch
from torch import nn
from bert import BERT 
from data import validation_dataset, train_dataset
import torch.optim as optim

batch_size = [16, 32]
num_epochs = [2, 3, 4]
learning_rate_adam = [5^(-5), 3^(-5), 2^(-5)]
beta1=0.9
beta2=0.999

# Question and context
question = "Where does photosynthesis mainly occur in plant cells?"
context = """The process of photosynthesis allows plants to convert light energy from the sun into chemical energy stored in glucose. 
This process takes place mainly in the chloroplasts of plant cells, where chlorophyll captures sunlight. During photosynthesis, carbon dioxide and water are used to produce glucose and oxygen."""

# Get logits
model = BERT()

# Define optimizer with weight decay
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)

# Loss function
loss_fn = nn.CrossEntropyLoss()

# Freeze all layers
for name, param in model.named_parameters(recurse=True, remove_duplicate=False):
    param.requires_grad = False
    if 'WQ' in name or 'WK' in name or 'WO' in name or 'WV' in name:
        pass

## FIND A WAY TO UNFREEZE A and B

num_epochs=2

# Fix this
for epoch in range(num_epochs):
    for id, _, (context, question, answers) in train_dataset:
        optimizer.zero_grad()
        start_logits, end_logits = model(x, y)
        print(x)
        print(y)
        start_positions, end_positions = y # CHECK IF THIS HOLDS

        start_loss = loss_fn(start_logits, start_positions)
        end_loss = loss_fn(end_logits, end_positions)
        start_loss.backward() # find out if I should do this for both start and end
        end_loss.backward() # find out if I should do this for both start and end
        optimizer.step()



















"""
inputs = Bert_tokenizer(question, 
                        context, 
                        max_length=384, 
                        trunction="only_second", 
                        return_tensors='pt', 
                        return_overflowing_tokens=True, 
                        stride=128, # stride determines overlap of overflowing tokens
                        padding="longest") 

# To get the inputs without the extra stuff now,
_ = inputs.pop("overflow_to_sample_mapping")
offsets = inputs.pop("offset_mapping")
inputs = inputs.convert_to_tensors("pt")
print(inputs["input_ids"].shape)

# To test the above
for ids in inputs["input_ids"]:
    print(Bert_tokenizer.decode(ids))
sequence_ids = inputs.sequence_ids()

# Mask all tokens that are not a part of the context
mask = [i != 1 for i in sequence_ids]
# Unmask [CLS] token
mask[0] = False
mask = torch.tensor(mask)[None] # [None] for batch dim

# Mask all [PAD] tokens
mask = torch.logical_or(torch.tensor(mask)[None], (inputs["attention_mask"] == 0))
# As all the True values correspond to the Question and [SEP] values, these are masked (set equal to -10000)
# True values, the context and the [CLS] token, are retained and softmax is performed on them w.r.t. start and end logits.
start_logits[mask] = -10000
end_logits[mask] = -10000
start_probabilities = torch.nn.functional.softmax(start_logits, dim=-1)[0]
end_probabilities = torch.nn.functional.softmax(end_logits, dim=-1)[0]

# To find the likelihood of a particular start to end index answer, multiple probabilities of each
candidates = []
for start_probs, end_probs in zip(start_probabilities, end_probabilities):
    scores = start_probs[:, None] * end_probs[None, :]
    # Mask all positions where start_index > end_index
    idx = torch.triu(scores).argmax().item()

    start_idx = idx // scores.shape[1]
    end_idx = idx % scores.shape[1]

    score = scores[start_idx, end_idx].item()
    candidates.append((start_idx, end_idx, score))

print(candidates)

for candidate, offset in zip(candidates, offsets):
    start_token, end_token, score = candidate
    start_char, _ = offset[start_token]
    _, end_char = offset[end_token]
    answer = context[start_char:end_char]
    result = {"answer": answer, "start": start_char, "end": end_char, "score": score}
    print(result)
"""