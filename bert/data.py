import torch
import re
from torch import nn
from encoder import Encoder
from transformers import BertTokenizerFast
from transformers import BertModel
from datasets import load_dataset
from bert import BERT 

# Dataset
dataset = load_dataset("squad")
Train_contexts = dataset["train"][:]["context"]
Train_questions = dataset["train"][:]["question"]
Train_answers = dataset["train"][:]["answers"]

# Tokenizer (for testing)
tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

# For chunks in which the answer does not appear, start_token=end_token=0 (predict [CLS] token) (0, 0)
# In truncated answers, start_token will be in one chunk and the end_token in another.
# Offset mappings allow mapping to token indices 
# Overflow to sample mapping maps each feature to the chunk it came from, and allows parallel examples.
"""
for ids in inputs["input_ids"]:
    print(tokenizer.decode(ids))
"""

max_length = 384
stride = 128

def preprocess_train_data(examples):
    questions = [q.strip() for q in examples["question"]]
    inputs = tokenizer(questions, 
                   examples['context'],
                   max_length = max_length, # max length of context
                   truncation="only_second", # truncate the context if the question is too long
                   stride=stride, # number of overlapping tokens between two successive chunks.
                   return_overflowing_tokens=True, # want overflowing tokens
                   return_offsets_mapping=True) 
    
    offset_mapping = inputs.pop("offset_mapping")
    sample_map = inputs.pop("overflow_to_sample_mapping")
    answers = examples["answers"]
    start_positions = []
    end_positions = []

    for i, offset in enumerate(offset_mapping):
        sample_idx = sample_map[i]
        answer = answers[sample_idx]
        start_char = answer['answer_start'][0]
        end_char = start_char + len(answer['text'][0])
        sequence_ids = inputs.sequence_ids(i)

        # Start and end of context
        idx=0
        
        while sequence_ids[idx] != 1:
            idx += 1
        start_context_idx = idx

        while sequence_ids[idx] == 1:
            idx += 1
        end_context_idx = idx - 1

        # If answer is not fully inside the context, answer is (0, 0)
        if offset[start_context_idx][0] > start_char or offset[end_context_idx][1] < end_char:
            start_positions.append(0)
            end_positions.append(0)
        else:
            idx = start_context_idx
            while idx <= end_context_idx and offset[idx][0] <= start_char:
                idx += 1
            start_positions.append(idx - 1)

            idx = end_context_idx
            while idx > start_context_idx and offset[idx][1] >= end_char:
                idx -= 1
            end_positions.append(idx+1)
    
    inputs["start_positions"] = start_positions
    inputs["end_positions"] = end_positions
    return inputs

# Testing if it works: print(len(dataset["train"]), len(train_dataset))

# For validation, do not need to generate labels

def preprocess_validation_data(examples):
    questions = [q.strip() for q in examples["question"]]
    inputs = tokenizer(
        questions,
        examples['context'],
        max_length=max_length,
        truncation="only_second",
        stride=stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding='max_length'
    )

    offset_mappings = inputs.pop('offset_mapping')

train_dataset = dataset["train"].map(
    preprocess_train_data,
    batched=True,
    remove_columns=dataset["train"].column_names,
)

validation_dataset = dataset["validation"].map(
    preprocess_validation_data,
    batched=True,
    remove_columns=dataset["validation"].column_names,
)
#print(dataset["validation"].column_names) # (id, title, context, question, answers)
#print(train_dataset.shape) # (88524, 5)
#print(validation_dataset.shape) # (10570, 5)