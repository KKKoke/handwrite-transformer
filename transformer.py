import torch
import numpy
import torch.nn as nn
import torch.nn.functional as F

# about word_embedding, taking sequence modeling as an example
# think about source sentence and target sentence
# construct a sequence whose characters are represented by their index in the vocabulary
batch_size = 2

# the size of vocabulary
max_num_src_words = 8
max_num_tgt_words = 8

# the nax length of sequence
max_src_seq_len = 5
max_tgt_seq_len = 5

# src_len = torch.randint(2, 5, (batch_size,))
# tgt_len = torch.randint(2, 5, (batch_size,))

# two sentences, with 2 and 4 words respectively
src_len = torch.Tensor([2, 4]).to(torch.int32)
# two sentences, with 4 and 3 words respectively
tgt_len = torch.Tensor([4, 3]).to(torch.int32)

# sentences made up of word indexes
src_seq = [torch.randint(1, max_num_src_words, (L,)) for L in src_len]
tgt_seq = [torch.randint(1, max_num_tgt_words, (L,)) for L in tgt_len]
print(src_seq)
print(tgt_seq)