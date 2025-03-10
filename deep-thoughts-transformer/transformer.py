import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

# about word_embedding, taking sequence modeling as an example
# think about source sentence and target sentence
# construct a sequence whose characters are represented by their index in the vocabulary
batch_size = 2

# the size of vocabulary
max_num_src_words = 8
max_num_tgt_words = 8
model_dim = 8

# the max length of sequence
max_src_seq_len = 5
max_tgt_seq_len = 5
max_position_len = 5

# src_len = torch.randint(2, 5, (batch_size,))
# tgt_len = torch.randint(2, 5, (batch_size,))

# two sentences, with 2 and 4 words respectively
src_len = torch.Tensor([2, 4]).to(torch.int32)
# two sentences, with 4 and 3 words respectively
tgt_len = torch.Tensor([4, 3]).to(torch.int32)

# Step 1: construct sentences made up of word indexes
src_seq = [torch.randint(1, max_num_src_words, (L,)) for L in src_len]
tgt_seq = [torch.randint(1, max_num_tgt_words, (L,)) for L in tgt_len]
# print("before padding")
# print(src_seq)
# print(tgt_seq)

# add padding to make the length consistent
src_seq = [F.pad(seq, pad=(0, max(src_len) - len(seq))) for seq in src_seq]
tgt_seq = [F.pad(seq, pad=(0, max(tgt_len) - len(seq))) for seq in tgt_seq]
# print("after padding")
# print(src_seq)
# print(tgt_seq)

# stack the sequence
src_seq = torch.stack(src_seq)
tgt_seq = torch.stack(tgt_seq)

# Step 2: construct embedding
# add 1 to account for padding
src_embedding_table = nn.Embedding(max_num_src_words + 1, model_dim)
tgt_embedding_table = nn.Embedding(max_num_tgt_words + 1, model_dim)
# print(src_embedding_table.weight)
# print(tgt_embedding_table.weight)

src_embedding = src_embedding_table(src_seq)
tgt_embedding = tgt_embedding_table(tgt_seq)
# print(src_seq)
# print(src_embedding)
# print(tgt_seq)
# print(tgt_embedding)

# Step 3: construct position embedding
# In the PE formula, pos represents the row and i represents the column
pos_mat = torch.arange(max_position_len).reshape((-1, 1))
# print(pos_mat)
i_mat = torch.pow(10000, torch.arange(0, 8, 2).reshape((1, -1)) / model_dim)
# print(i_mat)
pe_table = torch.zeros(max_position_len, model_dim)
# using the broadcast mechanism
pe_table[:, 0::2] = torch.sin(pos_mat / i_mat)
pe_table[:, 1::2] = torch.cos(pos_mat / i_mat)
# print(pe_table)

# Step 4: construct self-attention mask of encoder
# shape of mask: [batch_size, max_src_len, max_src_len], the value is 1 or negative inf
valid_encoder_pos = [F.pad(torch.ones(L), (0, max(src_len) - L)) for L in src_len]
valid_encoder_pos = torch.unsqueeze(torch.stack(valid_encoder_pos), 2)
# print(valid_encoder_pos)
valid_encoder_pos_mat = torch.bmm(valid_encoder_pos, valid_encoder_pos.transpose(1, 2))
# print(valid_encoder_pos_mat)
invalid_encoder_pos_mat = 1 - valid_encoder_pos_mat
mask_encoder_self_attention = invalid_encoder_pos_mat.to(torch.bool)
# print(mask_encoder_self_attention)

score = torch.randn(batch_size, max(src_len), max(src_len))
masked_score = score.masked_fill(mask_encoder_self_attention, -1e9)
prob = F.softmax(masked_score, dim=-1)
# print(score)
# print(masked_score)
# print(prob)

# Step 5: construct mask of intra-attention
# Q @ K^T shape: [batch_size, tgt_seq_len, src_seq_len]
valid_decoder_pos = [F.pad(torch.ones(L), (0, max(tgt_len) - L)) for L in tgt_len]
valid_decoder_pos = torch.unsqueeze(torch.stack(valid_decoder_pos), 2)
# print(valid_decoder_pos)
valid_cross_pos_mat = torch.bmm(valid_decoder_pos, valid_encoder_pos.transpose(1, 2))
# print(valid_cross_pos_mat)
invalid_cross_pos_mat = 1 - valid_cross_pos_mat
mask_cross_attention = invalid_cross_pos_mat.to(torch.bool)
# print(mask_cross_attention)

# Step 6: construct mask of decoder self-attention
# tri_matrix = [torch.tril(torch.ones((L, L))) for L in tgt_len]
valid_decoder_tri_mat = [F.pad(torch.tril(torch.ones((L, L))), (0, max(tgt_len) - L, 0, max(tgt_len) - L)) for L in tgt_len]
valid_decoder_tri_mat = torch.stack(valid_decoder_tri_mat)
invalid_decoder_tri_mat = 1 - valid_decoder_tri_mat
invalid_decoder_tri_mat = invalid_decoder_tri_mat.to(torch.bool)

score = torch.randn(batch_size, max(tgt_len), max(tgt_len))
masked_score = score.masked_fill(invalid_decoder_tri_mat, -1e9)
prob = F.softmax(masked_score, dim=-1)
print(tgt_len)
print(prob)

# Step 7: construct scaled self-attention
def scaled_dot_product_attention(q, k, v, attn_mask):
    # shape of q, k, v: (batch_size * num_head, seq_len, model_dim / num_head)
    score = torch.bmm(q, k.transpose(-2, -1)) / torch.sqrt(model_dim)
    masked_score = score.masked_fill(attn_mask, -1e9)
    prob = F.softmax(masked_score, dim=-1)
    context = torch.bmm(prob, v)
    return context