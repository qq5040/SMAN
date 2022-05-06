import torch
import torch.nn.functional as F
from torch import nn
import math
from copy import deepcopy

class MultiHeadAttn(nn.Module):
    def __init__(self, d_model, n_head=8, dropout=0.1, scale=True):

        super().__init__()
        assert d_model % n_head == 0

        self.n_head = n_head
        self.qkv_linear = nn.Linear(d_model, 3*d_model)

        self.fc = nn.Linear(d_model, d_model)

        self.dropout_layer = nn.Dropout(dropout)

        if scale:
            self.scale = math.sqrt(d_model//n_head)
        else:
            self.scale = 1

    def forward(self, x, attn_mask=None, padding_mask=None):

        batch_size, max_len, d_model = x.size()

        x = self.qkv_linear(x)
        q, k, v = torch.chunk(x, 3, dim=-1)

        q = q.view(batch_size, max_len, self.n_head, -1).transpose(1, 2)
        k = k.view(batch_size, max_len, self.n_head, -1).permute(0, 2, 3, 1)
        v = v.view(batch_size, max_len, self.n_head, -1).transpose(1, 2)

        attn = torch.matmul(q, k)  # batch_size x n_head x max_len x max_len
        attn = attn/self.scale

        if attn_mask is not None:
            attn.masked_fill_(mask=attn_mask.eq(0), value=float('-inf'))

        if padding_mask is not None:
            attn = attn.masked_fill(mask=padding_mask.eq(0).unsqueeze(1).unsqueeze(2), value=float('-inf'))

        attn = F.softmax(attn, dim=-1)  # batch_size x n_head x max_len x max_len
        attn = self.dropout_layer(attn)

        v = torch.matmul(attn, v)  # batch_size x n_head x max_len x d_model//n_head
        v = v.transpose(1, 2).reshape(batch_size, max_len, -1)  # batch_size x max_len x d_model
        v = self.fc(v)

        return v, attn.sum(dim=1)/self.n_head


class TransformerLayer(nn.Module):
    def __init__(self, d_model, self_attn, after_norm, dropout):

        super().__init__()

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.self_attn = self_attn

        self.dropout_layer = nn.Dropout(dropout)

        self.after_norm = after_norm

        self.lstm = nn.LSTM(input_size=d_model, hidden_size=d_model//2, num_layers=1, batch_first=True, bidirectional=True)

    def forward(self, x, attn_mask=None, padding_mask=None, lstm_len=None):

        residual = x
        if not self.after_norm:
            x = self.norm1(x)

        x, attn_weight = self.self_attn(x, attn_mask, padding_mask)
        x = self.dropout_layer(x)
        x = x + residual
        if self.after_norm:
            x = self.norm1(x)

        residual = x
        if not self.after_norm:
            x = self.norm2(x)

        if lstm_len is not None:
            x = nn.utils.rnn.pack_padded_sequence(x, lstm_len, True)
        x, h = self.lstm(x)
        if lstm_len is not None:
            x, _ = nn.utils.rnn.pad_packed_sequence(x, True)

        x = residual + x
        if self.after_norm:
            x = self.norm2(x)

        return x, h


class LAB(nn.Module):
    def __init__(self, num_layers=2, d_model=768, n_head=8, dropout=0.1, after_norm=True,
                 scale=True, dropout_attn=None, pos_embed='sin', device='cuda:0'):
        super().__init__()

        self.device = device

        if dropout_attn is None:
            dropout_attn = dropout
        self.d_model = d_model

        if pos_embed is None:
            self.pos_embed = None
        elif pos_embed == 'sin':
            self.pos_embed = SinusoidalPositionalEmbedding(d_model, init_size=512, padding_idx=0)
        elif pos_embed == 'fix':
            self.pos_embed = LearnedPositionalEmbedding(d_model, 512, 0)

        self_attn = MultiHeadAttn(self.d_model, n_head, dropout_attn, scale=scale)

        self.layers = nn.ModuleList([TransformerLayer(self.d_model, deepcopy(self_attn), after_norm, dropout)
                       for _ in range(num_layers)])

    def forward(self, x, attn_mask=None, padding_mask=None, lstm_len=None):

        if self.pos_embed is not None:
            x = x + self.pos_embed(padding_mask)

        for layer in self.layers:
            x, h = layer(x, attn_mask, padding_mask, lstm_len)
        return x, h


def make_positions(tensor, padding_idx):

    mask = tensor.ne(padding_idx).int()
    return (
        torch.cumsum(mask, dim=1).type_as(mask) * mask
    ).long() + padding_idx


class SinusoidalPositionalEmbedding(nn.Module):

    def __init__(self, embedding_dim, init_size=1024, padding_idx=0):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        self.weights = SinusoidalPositionalEmbedding.get_embedding(
            init_size,
            embedding_dim,
            padding_idx,
        )
        self.register_buffer('_float_tensor', torch.FloatTensor(1))

    @staticmethod
    def get_embedding(num_embeddings, embedding_dim, padding_idx=None):

        half_dim = embedding_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float) * -emb)
        emb = torch.arange(num_embeddings, dtype=torch.float).unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1).view(num_embeddings, -1)
        if embedding_dim % 2 == 1:
            # zero pad
            emb = torch.cat([emb, torch.zeros(num_embeddings, 1)], dim=1)
        if padding_idx is not None:
            emb[padding_idx, :] = 0
        return emb

    def forward(self, input):
        """Input is expected to be of size [bsz x seqlen]."""
        bsz, seq_len = input.size()
        max_pos = self.padding_idx + 1 + seq_len
        if max_pos > self.weights.size(0):
            # recompute/expand embeddings if needed
            self.weights = SinusoidalPositionalEmbedding.get_embedding(
                max_pos,
                self.embedding_dim,
                self.padding_idx,
            )
        self.weights = self.weights.to(self._float_tensor)

        positions = make_positions(input, self.padding_idx)
        return self.weights.index_select(0, positions.view(-1)).view(bsz, seq_len, -1).detach()

    def max_positions(self):
        """Maximum number of supported positions."""
        return int(1e5)  # an arbitrary large number


class LearnedPositionalEmbedding(nn.Embedding):

    def __init__(
            self,
            embedding_dim: int,
            num_embeddings: int = 1024,
            padding_idx: int = 0,
    ):
        super().__init__(num_embeddings, embedding_dim, padding_idx)

    def forward(self, input):
        positions = make_positions(input, self.padding_idx)
        return super().forward(positions)

