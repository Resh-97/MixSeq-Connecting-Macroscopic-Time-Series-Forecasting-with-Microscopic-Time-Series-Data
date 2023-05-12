import torch
import math
import log_sparse_mask as log_sparse
import casual_convolution as cc


def sequence_mask(x, s_mask, value=0):
    """Mask irrelevant entries in sequences."""
    maxlen = x.size(1)
    mask = torch.arange((maxlen), dtype=torch.float32,
                        device=x.device)[None, :] < s_mask[:, None]
    x[~mask] = value
    return x


def masked_softmax(x, s_mask):
    """Perform softmax operation by masking elements on the last axis."""
    # `x`: 3D tensor, `valid_lens`: 1D or 2D tensor
    if s_mask is None:
        return torch.nn.functional.softmax(x, dim=-1)
    else:
        shape = x.shape
        if s_mask.dim() == 1:
            s_mask = torch.repeat_interleave(s_mask, shape[1])
        else:
            s_mask = s_mask.reshape(-1)
        # On the last axis - Fills elements of self tensor with value where mask is one.
        x = sequence_mask(x.reshape(-1, shape[-1]), s_mask,
                          value=-1e6)
        return torch.nn.functional.softmax(x.reshape(shape), dim=-1)


class ScaledDotProductAttention(torch.nn.Module):
    """Scaled dot product attention."""

    def __init__(self, attn_dropout=0.1, **kwargs):
        super(ScaledDotProductAttention, self).__init__(**kwargs)
        self.dropout = torch.nn.Dropout(attn_dropout)

    """ Shape of queries: (batch_size, no. of queries, d)
        Shape of keys: (batch_size, no. of key-value pairs, d)
        Shape of values: (batch_size, no. of key-value pairs, value dimension)
        Shape of attn_mask: (batch_size,) or (batch_size, no. of queries)"""

    def forward(self, queries, keys, values, attn_mask=None):
        d_key = queries.shape[-1]
        # Set `transpose_b=True` to swap the last two dimensions of `keys`
        # scores : [batch_size x n_heads x len_q(=len_k) x len_k(=len_q)]
        scores = torch.matmul(queries, keys.transpose(1, 2)) / math.sqrt(d_key)

        attention_weights = masked_softmax(scores, attn_mask)
        output = torch.matmul(self.dropout(attention_weights), values)
        return output, attention_weights


class MultiheadAttention(torch.nn.Module):
    """Multi-head attention.
        O_h = Attention(Q_h, K_h, V_h)"""

    def __init__(self, key_size, query_size, value_size, n_heads, n_hiddens, dropout=0.1, bias=False, **kwargs):
        super(MultiheadAttention, self).__init__(**kwargs)
        self.n_heads = n_heads
        self.query_size = query_size
        self.key_size = key_size
        self.value_size = value_size

        self.w_q = cc.CasualConvolution1d(
            in_channels=n_hiddens, out_channels=n_heads * query_size, kernel_size=1)
        self.w_k = cc.CasualConvolution1d(
            in_channels=n_hiddens, out_channels=n_heads * key_size, kernel_size=1)
        self.w_v = torch.nn.Linear(n_hiddens, n_heads * value_size, bias=False)
        self.fc = torch.nn.Linear(n_heads * value_size, n_hiddens, bias=False)
==========
        self.attention = ScaledDotProductAttention(dropout)

        self.dropout = torch.nn.Dropout(dropout)
        self.layer_norm = torch.nn.LayerNorm(n_hiddens, eps=1e-6)

    def forward(self, queries, keys, values, attn_mask=None):

        residual, batch_size = queries, queries.size(0)
        # q,k,v: [batch_size x n_heads x (queries or keys or values) x (query_size or key_size or value_size)]

        queries = self.w_q(queries).view(
            batch_size, -1, self.n_heads, self.query_size).transpose(1, 2)
        keys = self.w_k(keys).view(batch_size, -1, self.n_heads,
                                   self.key_size).transpose(1, 2)
        values = self.w_v(values).view(
            batch_size, -1, self.n_heads, self.value_size).transpose(1, 2)

        # attn_mask: [(batch_size,) or (batch_size,q_size)]
        if attn_mask is not None:
            # On axis 0, copy the first item (scalar or vector) for n_heads times, then copy the next item, and so on
            attn_mask = torch.repeat_interleave(
                attn_mask, repeats=self.n_heads, dim=0)
#             attn_mask = attn_mask.unsqueeze(1).repeat(1, n_heads, 1, 1)

        # output: [batch_size x n_heads x q_size x v_size], attn: [batch_size x n_heads x q_size(=k_size) x k_size(=q_size)]
        output, attn = self.attention(queries, keys, values, attn_mask)

        output = output.transpose(1, 2).contiguous().view(
            batch_size, -1, self.n_heads * self.value_size)
        output = self.dropout(self.fc(output))

        # output: [batch_size x queries x n_hiddens]
        return self.layer_norm(output + residual), attn


class PositionwiseFeedForward(torch.nn.Module):
    """Positionwise feed-forward network -- as per paper the MLP part"""

    def __init__(self, ffn_d_in, ffn_d_hid, dropout=0.1, eps=1e-6, **kwargs):
        super(PositionwiseFeedForward, self).__init__(**kwargs)
        self.dense_1 = torch.nn.Conv1d(
            in_channels=ffn_d_in, out_channels=ffn_d_hid, kernel_size=1)  # position-wise
        self.dense_2 = torch.nn.Conv1d(
            in_channels=ffn_d_hid, out_channels=ffn_d_in, kernel_size=1)  # position-wise
        self.relu = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(dropout)
        self.layer_norm = torch.nn.LayerNorm(ffn_d_in, eps)

    def forward(self, x):
        residual = x  # x : [batch_size, len_q, ffn_d_in]
        hidden_1 = self.relu(self.dense_1(x.transpose(1, 2)))
        hidden_2 = self.dense_2(hidden_1).transpose(1, 2)
        output = self.dropout(hidden_2)
        return self.layer_norm(output + residual)
