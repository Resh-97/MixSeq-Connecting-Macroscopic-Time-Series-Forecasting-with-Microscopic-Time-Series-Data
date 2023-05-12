import torch
import atention as attn


class EncoderBlock(torch.nn.Module):
    """Transformer encoder block."""

    def __init__(self, key_size, query_size, value_size, n_heads, n_hiddens,
                 ffn_d_in, ffn_d_hid, dropout=0.1, bias=False, **kwargs):
        super(EncoderBlock, self).__init__(**kwargs)
        self.dropout = dropout[0] if type(dropout) is tuple else dropout
        self.slf_attn = attn.MultiheadAttention(
            key_size, query_size, value_size, n_heads, n_hiddens, self.dropout, bias=bias)
        self.layer_norm_1 = torch.nn.LayerNorm(n_hiddens)
        self.pos_ffn = attn.PositionwiseFeedForward(
            ffn_d_in, ffn_d_hid, self.dropout)
        self.layer_norm_2 = torch.nn.LayerNorm(n_hiddens)

    def forward(self, enc_x, slf_attn_mask=None):
        self.slf_attn.attention.eval()
        enc_output, enc_slf_attn = self.slf_attn(
            enc_x, enc_x, enc_x, slf_attn_mask)  # enc_x to same queries,keys,values
        ln1 = self.layer_norm_1(enc_x, enc_slf_attn)
        # enc_outputs: [batch_size x query_size x n_hiddens]
        enc_output = self.pos_ffn(enc_output)
        enc_output = self.layer_norm_2(ln1, enc_output)
        return enc_output, enc_slf_attn


class DecoderBlock(torch.nn.Module):
    # The `i`-th block in the decoder
    def __init__(self, key_size, query_size, value_size, n_heads, n_hiddens, ffn_d_in, ffn_d_hid,
                 dropout, **kwargs):
        super(DecoderBlock, self).__init__(**kwargs)
        self.dropout = dropout[0] if type(dropout) is tuple else dropout
        self.slf_attn = attn.MultiheadAttention(
            key_size, query_size, value_size, n_heads, n_hiddens, self.dropout)
        self.layer_norm_1 = torch.nn.LayerNorm(n_hiddens)
        self.enc_attn = attn.MultiheadAttention(
            key_size, query_size, value_size, n_heads, n_hiddens, self.dropout)
        self.layer_norm_2 = torch.nn.LayerNorm(n_hiddens)
        self.pos_ffn = attn.PositionwiseFeedForward(
            ffn_d_in, ffn_d_hid, self.dropout)
        self.layer_norm_3 = torch.nn.LayerNorm(n_hiddens)

    def forward(self, dec_x, enc_outputs, dec_self_attn_mask=None, dec_enc_attn_mask=None):
        # Self-attention
        dec_output, dec_slf_attn = self.slf_attn(
            dec_x, dec_x, dec_x, dec_self_attn_mask)
        ln1 = self.layer_norm_1(dec_x, dec_slf_attn)
        dec_output, dec_enc_attn = self.enc_attn(
            dec_output, enc_outputs, enc_outputs, dec_enc_attn_mask)
        ln2 = self.layer_norm_2(ln1, dec_enc_attn)
        dec_output = self.pos_ffn(dec_output)
        dec_output = self.layer_norm_3(ln2, dec_output)
        return dec_output, dec_slf_attn, dec_enc_attn
