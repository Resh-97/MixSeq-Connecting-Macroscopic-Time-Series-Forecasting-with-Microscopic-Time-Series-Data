import torch
import numpy as np
import layers as layer


class PositionalEncoding(torch.nn.Module):
    """Positional encoding."""

    def __init__(self, n_hiddens, dropout=0.1, n_position=200):
        super(PositionalEncoding, self).__init__()
        self.dropout = torch.nn.Dropout(
            p=dropout[0] if type(dropout) is tuple else dropout)
        # Create a long enough pos
        self.register_buffer(
            'pos', self.get_sinusoid_encoding_table(n_position, n_hiddens))

    def get_sinusoid_encoding_table(self, n_position, n_hiddens):

        def cal_angle(position, hid_idx):
            return position / np.power(10000, 2 * (hid_idx // 2) / n_hiddens)

        def get_position_angle_vec(position):
            return [cal_angle(position, hid_j) for hid_j in range(n_hiddens)]

        sinusoid_table = np.array([get_position_angle_vec(pos_i)
                                  for pos_i in range(n_position)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1
        return torch.FloatTensor(sinusoid_table).unsqueeze(0)

    def forward(self, x):
        #         x = x + self.pos[:, :x.shape[1], :].to(x.device)
        x + self.pos[:, :x.size(1)].clone().detach()
        return self.dropout(x)


def get_attn_pad_mask(seq_q, seq_k):
    batch_size, query_size = seq_q.size()
    batch_size, key_size = seq_k.size()
    # eq(zero) is PAD token
    # batch_size x 1 x len_k(=len_q), one is masking
    pad_attn_mask = seq_k.data.eq(0).unsqueeze(1)
    # batch_size x len_q x len_k
    return pad_attn_mask.expand(batch_size, query_size, key_size)


def get_attn_subsequent_mask(seq):
    ''' For masking out the subsequent info. '''
    sz_b, len_s = seq.size()
    subsequent_mask = (1 - torch.triu(
        torch.ones((1, len_s, len_s), device=seq.device), diagonal=1)).bool()
    return subsequent_mask


class Encoder(torch.nn.Module):
    """The base encoder interface for the encoder-decoder architecture with self attention mechanism."""

    def __init__(self, **kwargs):
        super(Encoder, self).__init__()
        self.src_emb = torch.nn.Embedding(self.vocab_size, self.n_hiddens)
        self.pos_enc = PositionalEncoding(
            self.n_hiddens, self.dropout, self.n_position)
        self.dropout = torch.nn.Dropout(p=self.dropout)
        self.layers = torch.nn.ModuleList([
            layer.EncoderBlock(self.key_size, self.query_size, self.value_size, self.n_heads, self.n_hiddens, self.ffn_d_in, self.ffn_d_hid,
                               self.dropout) for _ in range(self.n_layers)])
        self.layer_norm = torch.nn.LayerNorm(self.n_hiddens, eps=1e-6)

    def forward(self, enc_x, *args):
        #enc_outputs = self.src_emb(enc_x) + self.pos_enc(torch.LongTensor([[1,2,3,4,0]]))
        enc_output = self.src_emb(enc_x)
        enc_output = self.dropout(self.pos_enc(enc_output))
        enc_output = self.layer_norm(enc_output)
        slf_attn_mask = get_attn_pad_mask(enc_x, enc_x)
        enc_slf_attn_list = []
        for enc_layer in self.layers:
            enc_output, enc_slf_attn = enc_layer(enc_output, slf_attn_mask)
            enc_slf_attn_list.append(enc_slf_attn)

        return enc_output, enc_slf_attn_list


class Decoder(torch.nn.Module):
    """The base decoder interface for the encoder-decoder architecture."""

    def __init__(self, **kwargs):
        super(Decoder, self).__init__()
        self.tgt_emb = torch.nn.Embedding(self.tgt_vocab_size, self.n_hiddens)
        self.pos_enc = PositionalEncoding(
            self.n_hiddens, self.dropout, self.n_position)
        self.dropout = torch.nn.Dropout(p=self.dropout)
        self.layers = torch.nn.ModuleList([
            layer.DecoderBlock(self.key_size, self.query_size, self.value_size, self.n_heads, self.n_hiddens, self.ffn_d_in, self.ffn_d_hid,
                               self.dropout) for _ in range(self.n_layers)])
        self.layer_norm = torch.nn.LayerNorm(self.n_hiddens, eps=1e-6)

    def forward(self, dec_x, enc_x, enc_outputs, dec_self_attn_mask=None, dec_enc_attn_mask=None):
       # dec_outputs = self.tgt_emb(dec_x) + self.pos_emb(torch.LongTensor([[5,1,2,3,4]]))
        dec_output = self.tgt_emb(dec_x)
        dec_output = self.dropout(self.pos_enc(dec_output))
        dec_output = self.layer_norm(dec_output)
        dec_self_attn_pad_mask = get_attn_pad_mask(dec_x, dec_x)
        dec_self_attn_subsequent_mask = get_attn_subsequent_mask(dec_x)
        dec_self_attn_mask = torch.gt(
            (dec_self_attn_pad_mask + dec_self_attn_subsequent_mask), 0)
        dec_enc_attn_mask = get_attn_pad_mask(dec_x, enc_x)

        dec_self_attns, dec_enc_attns = [], []
        for dec_layer in self.layers:
            dec_outputs, dec_self_attn, dec_enc_attn = dec_layer(
                dec_output, enc_outputs, dec_self_attn_mask, dec_enc_attn_mask)
            dec_self_attns.append(dec_self_attn)
            dec_enc_attns.append(dec_enc_attn)

        return dec_outputs, dec_self_attns, dec_enc_attns


class Transformer(torch.nn.Module):
    """ A sequence to sequence model with attention mechanism. """

    def __init__(self, **kwargs):
        super(Transformer, self).__init__()
        #TODO while testing add parameters
        self.encoder = Encoder(vocab_size=self.vocab_size, n_position=self.n_position, n_hiddens=self.n_hiddens, ffn_d_in=self.ffn_d_in, ffn_d_hid=self.ffn_d_hid,
                               n_layers=self.n_layers, n_heads=self.n_heads, key_size=self.key_size, query_size=self.query_size, value_size=self.value_size, dropout=self.dropout)

        self.decoder = Decoder(tgt_vocab_size=self.tgt_vocab_size, n_position=self.n_position, n_hiddens=self.n_hiddens,
                               ffn_d_in=self.ffn_d_in, ffn_d_hid=self.ffn_d_hid, n_layers=self.n_layers, n_heads=self.n_heads,
                               key_size=self.key_size, query_size=self.query_size, value_size=self.value_size, dropout=self.dropout)
        self.projection = torch.nn.Linear(
            self.n_hiddens, self.vocab_size, bias=False)

    def forward(self, enc_x, dec_x):
        enc_outputs, enc_self_attns = self.encoder(enc_x)
        dec_outputs, dec_self_attns, dec_enc_attns = self.decoder(
            dec_x, enc_x, enc_outputs)
        dec_logit = self.projection(dec_outputs)
        return dec_logit.view(-1, dec_logit.size(-1)), enc_self_attns, dec_self_attns, dec_enc_attns
