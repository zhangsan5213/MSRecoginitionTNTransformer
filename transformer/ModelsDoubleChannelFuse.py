''' Define the Transformer model '''
import random
import torch
import torch.nn as nn
import numpy as np
import transformer.Constants as Constants
from transformer.Layers import EncoderLayer, DecoderLayer, TEncoderLayer, TDecoderLayer

def get_non_pad_mask(seq):
    assert seq.dim() == 2
    return seq.ne(Constants.PAD).type(torch.float).unsqueeze(-1)

def get_sinusoid_encoding_table(n_position, d_hid, padding_idx=None):
    ''' Sinusoid position encoding table '''

    def cal_angle(position, hid_idx):
        return position / np.power(10000, 2 * (hid_idx // 2) / d_hid)

    def get_posi_angle_vec(position):
        return [cal_angle(position, hid_j) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_posi_angle_vec(pos_i) for pos_i in range(n_position)])

    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    if padding_idx is not None:
        # zero vector for padding dimension
        sinusoid_table[padding_idx] = 0.

    return torch.FloatTensor(sinusoid_table)

def get_attn_key_pad_mask(seq_k, seq_q):
    ''' For masking out the padding part of key sequence. '''

    # Expand to fit the shape of key query attention matrix.
    len_q = seq_q.size(1)
    padding_mask = seq_k.eq(Constants.PAD)
    padding_mask = padding_mask.unsqueeze(1).expand(-1, len_q, -1)  # b x lq x lk

    return padding_mask

def get_subsequent_mask(seq):
    ''' For masking out the subsequent info. '''

    sz_b, len_s = seq.size()
    subsequent_mask = torch.triu(
        torch.ones((len_s, len_s), device=seq.device, dtype=torch.uint8), diagonal=1)
    subsequent_mask = subsequent_mask.unsqueeze(0).expand(sz_b, -1, -1)  # b x ls x ls

    return subsequent_mask

# class FuseAttention(nn.Module):
#     ''' An attention fuse layer to fuse mz and intensity channels. '''
#     def __init__(self, _batch_size, _len_max_seq, _n_embedding, _n_channels=512, _n_kernel=2, _n_lin_dim=512):
#         super().__init__()
#         # self.len_max_seq = _len_max_seq
#         # self.conv11 = nn.Conv1d(_len_max_seq, _n_channels, _n_kernel)
#         # self.conv12 = nn.Conv1d(_n_channels, _len_max_seq, _n_kernel)
#         self.lin11 = nn.Linear(_n_embedding-2*(_n_kernel-1), _n_lin_dim)
#         self.lin12 = nn.Linear(_n_lin_dim, _n_embedding)
#         # self.conv21 = nn.Conv1d(_len_max_seq, _n_channels, _n_kernel)
#         # self.conv22 = nn.Conv1d(_n_channels, _len_max_seq, _n_kernel)
#         self.lin21 = nn.Linear(_n_embedding-2*(_n_kernel-1), _n_lin_dim)
#         self.lin22 = nn.Linear(_n_lin_dim, _n_embedding)
#         self.contraction = nn.Parameter(torch.randn(_batch_size, _len_max_seq, _n_embedding,
#                                                     _batch_size, _len_max_seq, _n_embedding,
#                                                     _batch_size, _len_max_seq, _n_embedding))
#         self.leaky_relu = nn.LeakyReLU(negative_slope=0.05)

#     def forward(self, _mz, _ms):
#         ## THE PADDING IS MISSING HERE.
#         # mz = self.leaky_relu(self.conv12(self.conv11(_mz)))
#         mz = self.leaky_relu(self.lin12(self.leaky_relu(self.lin11(_mz))))
#         # ms = self.leaky_relu(self.conv22(self.conv21(_ms)))
#         ms = self.leaky_relu(self.lin22(self.leaky_relu(self.lin21(_ms))))
#         return torch.einsum("abcdefghi,def,ghi->abc", [mz, ms, self.contraction])
        

class TEncoder(nn.Module):
    ''' A encoder model with self attention mechanism. '''

    def __init__(
            self,
            n_src_mz, len_max_seq, d_word_vec,
            n_layers, n_head,d_k, d_v,
            d_model, d_inner, dropout=0.1):

        super().__init__()

        n_position = len_max_seq + 1

        self.src_word_emb = nn.Embedding(
            n_src_mz, d_word_vec, padding_idx=Constants.PAD)

        self.position_enc = nn.Embedding.from_pretrained(
            get_sinusoid_encoding_table(n_position, d_word_vec, padding_idx=0),
            freeze=True)

        self.layer_stack = nn.ModuleList([
            TEncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)])

    def forward(self, src_seq, src_pos, return_attns=False):

        enc_slf_attn_list = []

        # -- Prepare masks
        slf_attn_mask = get_attn_key_pad_mask(seq_k=src_seq, seq_q=src_seq)
        non_pad_mask = get_non_pad_mask(src_seq)

        # -- Forward
        # print(src_seq.shape)
        # print(self.src_word_emb)
        # print(self.position_enc)
        # print(self.src_word_emb(src_seq))
        # print(self.position_enc(src_seq))
        enc_output = self.src_word_emb(src_seq) + self.position_enc(src_pos)
        for enc_layer in self.layer_stack:
            # print(return_attns)
            enc_output, enc_slf_attn = enc_layer(
                enc_output,
                non_pad_mask=non_pad_mask,
                slf_attn_mask=slf_attn_mask)
            if return_attns:
                enc_slf_attn_list += [enc_slf_attn]
        
        if return_attns:
            return enc_output, enc_slf_attn_list

        return enc_output,

class TDecoder(nn.Module):
    ''' A decoder model with self attention mechanism. '''

    def __init__(
            self,
            n_tgt_mz, len_max_seq, d_word_vec,
            n_layers, n_head, d_k, d_v,
            d_model, d_inner, dropout=0.1):

        super().__init__()
        n_position = len_max_seq + 1

        self.tgt_word_emb = nn.Embedding(
            n_tgt_mz, d_word_vec, padding_idx=Constants.PAD)

        self.position_enc = nn.Embedding.from_pretrained(
            get_sinusoid_encoding_table(n_position, d_word_vec, padding_idx=0),
            freeze=True)

        self.layer_stack = nn.ModuleList([
            TDecoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)])

    def forward(self, tgt_seq, tgt_pos, src_seq, enc_output, return_attns=False):

        dec_slf_attn_list, dec_enc_attn_list = [], []

        # -- Prepare masks
        non_pad_mask = get_non_pad_mask(tgt_seq)

        slf_attn_mask_subseq = get_subsequent_mask(tgt_seq)
        slf_attn_mask_keypad = get_attn_key_pad_mask(seq_k=tgt_seq, seq_q=tgt_seq)
        slf_attn_mask = (slf_attn_mask_keypad + slf_attn_mask_subseq).gt(0)

        dec_enc_attn_mask = get_attn_key_pad_mask(seq_k=src_seq, seq_q=tgt_seq)

        # -- Forward
        dec_output = self.tgt_word_emb(tgt_seq) + self.position_enc(tgt_pos)

        for dec_layer in self.layer_stack:
            # print("forward loop")
            # print(dec_output.shape, enc_output.shape)
            dec_output, dec_slf_attn, dec_enc_attn = dec_layer(
                dec_output, enc_output,
                non_pad_mask=non_pad_mask,
                slf_attn_mask=slf_attn_mask,
                dec_enc_attn_mask=dec_enc_attn_mask)

            if return_attns:
                dec_slf_attn_list += [dec_slf_attn]
                dec_enc_attn_list += [dec_enc_attn]

        if return_attns:
            return dec_output, dec_slf_attn_list, dec_enc_attn_list
        return dec_output,


class Tensorized_T(nn.Module):
    ''' A sequence to sequence model with attention mechanism. '''

    def __init__(
            self,
            n_src_mz, n_src_ms, n_tgt_mz, len_max_seq,
            d_word_vec, d_model, d_inner,
            n_encoder_layers, n_decoder_layers, n_head, d_k, d_v, dropout,
            tgt_emb_prj_weight_sharing=True,
            emb_src_tgt_weight_sharing=True):

        super().__init__()

        self.encoder_mz = TEncoder(
            n_src_mz=n_src_mz, len_max_seq=len_max_seq,
            d_word_vec=d_word_vec, d_model=d_model, d_inner=d_inner,
            n_layers=n_encoder_layers, n_head=n_head,d_k=d_k, d_v=d_v,
            dropout=dropout)
        self.encoder_ms = TEncoder(
            n_src_mz=n_src_ms, len_max_seq=len_max_seq,
            d_word_vec=d_word_vec, d_model=d_model, d_inner=d_inner,
            n_layers=n_encoder_layers, n_head=n_head,d_k=d_k, d_v=d_v,
            dropout=dropout)
        self.src_seq_merge_lin0 = nn.Linear(2*d_model, d_model)
        self.src_seq_merge_lin1 = nn.Linear(d_model, 1)

        self.encoder_merge_lin0 = nn.Linear(2*d_model, 4*d_model)
        self.encoder_merge_lin1 = nn.Linear(4*d_model, d_model)
        self.activation = nn.LeakyReLU(negative_slope=0.1)

        self.decoder = TDecoder(
            n_tgt_mz=n_tgt_mz, len_max_seq=len_max_seq,
            d_word_vec=d_word_vec, d_model=d_model, d_inner=d_inner,
            n_layers=n_decoder_layers, n_head=n_head,d_k=d_k, d_v=d_v,
            dropout=dropout)

        self.tgt_word_prj = nn.Linear(d_model, n_tgt_mz, bias=False)
        nn.init.xavier_normal_(self.tgt_word_prj.weight)

        assert d_model == d_word_vec, \
        'To facilitate the residual connections, \
         the dimensions of all module outputs shall be the same.'

        if tgt_emb_prj_weight_sharing:
            # Share the weight matrix between target word embedding & the final logit dense layer
            self.tgt_word_prj.weight = self.decoder.tgt_word_emb.weight
            self.x_logit_scale = (d_model ** -0.5)
        else:
            self.x_logit_scale = 1.

        if emb_src_tgt_weight_sharing:
            # Share the weight matrix between source & target word embeddings
            assert n_src_mz == n_tgt_mz, \
            "To share word embedding table, the vocabulary size of src/tgt shall be the same."
            self.encoder.src_word_emb.weight = self.decoder.tgt_word_emb.weight

    def forward(self, src_mz_seq, src_mz_pos, src_ms_seq, src_ms_pos, tgt_seq, tgt_pos):

        tgt_seq, tgt_pos = tgt_seq[:, :-1], tgt_pos[:, :-1]

        enc_output_mz, *_ = self.encoder_mz(src_mz_seq, src_mz_pos)
        enc_output_ms, *_ = self.encoder_ms(src_ms_seq, src_ms_pos)
        enc_output = self.activation(self.encoder_merge_lin0(torch.cat([enc_output_mz, enc_output_ms], dim=2)))
        enc_output = self.activation(self.encoder_merge_lin1(enc_output))

        src_mz_embedding = self.encoder_mz.src_word_emb(src_mz_seq) + self.encoder_mz.position_enc(src_mz_pos)
        src_ms_embedding = self.encoder_ms.src_word_emb(src_ms_seq) + self.encoder_ms.position_enc(src_ms_pos)
        src_seq = self.activation(self.src_seq_merge_lin0(torch.cat([src_mz_embedding, src_ms_embedding], dim=2)))
        src_seq = self.activation(self.src_seq_merge_lin1(src_seq)).view(src_seq.shape[0], -1)

        dec_output, *_ = self.decoder(tgt_seq, tgt_pos, src_seq, enc_output)
        seq_logit = self.tgt_word_prj(dec_output) * self.x_logit_scale

        return seq_logit.view(-1, seq_logit.size(2))