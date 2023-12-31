''' Define the Layers '''
import torch.nn as nn
from transformer.SubLayers import MultiHeadAttention, PositionwiseFeedForward, MultiLinearAttention


class EncoderLayer(nn.Module):
    ''' Compose with two layers '''

    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(
            n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)

    def forward(self, enc_input, non_pad_mask=None, slf_attn_mask=None):
        enc_output, enc_slf_attn = self.slf_attn(
            enc_input, enc_input, enc_input, mask=slf_attn_mask)
        enc_output *= non_pad_mask

        enc_output = self.pos_ffn(enc_output)
        enc_output *= non_pad_mask

        return enc_output, enc_slf_attn


class DecoderLayer(nn.Module):
    ''' Compose with three layers '''

    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.enc_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)

    def forward(self, dec_input, enc_output, non_pad_mask=None, slf_attn_mask=None, dec_enc_attn_mask=None):
        dec_output, dec_slf_attn = self.slf_attn(
            dec_input, dec_input, dec_input, mask=slf_attn_mask)
        dec_output *= non_pad_mask

        dec_output, dec_enc_attn = self.enc_attn(
            dec_output, enc_output, enc_output, mask=dec_enc_attn_mask)
        dec_output *= non_pad_mask

        dec_output = self.pos_ffn(dec_output)
        dec_output *= non_pad_mask

        return dec_output, dec_slf_attn, dec_enc_attn

class TEncoderLayer(nn.Module):
    ''' Compose with two layers '''

    def __init__(self, d_model, d_inner,n_head,d_k, d_v, dropout=0.1):
        super(TEncoderLayer, self).__init__()
        self.slf_attn = MultiLinearAttention(
            n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)

    def forward(self, enc_input, non_pad_mask=None, slf_attn_mask=None):
        enc_output,enc_slf_attn = self.slf_attn(
            enc_input, enc_input, enc_input, mask=slf_attn_mask)
        enc_output *= non_pad_mask

        enc_output = self.pos_ffn(enc_output)
        enc_output *= non_pad_mask

        return enc_output, enc_slf_attn


class TDecoderLayer(nn.Module):
    ''' Compose with three layers '''

    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        super(TDecoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.enc_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)

    def forward(self, dec_input, enc_output, non_pad_mask=None, slf_attn_mask=None, dec_enc_attn_mask=None):
        # print("TDecoderLayer forward slf_attn")
        # print(dec_input.shape, dec_input.shape, dec_input.shape)
        dec_output, dec_slf_attn = self.slf_attn(
            dec_input, dec_input, dec_input, mask=slf_attn_mask)
        dec_output *= non_pad_mask

        # print("TDecoderLayer forward enc_attn")
        # print(dec_output.shape, enc_output.shape, enc_output.shape)
        dec_output, dec_enc_attn= self.enc_attn(
            dec_output, enc_output, enc_output, mask=dec_enc_attn_mask)
        dec_output *= non_pad_mask

        dec_output = self.pos_ffn(dec_output)
        dec_output *= non_pad_mask

        return dec_output, dec_slf_attn, dec_enc_attn

class TDecoderLayerDoubleChannelFuse(nn.Module):
    ''' Compose with three layers '''

    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        super(TDecoderLayerDoubleChannelFuse, self).__init__()
        self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.enc_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)
        self.merge = nn.Linear(d_model*2, d_model) ## stack the channels together and Linear them
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.125)

    def forward(self, dec_input_0, enc_output_0, dec_input_1, enc_output_1, non_pad_mask=None, slf_attn_mask=None, dec_enc_attn_mask=None):
        dec_output_0, dec_slf_attn_0 = self.slf_attn(
            dec_input_0, dec_input_0, dec_input_0, mask=slf_attn_mask)
        dec_output_0 *= non_pad_mask
        dec_output_0, dec_enc_attn_0= self.enc_attn(
            dec_output_0, enc_output_0, enc_output_0, mask=dec_enc_attn_mask)
        dec_output_0 *= non_pad_mask
        dec_output_0 = self.pos_ffn(dec_output_0)
        dec_output_0 *= non_pad_mask

        dec_output_1, dec_slf_attn_1 = self.slf_attn(
            dec_input_1, dec_input_1, dec_input_1, mask=slf_attn_mask)
        dec_output_1 *= non_pad_mask
        dec_output_1, dec_enc_attn_1= self.enc_attn(
            dec_output_1, enc_output_1, enc_output_1, mask=dec_enc_attn_mask)
        dec_output_1 *= non_pad_mask
        dec_output_1 = self.pos_ffn(dec_output_1)
        dec_output_1 *= non_pad_mask

        dec_output = self.leaky_relu(self.merge(torch.cat([dec_output_0, dec_output_1], dim=2)))

        return dec_output