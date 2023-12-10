


import torch
import torch.nn as nn
import torch.nn.functional as F
import math

import torch
import torch.nn as nn
import math

class FMRI_Transformer(nn.Module):
    def __init__(self, ntoken, ninp, nhead, nhid, nlayers, out_size, dropout=0.5):
        super(FMRI_Transformer, self).__init__()
        from torch.nn import TransformerDecoder, TransformerDecoderLayer
        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(ninp, dropout)
        decoder_layers = TransformerDecoderLayer(ninp, nhead, nhid, dropout)
        self.transformer_decoder = TransformerDecoder(decoder_layers, nlayers)
        self.encoder = nn.Linear(ntoken, ninp)  # Note: Changed to Linear for fMRI data
        self.ninp = ninp
        self.decoder = nn.Linear(ninp, out_size)  # Assuming a single output for disease/cognition measure

        self.init_weights()

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src):
        src = self.encoder(src) * math.sqrt(self.ninp)
        src = self.pos_encoder(src)
        memory = torch.zeros_like(src)  # Replace this with actual memory if using encoder-decoder architecture
        mask = self.generate_square_subsequent_mask(src.size(0))
        output = self.transformer_decoder(src, memory, tgt_mask=mask)
        output = torch.sigmoid(self.decoder(output))
        return output


# Positional Encoding to inject some information about the relative or absolute position of the tokens in the sequence.
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model).unsqueeze(1)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)



