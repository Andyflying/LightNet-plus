import torch
import torch.nn as nn
from layers.PositionEmbedding import PositionEmbeddingSine


class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=256, dropout=0.0, normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.dec_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = nn.ReLU()
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward(self, src, enc, pos):
        q = k = self.with_pos_embed(src, pos)
        # self attention
        src2 = self.self_attn(query=q, key=k, value=src, attn_mask=None, key_padding_mask=None)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        # decoder attention
        src2 = self.dec_attn(query=src, key=enc, value=enc, attn_mask=None, key_padding_mask=None)[0]
        src = src + self.dropout2(src2)
        src = self.norm2(src)

        # feed forward
        # src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        # src = src + self.dropout3(src2)
        # src = self.norm3(src)
        return src


class TransformerDecoder(nn.Module):
    def __init__(self, channels, layers, nhead, is_posemb=False):
        super().__init__()
        self.self_attn_stack = nn.ModuleList([TransformerDecoderLayer(d_model=channels, nhead=nhead) for _ in range(layers)])
        self.posemb = PositionEmbeddingSine(num_pos_feats=channels // 2)
        self.is_posemb = is_posemb

    def forward(self, src, enc):
        batch_size, channels, x, y = src.shape
        pos = None
        if self.is_posemb:
            pos = self.posemb(src)
            pos = pos.flatten(2).permute(2, 0, 1)
        src = src.flatten(2).permute(2, 0, 1)
        enc = enc.flatten(2).permute(2, 0, 1)
        for layer in self.self_attn_stack:
            src = layer(src, enc, pos)
        src = src.reshape([x, y, batch_size, channels]).permute(2, 3, 0, 1)
        return src


if __name__ == '__main__':
    batchsize = 1
    channels = 32
    wrf = torch.zeros([batchsize, channels, 39, 39])
    h = torch.zeros([batchsize, channels, 39, 39])
    model = TransformerDecoder(channels=channels, layers=1, nhead=8)
    print('# generator parameters:', sum(param.numel() for param in model.parameters()))
    # y = model(wrf, h)
    # print(y.shape)
    model2 = TransformerDecoderLayer(d_model=channels, nhead=8)
    print('# generator parameters:', sum(param.numel() for param in model2.linear1.parameters()))

