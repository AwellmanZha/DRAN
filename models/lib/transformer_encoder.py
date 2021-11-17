from torch import nn
from torch.nn import functional as F
from .utils import PositionWiseFeedForward
import torch
from .attention import MultiHeadAttention, MultiHeadBoxAttention
from .positional_encoding import BoxRelationalEmbedding


class EncoderLayer(nn.Module):
    def __init__(self, d_model=1024, d_k=64, d_v=64, h=8, d_ff=2048, dropout=.2, identity_map_reordering=False,
                 attention_module=None, attention_module_kwargs=None):
        super(EncoderLayer, self).__init__()
        self.identity_map_reordering = identity_map_reordering
        self.mhatt = MultiHeadAttention(d_model, d_k, d_v, h, dropout, identity_map_reordering=identity_map_reordering,
                                        attention_module=attention_module,
                                        attention_module_kwargs=attention_module_kwargs)
        self.pwff = PositionWiseFeedForward(d_model, d_ff, dropout, identity_map_reordering=identity_map_reordering)

    def forward(self, queries, keys, values, attention_mask=None, attention_weights=None):
        att = self.mhatt(queries, keys, values, attention_mask, attention_weights)
        ff = self.pwff(att)
        return ff


class EncoderLayerWithBox(nn.Module):
    def __init__(self, d_model=1024, d_k=64, d_v=64, h=8, d_ff=2048, dropout=.1, identity_map_reordering=False,
                 attention_module=None, attention_module_kwargs=None):
        super(EncoderLayerWithBox, self).__init__()
        self.identity_map_reordering = identity_map_reordering
        self.mhatt = MultiHeadBoxAttention(d_model, d_k, d_v, h, dropout, identity_map_reordering=identity_map_reordering,
                                        attention_module=attention_module,
                                        attention_module_kwargs=attention_module_kwargs)
        self.dropout = nn.Dropout(dropout)
        self.lnorm = nn.LayerNorm(d_model)
        self.pwff = PositionWiseFeedForward(d_model, d_ff, dropout, identity_map_reordering=identity_map_reordering)

    def forward(self, queries, keys, values, relative_geometry_weights, attention_mask=None, attention_weights=None,
                pos=None):
        # print('-' * 50)
        # print('layer input')
        # print(queries[11])
        q = queries + pos
        k = keys + pos
        att = self.mhatt(q, k, values, relative_geometry_weights, attention_mask, attention_weights)
        # print('mhatt outpout')
        # print(att[11])
        att = self.lnorm(queries + self.dropout(att))
        # print('norm out')
        # print(att[11])
        ff = self.pwff(att)
        # print('ff out')
        # print(ff[11])
        # print('-' * 50)
        return ff


class MutliLevelEncoder(nn.Module):
    def __init__(self, N, padding_idx, d_model=1024, d_k=64, d_v=64, h=8, d_ff=2048, dropout=.1,
                 identity_map_reordering=False, attention_module=None, attention_module_kwargs=None):
        super(MutliLevelEncoder, self).__init__()
        self.d_model = d_model
        self.dropout = dropout
        self.layers = nn.ModuleList([EncoderLayer(d_model, d_k, d_v, h, d_ff, dropout,
                                                  identity_map_reordering=identity_map_reordering,
                                                  attention_module=attention_module,
                                                  attention_module_kwargs=attention_module_kwargs)
                                     for _ in range(N)])
        self.padding_idx = padding_idx

    def forward(self, input, attention_weights=None):
        # input (b_s, seq_len, d_in)
        attention_mask = (torch.sum(input, -1) == self.padding_idx).unsqueeze(1).unsqueeze(1)  # (b_s, 1, 1, seq_len)

        out = input
        for l in self.layers:
            residual = out
            out = l(out, out, out, attention_mask, attention_weights)
            out = out + residual
        return out, attention_mask


class MutliLevelEncoderWithBox(nn.Module):
    def __init__(self, N, padding_idx, d_model=1024, d_k=64, d_v=64, h=8, d_ff=2048, dropout=.1,
                 identity_map_reordering=False, attention_module=None, attention_module_kwargs=None):
        super(MutliLevelEncoderWithBox, self).__init__()
        self.d_model = d_model
        self.dropout = dropout
        self.layers = nn.ModuleList([EncoderLayerWithBox(d_model, d_k, d_v, h, d_ff, dropout,
                                                  identity_map_reordering=identity_map_reordering,
                                                  attention_module=attention_module,
                                                  attention_module_kwargs=attention_module_kwargs)
                                     for _ in range(N)])
        self.padding_idx = padding_idx

        self.WGs = nn.ModuleList([nn.Linear(64, 1, bias=True) for _ in range(h)])

    def forward(self, regions, boxes, attention_weights=None, region_embed=None):
        # input (b_s, seq_len, d_in)
        attention_mask = (torch.sum(regions, -1) == self.padding_idx).unsqueeze(1).unsqueeze(1)  # (b_s, 1, 1, seq_len)

        # box embedding
        relative_geometry_embeddings = BoxRelationalEmbedding(boxes)
        flatten_relative_geometry_embeddings = relative_geometry_embeddings.view(-1, 64)
        box_size_per_head = list(relative_geometry_embeddings.shape[:3])
        box_size_per_head.insert(1, 1)
        relative_geometry_weights_per_head = [l(flatten_relative_geometry_embeddings).view(box_size_per_head) for l in
                                              self.WGs]
        relative_geometry_weights = torch.cat((relative_geometry_weights_per_head), 1)
        relative_geometry_weights = F.relu(relative_geometry_weights)

        n_regions = regions.shape[1]  # 50

        region2region = relative_geometry_weights[:, :, :n_regions, :n_regions]

        out = regions
        for l in self.layers:
            residual = out
            out = l(out, out, out, region2region, attention_mask, attention_weights, pos=region_embed)
            out = out + residual
        return out, attention_mask


class TransformerEncoder(MutliLevelEncoder):
    def __init__(self, N, padding_idx, d_in=1024, **kwargs):
        super(TransformerEncoder, self).__init__(N, padding_idx, **kwargs)
        self.fc = nn.Linear(d_in, self.d_model)
        self.dropout = nn.Dropout(p=self.dropout)
        self.layer_norm = nn.LayerNorm(self.d_model)

    def forward(self, input, attention_weights=None):
        out = F.relu(self.fc(input))
        out = self.dropout(out)
        out = self.layer_norm(out)
        return super(TransformerEncoder, self).forward(out, attention_weights=attention_weights)




class TransformerEncoderWithBox(MutliLevelEncoderWithBox):
    def __init__(self, N, padding_idx, d_in=1024, **kwargs):
        super(TransformerEncoderWithBox, self).__init__(N, padding_idx, **kwargs)
        self.fc = nn.Linear(d_in, self.d_model)
        self.dropout = nn.Dropout(p=self.dropout)
        self.layer_norm = nn.LayerNorm(self.d_model)

    def forward(self, regions, boxes, attention_weights=None, region_embed=None):
        # MyNote: No mask needed here in my program, cause calculated in later
        # mask_regions = (torch.sum(regions, dim=-1) == 0).unsqueeze(-1)
        # print('\ninput', input.view(-1)[0].item())
        out_region = F.relu(self.fc(regions))
        out_region = self.dropout(out_region)
        out_region = self.layer_norm(out_region)
        # out_region = out_region.masked_fill(mask_regions, 0)
        # print('out4',out[11])
        return super(TransformerEncoderWithBox, self).forward(out_region, boxes,
                                                       attention_weights=attention_weights,
                                                       region_embed=region_embed)
