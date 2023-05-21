# Copyright (c) Facebook, Inc. and its affiliates.

import math
from typing import Optional, Tuple, Type

import torch
from mmf.modules.pos_coding import LearnedPositionalEncoding1D, SinePositionalEncoding2D
from mmf.modules.pos_coding import with_pos
from torch import nn
import torch.nn.functional as F


class SelfAttention(nn.Module):
    def __init__(self, dim: int, num_attn: int, dropout: float):
        super().__init__()
        self.multi_head_attn = nn.MultiheadAttention(dim, num_attn, dropout=0.1, batch_first=True)
        self.fcn = nn.Sequential(
            nn.Linear(dim, 4 * dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(4 * dim, dim),
        )
        self.drop_mha = nn.Dropout(p=dropout)
        self.ln_mha = nn.LayerNorm(dim)
        self.drop_fcn = nn.Dropout(p=dropout)
        self.ln_fcn = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor, x_mask: torch.Tensor) -> torch.Tensor:
        att, weight = self.multi_head_attn(x, x, x, attn_mask = x_mask)
        att = self.drop_mha(att)
        x = self.ln_mha(x + att)
        x = self.ln_fcn(x + self.drop_fcn(self.fcn(x)))

        return x, weight


class SelfGuidedAttention(nn.Module):
    def __init__(self, dim: int, num_attn: int, dropout: float):
        super().__init__()
        self.multi_head_attn = nn.ModuleList(
            [nn.MultiheadAttention(dim, num_attn, dropout=0.1, batch_first=True) for _ in range(2)]
        )
        self.fcn = nn.Sequential(
            nn.Linear(dim, 4 * dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(4 * dim, dim),
        )
        self.drop_mha = nn.ModuleList([nn.Dropout(p=dropout) for _ in range(2)])
        self.ln_mha = nn.ModuleList([nn.LayerNorm(dim) for _ in range(3)])
        self.drop_fcn = nn.Dropout(p=dropout)
        self.ln_fcn = nn.LayerNorm(dim)


    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        x_mask: torch.Tensor,
        y_mask: torch.Tensor,
    ) -> torch.Tensor:
        weights = []
        att1, weight_1 = self.multi_head_attn[0](x, x, x, attn_mask = x_mask)
        att1= self.drop_mha[0](att1)
        x = self.ln_mha[0](
            x + att1
        )
        weights.append(weight_1)
        att2, weight_2 = self.multi_head_attn[1](x, y, y, key_padding_mask = y_mask)
        att2= self.drop_mha[1](att2)
        x = self.ln_mha[1](
            x + att2
        )
        x = self.ln_fcn(x + self.drop_fcn(self.fcn(x)))
        weights.append(weight_2)
        return x, weights


class DDTNAttention(nn.Module):
    def __init__(self, dim: int, num_attn: int, dropout: float, batch_first: bool):
        super(FVEAttention, self).__init__()

        self.multi_head_attn = nn.ModuleList([nn.MultiheadAttention(dim, num_attn, dropout=0.1, batch_first= batch_first) for _ in range(2)])
        self.fcn = nn.Sequential(nn.Linear(dim, 4 * dim), nn.ReLU(inplace=True),
                                 nn.Dropout(p=dropout), nn.Linear(4 * dim, dim))
        self.drop_mha = nn.ModuleList([nn.Dropout(p=dropout) for _ in range(2)])
        self.ln_mha = nn.ModuleList([nn.LayerNorm(dim) for _ in range(3)])
        self.drop_fcn = nn.Dropout(p=dropout)
        self.ln_fcn = nn.LayerNorm(dim)

    def forward(
        self,
        x: torch.Tensor,
        p: torch.Tensor,
        p_position: torch.Tensor,
        p_mask: torch.Tensor,
        x_mask: torch.Tensor,
        x_position: torch.Tensor,
        x_shape
    ) -> torch.Tensor:
        h, w = x_shape
        q = k = with_pos(p, p_position)
        att_lists = []
        att1, att_weight_1 = self.multi_head_attn[0](q, k, p, attn_mask=p_mask)
        att1= self.drop_mha[0](att1)
        att_lists.append(att_weight_1)
        p = self.ln_mha[0](p + att1)
        att2, att_weight_2 = self.multi_head_attn[1](query = with_pos(p,p_position), key = with_pos(x, x_position),
                                                        value = x, key_padding_mask= x_mask)
        att2 = self.drop_mha[1](att2)
        att_weight_2 = att_weight_2.unflatten(2,(h,w))
        att_lists.append(att_weight_2)
        p = self.ln_mha[1](
            p + att2)
        p = self.ln_fcn(p + self.drop_fcn(self.fcn(p)))

        return p, att_lists
