# Copyright (c) Facebook, Inc. and its affiliates.
# TODO: Update kwargs with defaults

import os
import math
import pickle
from copy import deepcopy
from functools import lru_cache
from typing import Optional, Tuple
import torch.nn.functional as F

import numpy as np
import torch
from mmf.modules.attention import  SelfAttention, SelfGuidedAttention, DDTNAttention
from mmf.modules.layers import AttnPool1d, Identity
from mmf.utils.file_io import PathManager
from mmf.utils.vocab import Vocab
from torch import Tensor, nn
from mmf.modules.pos_coding import LearnedPositionalEncoding1D, SinePositionalEncoding2D


class TextEmbedding(nn.Module):
    def __init__(self, emb_type, **kwargs):
        super().__init__()
        self.model_data_dir = kwargs.get("model_data_dir", None)
        self.embedding_dim = kwargs.get("embedding_dim", None)

        # Update kwargs here
        if emb_type == "identity":
            self.module = Identity()
            self.module.text_out_dim = self.embedding_dim
        elif emb_type == "vocab":
            self.module = VocabEmbedding(**kwargs)
            self.module.text_out_dim = self.embedding_dim
        elif emb_type == "projection":
            self.module = ProjectionEmbedding(**kwargs)
            self.module.text_out_dim = self.module.out_dim
        elif emb_type == "preextracted":
            self.module = PreExtractedEmbedding(**kwargs)
        elif emb_type == "bilstm":
            self.module = BiLSTMTextEmbedding(**kwargs)
        elif emb_type == "attention":
            self.module = AttentionTextEmbedding(**kwargs)
        elif emb_type == "mcan":
            self.module = SAEmbedding(**kwargs)
        elif emb_type == "torch":
            vocab_size = kwargs["vocab_size"]
            embedding_dim = kwargs["embedding_dim"]
            self.module = nn.Embedding(vocab_size, embedding_dim)
            self.module.text_out_dim = self.embedding_dim
        else:
            raise NotImplementedError("Unknown question embedding '%s'" % emb_type)

        self.text_out_dim = self.module.text_out_dim

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)


class VocabEmbedding(nn.Module):
    def __init__(self, embedding_dim, **vocab_params):
        super().__init__()
        self.vocab = Vocab(**vocab_params)
        self.module = self.vocab.get_embedding(
            nn.Embedding, embedding_dim=embedding_dim
        )

    def forward(self, x):
        return self.module(x)


class BiLSTMTextEmbedding(nn.Module):
    def __init__(
        self,
        hidden_dim,
        embedding_dim,
        num_layers,
        dropout,
        bidirectional=False,
        rnn_type="GRU",
    ):
        super().__init__()
        self.text_out_dim = hidden_dim
        self.bidirectional = bidirectional

        if rnn_type == "LSTM":
            rnn_cls = nn.LSTM
        elif rnn_type == "GRU":
            rnn_cls = nn.GRU

        self.recurrent_encoder = rnn_cls(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=bidirectional,
            batch_first=True,
        )

    def forward(self, x):
        out, _ = self.recurrent_encoder(x)
        # Return last state
        if self.bidirectional:
            return out[:, -1]

        forward_ = out[:, -1, : self.num_hid]
        backward = out[:, 0, self.num_hid :]
        return torch.cat((forward_, backward), dim=1)

    def forward_all(self, x):
        output, _ = self.recurrent_encoder(x)
        return output


class PreExtractedEmbedding(nn.Module):
    def __init__(self, out_dim, base_path):
        super().__init__()
        self.text_out_dim = out_dim
        self.base_path = base_path
        self.cache = {}

    def forward(self, qids):
        embeddings = []
        for qid in qids:
            embeddings.append(self.get_item(qid))
        return torch.stack(embeddings, dim=0)

    @lru_cache(maxsize=5000)
    def get_item(self, qid):
        return np.load(os.path.join(self.base_path, str(qid.item()) + ".npy"))


class AttentionTextEmbedding(nn.Module):
    def __init__(self, hidden_dim, embedding_dim, num_layers, dropout, **kwargs):
        super().__init__()

        self.text_out_dim = hidden_dim * kwargs["conv2_out"]

        bidirectional = kwargs.get("bidirectional", False)

        self.recurrent_unit = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim // 2 if bidirectional else hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
        )

        self.dropout = nn.Dropout(p=dropout)

        conv1_out = kwargs["conv1_out"]
        conv2_out = kwargs["conv2_out"]
        kernel_size = kwargs["kernel_size"]
        padding = kwargs["padding"]

        self.conv1 = nn.Conv1d(
            in_channels=hidden_dim,
            out_channels=conv1_out,
            kernel_size=kernel_size,
            padding=padding,
        )

        self.conv2 = nn.Conv1d(
            in_channels=conv1_out,
            out_channels=conv2_out,
            kernel_size=kernel_size,
            padding=padding,
        )

        self.relu = nn.ReLU()

    def forward(self, x):
        batch_size = x.size(0)

        self.recurrent_unit.flatten_parameters()
        lstm_out, _ = self.recurrent_unit(x)
        lstm_drop = self.dropout(lstm_out)
        lstm_reshape = lstm_drop.permute(0, 2, 1)

        qatt_conv1 = self.conv1(lstm_reshape)
        qatt_relu = self.relu(qatt_conv1)
        qatt_conv2 = self.conv2(qatt_relu)

        qtt_softmax = nn.functional.softmax(qatt_conv2, dim=2)
        qtt_feature = torch.bmm(qtt_softmax, lstm_drop)
        qtt_feature_concat = qtt_feature.view(batch_size, -1)

        return qtt_feature_concat


class ProjectionEmbedding(nn.Module):
    def __init__(self, module, in_dim, out_dim, **kwargs):
        super().__init__()
        if module == "linear":
            self.layers = nn.Linear(in_dim, out_dim)
            self.out_dim = out_dim
        elif module == "conv":
            last_out_channels = in_dim
            layers = []
            for conv in kwargs["convs"]:
                layers.append(nn.Conv1d(in_channels=last_out_channels, **conv))
                last_out_channels = conv["out_channels"]
            self.layers = nn.ModuleList(*layers)
            self.out_dim = last_out_channels
        else:
            raise TypeError(
                "Unknown module type for 'ProjectionEmbedding',"
                "use either 'linear' or 'conv'"
            )

    def forward(self, x):
        return self.layers(x)

class MultiHeadImageFeatureEmbedding(nn.Module):
    def __init__(self, img_dim, question_dim, **kwargs):
        super().__init__()
        self.module = nn.MultiheadAttention(
            embed_dim=question_dim, kdim=img_dim, vdim=img_dim, **kwargs
        )
        self.out_dim = question_dim

    def forward(self, image_feat_variable, question_embedding, image_dims, extra=None):
        if extra is None:
            extra = {}
        image_feat_variable = image_feat_variable.transpose(0, 1)
        question_embedding = question_embedding.unsqueeze(1).transpose(0, 1)
        output, weights = self.module(
            question_embedding, image_feat_variable, image_feat_variable
        )
        output = output.transpose(0, 1)

        return output.squeeze(), weights


class ImageFinetune(nn.Module):
    def __init__(self, in_dim, weights_file, bias_file):
        super().__init__()
        with PathManager.open(weights_file, "rb") as w:
            weights = pickle.load(w)
        with PathManager.open(bias_file, "rb") as b:
            bias = pickle.load(b)
        out_dim = bias.shape[0]

        self.lc = nn.Linear(in_dim, out_dim)
        self.lc.weight.data.copy_(torch.from_numpy(weights))
        self.lc.bias.data.copy_(torch.from_numpy(bias))
        self.out_dim = out_dim

    def forward(self, image):
        i2 = self.lc(image)
        i3 = nn.functional.relu(i2)
        return i3

class SAEmbedding(nn.Module):
    """Encoder block implementation in MCAN https://arxiv.org/abs/1906.10770"""

    def __init__(self, hidden_dim: int, embedding_dim: int, **kwargs):
        super().__init__()
        num_attn = kwargs["num_attn"]
        num_layers = kwargs["num_layers"]
        dropout = kwargs.get("dropout", 0.1)
        num_attn_pool = kwargs.get("num_attn_pool", 1)
        num_feat = kwargs.get("num_feat", -1)

        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
        )
        self.self_attns = nn.ModuleList(
            [SelfAttention(hidden_dim, num_attn, dropout) for _ in range(num_layers)]
        )
        self.attn_pool = None
        self.num_feat = num_feat
        self.text_out_dim = hidden_dim
        if num_attn_pool > 0:
            self.attn_pool = AttnPool1d(hidden_dim, num_feat * num_attn_pool)
            self.text_out_dim = hidden_dim * num_attn_pool

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        b = x.size(0)
        out, (h, c) = self.lstm(x)
        att_list = []
        for self_attn in self.self_attns:
            out, att = self_attn(out, mask)
            att_list.append(att)

        vec = h.transpose(0, 1).contiguous().view(b, 1, -1)
        if self.attn_pool:
            vec = self.attn_pool(out, out, mask).view(b, self.num_feat, -1)

        return out, vec, att_list

class SGAEmbedding(nn.Module):
    """Decoder block implementation in MCAN https://arxiv.org/abs/1906.10770"""

    def __init__(self, embedding_dim: int, **kwargs):
        super().__init__()
        num_attn = kwargs["num_attn"]
        num_layers = kwargs["num_layers"]
        dropout = kwargs.get("dropout", 0.1)
        hidden_dim = kwargs.get("hidden_dim", 512)

        self.linear = nn.Linear(embedding_dim, hidden_dim)
        self.self_guided_attns = nn.ModuleList(
            [
                SelfGuidedAttention(hidden_dim, num_attn, dropout)
                for _ in range(num_layers)
            ]
        )
        self.out_dim = hidden_dim
    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        x_mask: torch.Tensor,
        y_mask: torch.Tensor,
    ) -> torch.Tensor:

        if x.dim() == 4:
            b, c, h, w = x.shape
            x = x.view(b, c, -1).transpose(1, 2).contiguous()

        x = self.linear(x)

        att_list= []
        for self_guided_attn in self.self_guided_attns:
            x, att = self_guided_attn(x, y, x_mask, y_mask)
            att_list.append(att)
        return x, att_list


class FVEmbedding(nn.Module):
    def __init__(self, embedding_dim: int, **kwargs):
        super().__init__()
        self.num_bin = kwargs['num_bin']
        self.num_ray = kwargs['num_ray']
        self.mapping = kwargs['mapping']
        num_layers = kwargs['fve_num_layers']
        batch_first = kwargs['batch_first']
        self.hidden_dim = kwargs['hidden_dim']
        num_attn = kwargs["num_attn"]
        dropout = kwargs.get("dropout", 0.1)

        self.out_dim = 2 * self.num_ray + 1
        self.query_embedding = nn.Embedding(2 * self.num_ray, self.hidden_dim)
        self.position_coding = SinePositionalEncoding2D(num_feature=self.hidden_dim//2)
        self.location_coding = LearnedPositionalEncoding1D(num_embedding=self.out_dim, num_feature= self.hidden_dim)

        self.Conv2d = nn.Conv2d(embedding_dim, self.hidden_dim, kernel_size=1)
        self.activate = nn.Tanh()
        self.fve_attns = nn.ModuleList(
            [
                DDTNAttention(self.hidden_dim, num_attn, dropout, batch_first)
                for _ in range(num_layers)
            ]
        )

    def sequentialize(self, p, pad_shape):
        seq_in_embeds = self.quantize(p, pad_shape)
        seq_in_embeds = seq_in_embeds.clamp_(min=0, max=self.num_bin)
        seq_in_embeds = self.query_embedding(seq_in_embeds)

        return seq_in_embeds

    def quantize(self, seq, pad_shapes):
        if self.mapping == "relative":
            num_pts = seq.size(1) // 2
            norm_factor = [pad_shape[:2][::-1] for pad_shape in pad_shapes]
            norm_factor = seq.new_tensor(norm_factor)

            norm_factor = norm_factor.repeat(1, num_pts)
            return (seq / norm_factor * self.num_bin).long()
        elif self.mapping == "absolute":
            return (seq / 640. * self.num_bin).long()

    def tri_mask(self, length):
        mask = (torch.triu(torch.ones(length, length))
                == 1).float().transpose(0, 1)
        mask.masked_fill_(mask == 0, float('-inf'))
        mask.masked_fill_(mask == 1, float(0.))
        return mask

    def forward(self, x: torch.Tensor, v: torch.bool,
                x_mask:torch.Tensor) -> torch.Tensor:
        if len(x.shape) == 4:
            b, _, h, w = x.shape
            v = v.squeeze().unsqueeze(-1).unsqueeze(-1)
            x = self.Conv2d(x)
            x = self.activate(x) * self.activate(v)
            x_mask = F.interpolate(x_mask.unsqueeze(
                1).float(), size=x.size()[-2:]).to(torch.bool).squeeze(1)
            x_position = self.position_coding(x_mask)
            x_mask = x_mask.view(b, -1)
            x_position = x_position.flatten(2).transpose(1, 2)
            x = x.flatten(2).transpose(1, 2)
            p = self.query_embedding.weight.unsqueeze(0).repeat(b, 1, 1)
            p_position = self.location_coding(p)

            att_list = []
            for fve_attn in self.fve_attns:
                p, att = fve_attn(x, p, p_position, None, x_mask, x_position, (h, w))
                att_list.append(att)
            return p, att_list

        else:
            b, n, c = x.shape
            v = v.squeeze().unsqueeze(1)
            x = self.linear(x)
            x = self.activate(x) * self.activate(v)
            p = self.query_embedding.weight.unsqueeze(0).repeat(b, 1, 1)
            p_position = self.location_coding(p)

            att_list = []
            for fve_attn in self.fve_attns:
                p, att = fve_attn(x, p, p_position, None, x_mask, None, None)
                att_list.append(att)
            return p, att_list


class TwoBranchEmbedding(nn.Module):
    """Attach MoVie into MCAN model as a counting module in
    https://arxiv.org/abs/2004.11883
    """

    def __init__(self, embedding_dim: int, **kwargs):
        super().__init__()
        hidden_dim = kwargs.get("hidden_dim", 512)
        self.sga = SGAEmbedding(embedding_dim, **kwargs)
        self.sga_pool = AttnPool1d(hidden_dim, 1)
        self.out_dim = hidden_dim

    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        x_mask: torch.Tensor,
        y_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x_sga = self.sga(x, y, x_mask, y_mask)
        x_sga = self.sga_pool(x_sga, x_sga, x_mask).squeeze(1)
        return x_sga


class VqaLocEmbedding(nn.Module):
    def __init__(self, embedding_dim, **kwargs):
        super().__init__()

        hidden_dim = kwargs.get("hidden_dim", 512)
        self.sga = SGAEmbedding(embedding_dim, **kwargs)
        self.sga_pool = AttnPool1d(hidden_dim, 1)
        self.fve = FVEmbedding(embedding_dim, **kwargs)
        self.out_dim = hidden_dim

    def forward(
            self,
            x: torch.Tensor,
            grid: torch.Tensor,
            grid_mask: torch.Tensor,
            y: torch.Tensor,
            v: torch.Tensor,
            x_mask: torch.Tensor,
            y_mask: torch.Tensor,
            pad_shape: list
            ):

        x_fve, fve_att_list = self.fve(grid, v, grid_mask, y_mask, pad_shape)
        x_sga, sga_att_list= self.sga(x, y, x_mask, y_mask)
        x_sga = self.sga_pool(x_sga, x_sga, None).squeeze(1)

        return (x_sga, x_fve), (fve_att_list, sga_att_list)
