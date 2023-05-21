import torch
import torch.nn as nn

from typing import Optional
import math
from torch import Tensor


def with_pos(tensor, pos: Optional[Tensor]):
    return tensor if pos is None else tensor + pos



class LearnedPositionalEncoding1D(nn.Module):
    """1D Position embedding with learnable embedding weights.

    Args:
        num_feature (int): The feature dimension for each position.
        num_embedding (int, optional): The dictionary size of embeddings.
            Default 5.
    """

    def __init__(self,
                 num_embedding=5,
                 num_feature=256,
                 padding_idx=-1,
                 ):
        super(LearnedPositionalEncoding1D, self).__init__()
        self.num_feature = num_feature

        self.num_embedding = num_embedding
        self.embedding = nn.Embedding(
            num_embedding, num_feature, padding_idx=padding_idx if padding_idx >= 0 else None)

    def forward(self, seq_in_embeds):
        """
        Args:
            seq_in_embeds (tensor): [bs, 5/num_ray*2+1, d_model].

        Returns:
            seq_in_pos_embeds (tensor): [bs, 5/num_ray*2+1, d_model].
        """
        seq_len = seq_in_embeds.size(1)
        position = torch.arange(seq_len, dtype=torch.long,
                                device=seq_in_embeds.device)
        position = position.unsqueeze(0).expand(seq_in_embeds.size()[:2])
        return self.embedding(position)


class SinePositionalEncoding2D(nn.Module):
    """Position encoding with sine and cosine functions.
    See `End-to-End Object Detection with Transformers
    <https://arxiv.org/pdf/2005.12872>`_ for details.
    Args:
        num_feats (int): The feature dimension for each position
            along x-axis or y-axis. Note the final returned dimension
            for each position is 2 times of this value.
        temperature (int, optional): The temperature used for scaling
            the position embedding. Defaults to 10000.
        normalize (bool, optional): Whether to normalize the position
            embedding. Defaults to False.
        scale (float, optional): A scale factor that scales the position
            embedding. The scale will be used only when `normalize` is True.
            Defaults to 2*pi.
        eps (float, optional): A value added to the denominator for
            numerical stability. Defaults to 1e-6.
        offset (float): offset add to embed when do the normalization.
            Defaults to 0.
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None
    """

    def __init__(self,
                 num_feature,
                 temperature=10000,
                 normalize=True,
                 scale=2 * math.pi,
                 eps=1e-6,
                 offset=0.):
        super().__init__()
        if normalize:
            assert isinstance(scale, (float, int)), 'when normalize is set,' \
                'scale should be provided and in float or int type, ' \
                f'found {type(scale)}'
        self.num_feature = num_feature
        self.temperature = temperature
        self.normalize = normalize
        self.scale = scale
        self.eps = eps
        self.offset = offset

    def forward(self, mask):
        """Forward function for `SinePositionalEncoding`.
        Args:
            mask (Tensor): ByteTensor mask. Non-zero values representing
                ignored positions, while zero values means valid positions
                for this image. Shape [bs, h, w].
        Returns:
            pos (Tensor): Returned position embedding with shape
                [bs, num_feats*2, h, w].
        """
        # For convenience of exporting to ONNX, it's required to convert
        # `masks` from bool to int.
        mask = mask.to(torch.int)
        not_mask = 1 - mask
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        if self.normalize:
            y_embed = (y_embed + self.offset) / \
                      (y_embed[:, -1:, :] + self.eps) * self.scale
            x_embed = (x_embed + self.offset) / \
                      (x_embed[:, :, -1:] + self.eps) * self.scale
        dim_t = torch.arange(
            self.num_feature, dtype=torch.float32, device=mask.device)
        dim_t = self.temperature**(2 * (dim_t // 2) / self.num_feature)
        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        # use `view` instead of `flatten` for dynamically exporting to ONNX
        B, H, W = mask.size()
        pos_x = torch.stack(
            (pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()),
            dim=4).view(B, H, W, -1)
        pos_y = torch.stack(
            (pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()),
            dim=4).view(B, H, W, -1)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos