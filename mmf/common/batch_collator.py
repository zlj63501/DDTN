# Copyright (c) Facebook, Inc. and its affiliates.
import torch
from mmf.common.sample import convert_batch_to_sample_list


class BatchCollator:
    def __init__(self, dataset_name, dataset_type):
        self._dataset_name = dataset_name
        self._dataset_type = dataset_type

    def __call__(self, batch):

        if "grid_feat" in batch[0].keys():
            max_size = self._max_by_axis([list(sample.grid_feat.shape[1:]) for sample in batch])
            c = batch[0].grid_feat.shape[0]
            batch_shape = [len(batch)] + [c] + max_size
            dtype = batch[0].grid_feat.dtype
            device = batch[0].grid_feat.device
            tensor = torch.zeros(batch_shape, dtype=dtype, device=device)
            mask = torch.ones([len(batch)] + max_size, dtype=torch.bool, device=device)
            for img, pad_img, m in zip(batch, tensor, mask):
                img_temp = img.grid_feat
                pad_img[: img_temp.shape[0], : img_temp.shape[1], : img_temp.shape[2]].copy_(img_temp)
                m[: img_temp.shape[1], :img_temp.shape[2]] = False
                img.grid_feat = pad_img
                img.img_mask = m

        sample_list = convert_batch_to_sample_list(batch)
        sample_list.dataset_name = self._dataset_name
        sample_list.dataset_type = self._dataset_type
        return sample_list

    def _max_by_axis(self, the_list):
        # type:  (List[List[int]]) -> List[int]
        maxes = the_list[0]
        for sublist in the_list[1:]:
            for index, item in enumerate(sublist):
                maxes[index] = max(maxes[index], item)
        return maxes
