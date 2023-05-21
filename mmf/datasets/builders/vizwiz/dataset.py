# Copyright (c) Facebook, Inc. and its affiliates.
import os
import cv2
import seaborn as sns
import numpy as np
import pycocotools.mask as maskUtils

from mmf.datasets.processors import BitmapMasks
from mmf.datasets.builders.vqa2 import VQA2Dataset

from visualize import visualize_grid_attention_v2

import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
_CMAP = "Purples"


class VizWizDataset(VQA2Dataset):
    def __init__(self, config, dataset_type, imdb_file_index, *args, **kwargs):
        super().__init__(
            config,
            dataset_type,
            imdb_file_index,
            dataset_name="vizwiz",
            *args,
            **kwargs
        )
        self.val_image_dir = config.images.val[0]
        val_anns = np.load(config.annotations.val[0], allow_pickle=True)
        question_dict = {}
        for val_ann in val_anns[1:]:
            question_dict[str(val_ann["image_id"])]={"question_tokens": val_ann["question_tokens"]}

        self.question_dicts = question_dict



    def load_item(self, idx):
        sample = super().load_item(idx)


        return sample

    def format_for_prediction(self, report):
        answers = report.scores.argmax(dim=1)

        predictions = []
        answer_space_size = self.answer_processor.get_true_vocab_size()

        for idx, image_id in enumerate(report.image_id):
            answer_id = answers[idx].item()

            if answer_id >= answer_space_size:
                answer_id -= answer_space_size
                answer = report.context_tokens[idx][answer_id]
            else:
                answer = self.answer_processor.idx2word(answer_id)
            if answer == "<unk>" or answer == "<pad>":
                answer = "unanswerable"
            predictions.append(
                {
                    "image": "VizWiz_"
                    + self._dataset_type
                    + "_"
                    + str(image_id.item()).zfill(8)
                    + ".jpg",
                    "answer": answer,
                }
            )
        return predictions


