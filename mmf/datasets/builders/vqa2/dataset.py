# Copyright (c) Facebook, Inc. and its affiliates.
import logging

import numpy
import torch
import tqdm
import numpy as np
from mmf.common.sample import Sample
from mmf.datasets.mmf_dataset import MMFDataset
from mmf.datasets.processors import BitmapMasks, SampleMaskVertices
from mmf.utils.distributed import is_main
import pycocotools.mask as maskUtils


logger = logging.getLogger(__name__)


class VQA2Dataset(MMFDataset):
    def __init__(self, config, dataset_type, imdb_file_index, *args, **kwargs):
        if "name" in kwargs:
            name = kwargs["name"]
        elif "dataset_name" in kwargs:
            name = kwargs["dataset_name"]
        else:
            name = "vqa2"
        super().__init__(name, config, dataset_type, index=imdb_file_index)

        self._should_fast_read = self.config.get("fast_read", False)
        self.use_ocr = self.config.use_ocr
        self.use_ocr_info = self.config.use_ocr_info
        self.num_ray = self.config.num_ray
        self.num_bin = self.config.num_bin
        self.mapping = self.config.mapping

    def init_processors(self):
        super().init_processors()
        if not self._use_features:
            self.image_db.transform = self.image_processor
        self.sample_mask = SampleMaskVertices(num_ray=self.num_ray,num_bin = self.num_bin, mapping = self.mapping)

    def try_fast_read(self):
        # Don't fast read in case of test set.
        if self._dataset_type == "test":
            return

        if hasattr(self, "_should_fast_read") and self._should_fast_read is True:
            logger.info(
                f"Starting to fast read {self.dataset_name} {self.dataset_type} "
                + "dataset"
            )
            self.cache = {}
            for idx in tqdm.tqdm(
                range(len(self.annotation_db)), miniters=100, disable=not is_main()
            ):
                self.cache[idx] = self.load_item(idx)

    def __getitem__(self, idx):
        if self._should_fast_read is True and self._dataset_type != "test":
            return self.cache[idx]
        else:
            return self.load_item(idx)

    def load_item(self, idx):
        sample_info = self.annotation_db[idx]
        current_sample = Sample()

        if "question_tokens" in sample_info:
            text_processor_argument = {
                "tokens": sample_info["question_tokens"],
                "text": sample_info["question_str"],
            }
        else:
            text_processor_argument = {"text": sample_info["question"]}

        processed_question = self.text_processor(text_processor_argument)

        current_sample.text = processed_question["text"]
        if "input_ids" in processed_question:
            current_sample.update(processed_question)

        current_sample.question_id = torch.tensor(
            sample_info["question_id"], dtype=torch.int
        )

        if isinstance(sample_info["image_id"], int):
            current_sample.image_id = torch.tensor(
                sample_info["image_id"], dtype=torch.int
            )
        else:
            current_sample.image_id = sample_info["image_id"]

        if "question_tokens" in sample_info:
            current_sample.text_len = torch.tensor(
                len(sample_info["question_tokens"]), dtype=torch.int
            )

        if self._use_features:
            features = self.features_db[idx]
            if hasattr(self, "transformer_bbox_processor"):
                features["image_info_0"] = self.transformer_bbox_processor(
                    features["image_info_0"]
                )
            current_sample.update(features)
        else:
            image_path = sample_info["image_name"] + ".jpg"
            current_sample.image = self.image_db.from_path(image_path)["images"][0]
        if self._use_grid_features:
            current_sample.grid_feat = torch.tensor(features["image_info_0"]["grid_feat"]).squeeze(0)

        # Add details for OCR like OCR bbox, vectors, tokens here
        current_sample = self.add_ocr_details(sample_info, current_sample)
        # Depending on whether we are using soft copy this can add
        # dynamic answer space
        current_sample = self.add_answer_info(sample_info, current_sample)
        current_sample = self.add_mask_info(sample_info, current_sample)
        current_sample = self.add_bbox_info(sample_info, current_sample)

        return current_sample

    def add_ocr_details(self, sample_info, sample):
        if self.use_ocr:
            # Preprocess OCR tokens
            ocr_tokens = [
                self.ocr_token_processor({"text": token})["text"]
                for token in sample_info["ocr_tokens"]
            ]
            # Get embeddings for tokens
            context = self.context_processor({"tokens": ocr_tokens})
            sample.context = context["text"]
            sample.context_tokens = context["tokens"]
            sample.context_feature_0 = context["text"]
            sample.context_info_0 = Sample()
            sample.context_info_0.max_features = context["length"]

            order_vectors = torch.eye(len(sample.context_tokens))
            order_vectors[context["length"] :] = 0
            sample.order_vectors = order_vectors

        if self.use_ocr_info and "ocr_info" in sample_info:
            sample.ocr_bbox = self.bbox_processor({"info": sample_info["ocr_info"]})[
                "bbox"
            ]

        return sample

    def add_answer_info(self, sample_info, sample):
        if "answers" in sample_info:
            answers = sample_info["answers"]
            answer_processor_arg = {"answers": answers}

            if self.use_ocr:
                answer_processor_arg["tokens"] = sample_info["ocr_tokens"]
            processed_soft_copy_answers = self.answer_processor(answer_processor_arg)

            sample.targets = processed_soft_copy_answers["answers_scores"]

        return sample

    def add_mask_info(self,sample_info, sample):
        pad_h, pad_w = sample['image_info_0']['pad_height'], sample['image_info_0']['pad_width']

        h = sample_info['height']
        w = sample_info['width']

        if h > w:
            if pad_h > pad_w:
                m_h = pad_h
                m_w = pad_w
            else:
                m_h = pad_w
                m_w = pad_h
        else:
            if pad_h < pad_w:
                m_h = pad_h
                m_w = pad_w
            else:
                m_h = pad_w
                m_w = pad_h

        self.scale = (pad_h, pad_w)
        self.scale_factor = np.array([m_w / w, m_h / h, m_w / w, m_h / h], dtype=np.float32)

        sample['pad_shape'] = torch.tensor(np.array([m_h, m_w, 3]))
        sample['norm_shape'] = (pad_h, pad_w, 3)

        if "mask" in sample_info:
            mask = sample_info['mask']
            if isinstance(mask, list):
                rles_mask = maskUtils.frPyObjects(mask, h, w)
                rle_mask = maskUtils.merge(rles_mask)
            else:
                rle_mask = mask
            mask = maskUtils.decode(rle_mask)
            mask = BitmapMasks(mask[None], h, w)
            sample['gt_mask'] = mask.resize(self.scale)
            sample['gt_mask_rle'] = maskUtils.encode(
                np.asfortranarray(sample['gt_mask'].masks[0]))
            sample = self.sample_mask(sample)

        return sample

    def add_bbox_info(self, sample_info, sample):
        if "bbox" in sample_info:
            bbox = sample_info["bbox"] * self.scale_factor
            x1, y1, x2, y2 = bbox
            x1=0 if x1<0 else x1
            y1=0 if y1<0 else y1
            sample['mass_center'] = torch.tensor([(x2+x1)/2, (y2+y1)/2]).long()
        if 'prefinded_bbox' in sample['image_info_0']:
            bbox = sample['image_info_0']['prefinded_bbox']
            sample['bbox'] = torch.tensor(bbox)

        return sample


    def idx_to_answer(self, idx):
        return self.answer_processor.convert_idx_to_answer(idx)

    def format_for_prediction(self, report):
        answers = report.scores.argmax(dim=1)

        predictions = []
        answer_space_size = self.answer_processor.get_true_vocab_size()

        for idx, question_id in enumerate(report.question_id):
            answer_id = answers[idx].item()

            if answer_id >= answer_space_size:
                answer_id -= answer_space_size
                answer = report.context_tokens[idx][answer_id]
                if answer == self.context_processor.PAD_TOKEN:
                    answer = "unanswerable"
            else:
                answer = self.answer_processor.idx2word(answer_id)

            predictions.append(
                {
                    "question_id": question_id.item(),
                    "answer": answer,
                }
            )

        return predictions
