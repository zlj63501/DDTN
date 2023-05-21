import copy
from typing import Any, Dict, List, Optional, Tuple

import omegaconf
import torch
from mmf.common.registry import registry
from mmf.models.base_model import BaseModel
from mmf.modules.embeddings import (
    PreExtractedEmbedding,
    TextEmbedding,
    VqaLocEmbedding
)
from mmf.modules.layers import ModalCombineLayer, ClassifierLayer
from mmf.utils.build import build_image_encoder
from mmf.utils.general import filter_grads
from omegaconf import DictConfig

@registry.register_model("ddtn")
class ddtn(BaseModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self._global_config = registry.get("config")
        self._datasets = self._global_config.datasets
        if isinstance(self._datasets, str):
            self._datasets = self._datasets.split(",")

    @classmethod
    def config_path(cls):
        return "configs/models/ddtn/defaults.yaml"

    def build(self):
        self.image_feature_dim = 2048
        self._build_word_embedding()
        self._init_text_embeddings("text")
        self._init_feature_encoders("image")
        self._init_feature_embeddings("image")
        # self._init_combine_layer("image", "text")
        self._init_classifier(self._get_classifier_input_dim())
        self._init_extras()

    def _build_word_embedding(self):
        assert len(self._datasets) > 0
        text_processor = registry.get(self._datasets[0] + "_text_processor")
        vocab = text_processor.vocab
        self.word_embedding = vocab.get_embedding(torch.nn.Embedding, embedding_dim=300)

    def _init_text_embeddings(self, attr: str = "text"):
        if "embeddings" not in attr:
            attr += "_embeddings"

        module_config = self.config[attr]
        embedding_type = module_config.type
        embedding_kwargs = copy.deepcopy(module_config.params)
        self._update_text_embedding_args(embedding_kwargs)
        embedding = TextEmbedding(embedding_type, **embedding_kwargs)
        embeddings_out_dim = embedding.text_out_dim

        setattr(self, attr + "_out_dim", embeddings_out_dim)
        setattr(self, attr, embedding)

    def _update_text_embedding_args(self, args):
        # Add model_data_dir to kwargs
        args.model_data_dir = self.config.model_data_dir

    def _init_feature_encoders(self, attr: str):

        feat_encoder= self.config[attr + "_feature_encodings"]
        feature_dim = self.config[attr + "_feature_dim"]
        setattr(self, attr + "_feature_dim", feature_dim)

        feat_encoder_config = copy.deepcopy(feat_encoder)
        with omegaconf.open_dict(feat_encoder_config):
            feat_encoder_config.params.model_data_dir = self.config.model_data_dir
            feat_encoder_config.params.in_dim = feature_dim
        feat_model = build_image_encoder(feat_encoder_config, direct_features=True)

        setattr(self, attr + "_feature_dim", feat_model.out_dim)
        setattr(self, attr + "_feature_encoders", feat_model)

    def _init_feature_embeddings(self, attr: str):

        feature_attn_model_params = self.config[attr + "_feature_embeddings"]["params"]

        feature_embedding = VqaLocEmbedding(
                    getattr(self, attr + "_feature_dim"),
                    self.text_embeddings_out_dim,
                    **feature_attn_model_params,
        )
        setattr(
            self, attr + "_feature_embeddings_out_dim", feature_embedding.out_dim
        )
        setattr(self, attr + "_feature_embeddings_list", feature_embedding)

    def _get_embeddings_attr(self, attr):
        embedding_attr1 = attr
        if hasattr(self, attr + "_embeddings_out_dim"):
            embedding_attr1 = attr + "_embeddings_out_dim"
        else:
            embedding_attr1 = attr + "_feature_embeddings_out_dim"

        return embedding_attr1

    def _init_combine_layer(self, attr1: str, attr2: str):
        config_attr = attr1 + "_" + attr2 + "_modal_combine"

        multi_modal_combine_layer = ModalCombineLayer(
            self.config[config_attr].type,
            getattr(self, self._get_embeddings_attr(attr1)),
            getattr(self, self._get_embeddings_attr(attr2)),
            **self.config[config_attr].params,
        )

        setattr(
            self,
            attr1 + "_" + attr2 + "_multi_modal_combine_layer",
            multi_modal_combine_layer
        )

    def _init_classifier(self, combined_embedding_dim: int):
        # TODO: Later support multihead
        num_choices = registry.get(self._datasets[0] + "_num_final_outputs")
        params = self.config["classifier"].get("params")
        if params is None:
            params = {}

        self.classifier = ClassifierLayer(
            self.config.classifier.type,
            in_dim=combined_embedding_dim,
            out_dim=num_choices,
            **params
        )

    def _get_classifier_input_dim(self):
        return self.image_feature_embeddings_list.out_dim

    def _init_extras(self):
        self.inter_model = None

    def get_optimizer_parameters(self, config: DictConfig) -> List[Dict[str, Any]]:
        params = [
            {"params": filter_grads(self.word_embedding.parameters())},
            {"params": filter_grads(self.image_feature_embeddings_list.sga.parameters())},
            {
                "params": filter_grads(self.image_feature_embeddings_list.fve.parameters()),
                "lr": (config.optimizer.params.lr ),
             },
            {"params": filter_grads(
                self.image_feature_embeddings_list.sga_pool.parameters())},
            {"params": filter_grads(self.text_embeddings.parameters())},
            {"params": filter_grads(self.classifier.parameters())},
            {
                "params": self.image_feature_encoders.parameters(),
                "lr": (config.optimizer.params.lr*0.1)
            },
        ]

        return params

    def process_text_embedding(
            self, sample_list: Dict[str, Any], embedding_attr: str = "text_embeddings"
    ):

        # Get "text" attribute in case of "text_embeddings" case
        # and "context" attribute in case of "context_embeddings"
        texts = getattr(sample_list, embedding_attr.split("_")[0])

        # Get embedding models
        text_embedding_model = getattr(self, embedding_attr)

        # TODO: Move this logic inside
        if isinstance(text_embedding_model, PreExtractedEmbedding):
            text_embedding_total = text_embedding_model(sample_list.question_id)
        else:
            text_embedding_total, text_embedding_vec, att_enc  = text_embedding_model(
                texts
            )

        return text_embedding_total, text_embedding_vec, att_enc

    def process_feature_embedding(
        self, attr: str,
            sample_list: Dict[str, Any],
            text_embedding_total: torch.Tensor,
            text_embedding_vec: torch.Tensor,
    ):


        img_feature = getattr(sample_list, f"{attr}_feature_0")
        grid_feature = getattr(sample_list, "grid_feat", img_feature)

        feature_encoder = getattr(self, attr + "_feature_encoders")

        # Encode the features
        encoded_feature = feature_encoder(grid_feature)

        feature_embedding_model = getattr(self,  attr + "_feature_embeddings_list")

        img_mask = getattr(sample_list, "img_mask", None)

        feature_embedding_total, att_lists = feature_embedding_model(
            img_feature,
            encoded_feature,
            img_mask,
            text_embedding_total,
            text_embedding_vec,
            None,
            sample_list.text_mask,
            sample_list.norm_shape,
        )


        return feature_embedding_total, att_lists

    def combine_embeddings(self, *args):
        feature_names = args[0]
        v1, v2 = args[1]
        # q = args[2]

        layer = "_".join(feature_names) + "_multi_modal_combine_layer"

        return getattr(self, layer)(v1, v2)

    def calculate_logits(self, joint_embeddings: torch.Tensor, **kwargs):
        feature_sga, feature_fve = joint_embeddings
        return self.classifier(feature_sga, feature_fve)


    def forward(self, sample_list):
        sample_list.text_mask = sample_list.text.eq(0)
        sample_list.text = self.word_embedding(sample_list.text)
        text_embedding_total, text_embedding_vec, att_enc = self.process_text_embedding(
            sample_list
        )

        joint_embeddings, att_lists = self.process_feature_embedding(
            "image", sample_list, text_embedding_total , text_embedding_vec[:, 0]
        )

        sample_list.att_enc = att_enc
        sample_list.att_ddtn = att_lists

        acc_vqa, acc_seg = self.calculate_logits(joint_embeddings)

        model_output = {"scores":  acc_vqa,
                        "acc_seg": acc_seg
        }

        return model_output