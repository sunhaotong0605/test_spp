from typing import Optional, Tuple, Union

import torch
from peft import LoraConfig
from transformers import AutoModelForSequenceClassification, AutoTokenizer, PretrainedConfig
from transformers.modeling_outputs import SequenceClassifierOutput, \
    BaseModelOutputWithPastAndCrossAttentions

from src.models.base_model import BaseEmbedding, BaseLoraForClassifier, BaseForClassifier, BaseForMultiTaskSequence
from src.self_logger import logger


class NTEmbedding(BaseEmbedding):
    def __init__(self, pretrained_model_name_or_path: str, device: str = None, ignore_mismatched_sizes=True):
        super(NTEmbedding, self).__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path,
            trust_remote_code=True
        )
        base_model = AutoModelForSequenceClassification.from_pretrained(
            pretrained_model_name_or_path,
            device_map=device,
            trust_remote_code=True,
            ignore_mismatched_sizes=ignore_mismatched_sizes
        )
        # print(base_model)
        # logger.info(f"load model from: {pretrained_model_name_or_path}")
        self.base_embed = base_model.esm
        self.classifier = base_model.classifier
        self.config: PretrainedConfig = base_model.config
        self.dtype = base_model.dtype
        del base_model

    def forward(
            self,
            input_ids: Optional[torch.LongTensor] = None,
            **kwargs
    ) -> BaseModelOutputWithPastAndCrossAttentions:
        embedding = self.base_embed(
            input_ids=input_ids,
            **kwargs
        ).last_hidden_state
        output = self.classifier(embedding)

        return BaseModelOutputWithPastAndCrossAttentions(last_hidden_state=output, hidden_states=embedding)

    @property
    def get_config(self) -> PretrainedConfig:
        return self.config

    @property
    def get_tokenizer(self):
        return self.tokenizer

    @property
    def get_embedding_dim(self, **kwargs):
        return self.base_embed.config.hidden_size

    @property
    def get_embedding_dtype(self, **kwargs):
        return self.dtype

    def get_embedding(self, embedding, pooling="cls"):
        if pooling == "cls":
            return embedding[:, 0]
        elif pooling == "mean":
            return embedding.mean(dim=1)
        elif pooling == "max":
            return embedding.max(dim=1)
        elif pooling == "last":
            return embedding[:, -1, :]
        elif pooling == "original":
            return embedding
        else:
            raise ValueError(f"Pooling method {pooling} not supported.")


class NTForClassifier(BaseForClassifier):
    def __init__(
            self,
            pretrained_model_name_or_path: str,
            num_classes: int = 2,
            hidden_size: Optional[int] = 512,
            device: Optional[str] = None,
            pooling: str = "cls",
            freeze_embedding=True
    ):
        embedding = NTEmbedding(pretrained_model_name_or_path, device=device)
        super().__init__(
            embedding=embedding,
            device=device,
            num_classes=num_classes,
            hidden_size=hidden_size,
            pooling=pooling,
            freeze_embedding=freeze_embedding,
        )
        del embedding


class NTLoraForClassifier(BaseLoraForClassifier):
    def __init__(
            self,
            pretrained_model_name_or_path: str,
            num_classes: int = 2,
            hidden_size: Optional[int] = 512,
            device: Optional[str] = None,
            lora_inference_mode=False,
            lora_r=8,
            lora_alpha=32,
            lora_dropout=0.1,
            pooling: str = "cls",
            freeze_embedding=True,
            **kwargs
    ):
        embedding = NTEmbedding(pretrained_model_name_or_path, device=device)
        peft_config = LoraConfig(
            inference_mode=lora_inference_mode,
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
        )
        peft_config.target_modules = ["query", "key", "value", "out_proj"]
        super().__init__(
            embedding=embedding,
            lora_config=peft_config,
            device=device,
            num_classes=num_classes,
            hidden_size=hidden_size,
            pooling=pooling,
            freeze_embedding=freeze_embedding,
        )
        del embedding


class NTForMultiTaskSequence(BaseForMultiTaskSequence):
    def __init__(
            self,
            pretrained_model_name_or_path: str,
            multi_task: dict,
            device: Optional[str] = None,
            freeze_embedding=True,
            **kwargs
    ):
        embedding = NTEmbedding(pretrained_model_name_or_path, device=device)
        logger.info("multi task: \n{}".format(multi_task))
        super().__init__(
            embedding=embedding,
            device=device,
            freeze_embedding=freeze_embedding,
            multi_task=multi_task
        )
        del embedding
