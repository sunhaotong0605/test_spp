import os
from typing import Optional

import torch
from peft import LoraConfig
from torch import nn
from transformers import AutoConfig, AutoModelForCausalLM, PretrainedConfig
import yaml

from stripedhyena.utils import dotdict
from stripedhyena.model import StripedHyena
from transformers.modeling_outputs import SequenceClassifierOutput, BaseModelOutputWithPastAndCrossAttentions

from src.self_logger import logger
from src.models.evo.evo_tokenizer import EvoCharLevelTokenizer
from src.models.base_model import BaseEmbedding, BaseLoraForClassifier, BaseForClassifier

MODEL_NAMES = ['evo-1-8k-base', 'evo-1-131k-base']


class EvoPretrainedConfig(dotdict, PretrainedConfig):
    """dot.notation fixed to： AttributeError: 'PretrainedConfig' object has no attribute 'get'"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class EvoEmbedding(BaseEmbedding):
    def __init__(
            self,
            pretrained_model_name_or_path,
            device: str = None,
    ):
        """
        Loads an Evo model checkpoint given a model name.
        If the checkpoint does not exist, we automatically download it from HuggingFace.
        """
        super().__init__()
        # Check model name.
        model_name = os.path.basename(pretrained_model_name_or_path)
        if model_name not in MODEL_NAMES:
            raise ValueError(
                f'Invalid model name {pretrained_model_name_or_path}. Should be one of: '
                f'{", ".join(MODEL_NAMES)}.'
            )

        # Assign config path.
        current_work_dir = os.path.abspath(os.path.dirname(__file__))
        if model_name == 'evo-1-8k-base':
            config_path = os.path.join(current_work_dir, 'evo-1-8k-base_inference.yml')
        elif model_name == 'evo-1-131k-base':
            config_path = os.path.join(current_work_dir, 'evo-1-131k-base_inference.yml')
        else:
            raise ValueError(
                f'Invalid model name {model_name}. Should be one of: '
                f'{", ".join(MODEL_NAMES)}.'
            )
        # logger.info(f"Config path: {config_path}")

        # Load model.
        self.base_model = self.load_checkpoint(
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            config_path=config_path,
            device=device
        )
        # logger.info(f"load model weight from: {pretrained_model_name_or_path}")
        self.base_embed = self.base_model.unembed.unembed
        self.dtype = self.base_model.unembed.weight.dtype
        self.config: PretrainedConfig = PretrainedConfig(**self.base_model.config)
        # print(self.base_model)

    def forward(
            self,
            input_ids,
            inference_params_dict=None,
            attention_mask=None,
            **kwargs
    ) -> BaseModelOutputWithPastAndCrossAttentions:
        x = self.base_model.embedding_layer.embed(input_ids)
        if inference_params_dict is not None:
            x, inference_params_dict_out = self.base_model.stateful_forward(
                x,
                inference_params_dict=inference_params_dict,
            )
        else:
            x, inference_params_dict_out = self.base_model.stateless_forward(x, padding_mask=attention_mask)
        embedding = self.base_model.norm(x)
        last_hidden_state = self.base_embed(embedding)

        return BaseModelOutputWithPastAndCrossAttentions(last_hidden_state=last_hidden_state, hidden_states=embedding)

    @property
    def get_embedding_dim(self):
        return self.base_model.config.hidden_size

    @property
    def get_embedding_dtype(self, **kwargs):
        return self.dtype

    @property
    def get_config(self) -> PretrainedConfig:
        return self.config

    @property
    def get_tokenizer(self, vocab_size: int = 512, **kwargs):
        tokenizer = EvoCharLevelTokenizer(vocab_size)
        tokenizer.pad_token = "X"
        return tokenizer

    def get_embedding(self, embedding, pooling="cls"):
        if pooling == "cls":
            return embedding[:, 0, :]
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

    @staticmethod
    def load_checkpoint(
            pretrained_model_name_or_path: str,
            config_path: str,
            device: str = None,
            *args, **kwargs
    ):
        """
        Load checkpoint from HuggingFace and place it into SH model.
        """
        assert os.path.exists(config_path), f"Config file not found: {config_path}"
        # Map model name to HuggingFace model name.
        cache_dir = os.path.dirname(pretrained_model_name_or_path)

        model_config = AutoConfig.from_pretrained(
            pretrained_model_name_or_path,
            trust_remote_code=True,
            revision='1.1_fix',
            cache_dir=cache_dir
        )
        model_config.use_cache = True

        # Load model.
        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path,
            config=model_config,
            trust_remote_code=True,
            revision='1.1_fix',
            cache_dir=cache_dir
        )

        # # Load model state dict & cleanup.
        state_dict = model.backbone.state_dict()
        del model
        del model_config

        # Load SH config.
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        global_config = EvoPretrainedConfig(config)
        # NOTE: mlp_dtype and attn_block_dtype default to torch.bfloat16. attn_block_dtype must be [bfloat16, float16]
        # global_config.mlp_dtype = torch.float32
        # global_config.attn_block_dtype = torch.float16

        # Load SH Model.
        model = StripedHyena(global_config)
        model.load_state_dict(state_dict, strict=True)
        model.to_bfloat16_except_poles_residues()
        if device is not None:
            model = model.to(device)
        # print(model)
        return model



class EvoForClassifier(BaseForClassifier):

    def __init__(
            self,
            pretrained_model_name_or_path: str = MODEL_NAMES[0],
            device: str = None,
            hidden_size: int = None,
            num_classes: int = 2,
            freeze_embedding: bool = True,
            pooling: str = "cls",
    ):
        embedding = EvoEmbedding(pretrained_model_name_or_path=pretrained_model_name_or_path, device=device)
        super().__init__(
            embedding=embedding,
            device=device,
            num_classes=num_classes,
            hidden_size=hidden_size,
            freeze_embedding=freeze_embedding,
            pooling=pooling
        )
        # NOTE: Freeze the embedding layer.
        if freeze_embedding:
            for name, params in self.embedding.named_parameters():
                params.requires_grad = False
        self.classifier = nn.Linear(
            self.embedding.get_embedding_dim,
            num_classes,
            dtype=self.embedding.get_embedding_dtype,
            device=device)

    def forward(
            self,
            input_ids: Optional[torch.LongTensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            head_mask: Optional[torch.Tensor] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            **kwargs
    ) -> SequenceClassifierOutput:
        return_dict = (return_dict if return_dict is not None else self.config.use_return_dict)
        outputs = self.embedding(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            **kwargs
        )
        embedding = self.embedding.get_embedding(
            outputs.hidden_states,
            pooling=self.pooling
        )
        logits = self.classifier(embedding)

        loss = None
        if labels is not None:
            labels = labels.to(logits.device)
            loss = self.loss_fct(logits, labels)

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=embedding,
            attentions=outputs.attentions,
        )


class EvoLoraForClassifier(BaseLoraForClassifier):
    def __init__(
            self,
            pretrained_model_name_or_path: str = MODEL_NAMES[0],
            device: str = None,
            num_classes: int = 2,
            hidden_size: int = 512,
            lora_inference_mode=False,
            lora_r=8,
            lora_alpha=32,
            lora_dropout=0.1,
            freeze_embedding: bool = True,
            pooling: str = "cls"
    ):
        embedding = EvoEmbedding(pretrained_model_name_or_path=pretrained_model_name_or_path, device=device)
        peft_config = LoraConfig(
            inference_mode=lora_inference_mode,
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
        )
        peft_config.target_modules = ["Wqkv", "out_proj", "projections", "out_filter_dense", "l1", "l2", "l3"]
        super().__init__(
            embedding=embedding,
            lora_config=peft_config,
            device=device,
            num_classes=num_classes,
            hidden_size=hidden_size,
            freeze_embedding=freeze_embedding,
            pooling=pooling,
        )
        del embedding

    def forward(
            self,
            input_ids: Optional[torch.LongTensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            head_mask: Optional[torch.Tensor] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            **kwargs
    ):
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        outputs = self.embedding(
            input_ids,
            attention_mask=attention_mask
        )
        embedding = self.embedding.get_embedding(outputs.hidden_states, pooling=self.pooling)
        logits = self.classifier(embedding)

        loss = None
        if labels is not None:
            labels = labels.to(logits.device)
            loss = self.loss_fct(logits, labels)

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=embedding,
            attentions=outputs.attentions,
        )


if __name__ == '__main__':
    model = EvoEmbedding()
    print(model)
