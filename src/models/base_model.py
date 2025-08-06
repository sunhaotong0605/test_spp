from abc import abstractmethod
from dataclasses import dataclass
from typing import Optional, Union, Tuple, List, Dict

import torch
from torch import nn
from torch.nn import MSELoss, CrossEntropyLoss, BCEWithLogitsLoss
from peft import LoraConfig, get_peft_model
from transformers import PreTrainedTokenizerBase, PreTrainedModel, PretrainedConfig
from transformers.modeling_outputs import SequenceClassifierOutput, MaskedLMOutput

from src.self_logger import logger


class BaseEmbedding(nn.Module):

    @abstractmethod
    def get_embedding_dim(self, **kwargs) -> int:
        raise NotImplementedError()

    @abstractmethod
    def get_config(self, **kwargs) -> PretrainedConfig:
        raise NotImplementedError()

    def get_embedding(self, embedding: torch.Tensor, pooling="cls"):
        """such as support mean pooling, max pooling, cls pooling"""
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

    @abstractmethod
    def get_tokenizer(self, **kwargs) -> PreTrainedTokenizerBase:
        """return tokenizer"""
        raise NotImplementedError()

    @abstractmethod
    def get_embedding_dtype(self, **kwargs):
        """return tokenizer"""
        raise NotImplementedError()


class BaseClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config:PretrainedConfig):
        super(BaseClassificationHead, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size, dtype=config.torch_dtype)
        self.out_proj = nn.Linear(config.hidden_size, config.vocab_size, dtype=config.torch_dtype)

    def forward(self, features, **kwargs):
        x = self.dense(features)
        x = torch.tanh(x)
        x = self.out_proj(x)
        return x


class BaseForMaskedML(nn.Module):
    def __init__(
            self,
            embedding: BaseEmbedding,
            freeze_embedding: bool = False
    ):
        config = embedding.get_config
        super(BaseForMaskedML, self).__init__()
        self.config = config
        self.embedding: BaseEmbedding = embedding
        self.lm_head = BaseClassificationHead(config)
        # self.init_weights()
        if freeze_embedding:
            for name, params in self.embedding.named_parameters():
                params.requires_grad = False

    def get_input_embeddings(self):
        return self.embedding.get_input_embeddings

    def get_output_embeddings(self):
        return self.lm_head

    def forward(
            self,
            input_ids: Optional[torch.LongTensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            head_mask: Optional[torch.Tensor] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            encoder_hidden_states: Optional[torch.FloatTensor] = None,
            encoder_attention_mask: Optional[torch.Tensor] = None,
            labels: Optional[torch.LongTensor] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            **kwargs
    ) -> Union[Tuple, MaskedLMOutput]:
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )
        outputs = self.embedding(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        logits = self.lm_head(outputs.hidden_states)

        masked_lm_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            if -100 in labels:
                active_loss = labels.ne(-100)
                labels = labels[active_loss]
                prediction_scores = logits[active_loss]

            labels = labels.to(prediction_scores.device)
            masked_lm_loss = loss_fct(
                prediction_scores.view(-1, self.config.vocab_size), labels.view(-1)
            )

        if not return_dict:
            output = (prediction_scores,) + outputs[2:]
            return (
                ((masked_lm_loss,) + output) if masked_lm_loss is not None else output
            )

        return MaskedLMOutput(
            loss=masked_lm_loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class BaseForClassifier(nn.Module):
    def __init__(
            self,
            embedding: BaseEmbedding,
            device: str = None,
            num_classes: int = 2,
            hidden_size: Optional[int] = None,
            freeze_embedding: bool = True,
            pooling: str = "cls",
    ):
        super(BaseForClassifier, self).__init__()
        self.config = embedding.get_config
        self.embedding: BaseEmbedding = embedding
        self.pooling = pooling
        self.loss_fct = self._get_loss_fn()

        if freeze_embedding:
            for name, params in self.embedding.named_parameters():
                params.requires_grad = False
        if isinstance(hidden_size, int):
            self.classifier = nn.Sequential(
                nn.Linear(self.embedding.get_embedding_dim, hidden_size,
                        dtype=self.embedding.get_embedding_dtype, device=device),
                nn.ReLU(),
                nn.Linear(hidden_size, num_classes,
                        dtype=self.embedding.get_embedding_dtype, device=device),
            )
        else:
            self.classifier = nn.Sequential(
                nn.Linear(self.embedding.get_embedding_dim, num_classes,
                        dtype=self.embedding.get_embedding_dtype, device=device),
            )

    @staticmethod
    def _get_loss_fn(problem_type: str = "cross_entropy"):
        if problem_type == "mse":
            loss_fct = MSELoss()
        elif problem_type == "cross_entropy":
            loss_fct = CrossEntropyLoss()
        elif problem_type == "bce_with_logits":
            loss_fct = BCEWithLogitsLoss()
        else:
            raise ValueError(f"problem_type {problem_type} is not supported.")
        return loss_fct

    def forward(
            self,
            input_ids: Optional[torch.LongTensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            head_mask: Optional[torch.Tensor] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            encoder_hidden_states: Optional[torch.FloatTensor] = None,
            encoder_attention_mask: Optional[torch.Tensor] = None,
            labels: Optional[torch.LongTensor] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            **kwargs
    ) -> Union[Tuple, SequenceClassifierOutput]:
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        outputs = self.embedding(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
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


@dataclass
class MultiTaskSequenceClassifierOutput(SequenceClassifierOutput):
    loss: Optional[torch.FloatTensor] = None
    multi_loss: Optional[Union[Dict, str, torch.Tensor]] = None
    logits: Optional[torch.FloatTensor] = None
    multi_logits: Optional[Union[Dict, str, torch.Tensor]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None


class BaseForMultiTaskSequence(PreTrainedModel):
    multi_task = {
        "task_name": {
            "type": "class",
            "num_classes": 2,
            "hidden_size": 128,
            "pooling": "cls",
            "loss_type": "cross_entropy",
            "label_name": "task_label_name",
        }
    }

    def __init__(
            self,
            embedding: BaseEmbedding,
            device: str = None,
            freeze_embedding: bool = True,
            multi_task: Dict = None
    ):
        super(BaseForMultiTaskSequence, self).__init__(config=embedding.get_config)
        self.embedding: BaseEmbedding = embedding

        if freeze_embedding:
            for name, params in self.embedding.named_parameters():
                params.requires_grad = False

        self.multi_task_model = nn.ModuleDict()
        self.multi_task = multi_task
        for name, task in multi_task.items():
            model = self._get_task_model(
                task_type=task["type"],
                hidden_size=task["hidden_size"],
                num_classes=task["num_classes"],
                device=device,
            )
            self.multi_task_model.add_module(name, model)

    def forward(
            self,
            input_ids: Optional[torch.LongTensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            head_mask: Optional[torch.Tensor] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            encoder_hidden_states: Optional[torch.FloatTensor] = None,
            encoder_attention_mask: Optional[torch.Tensor] = None,
            labels: Optional[Union[Dict, torch.LongTensor]] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            **kwargs
    ) -> Union[Tuple, MultiTaskSequenceClassifierOutput]:
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        outputs = self.embedding(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            **kwargs
        )

        multi_losses = {}
        multi_logits = {}
        if labels is not None:
            weights = []
            for name, task in self.multi_task.items():
                if task["weight"] is not None and name in labels:
                    weights.append(task["weight"])
            weights = sum(weights)

            for label_name, label_value in labels.items():
                if label_value is not None:
                    embedding = self.embedding.get_embedding(
                        outputs.hidden_states,
                        pooling=self.multi_task[label_name]["pooling"]
                    )
                    multi_logits[label_name] = self.multi_task_model[label_name](embedding)
                    loss_fn = self._get_loss_fn(self.multi_task[label_name]["loss_type"])
                    predict = multi_logits[label_name].view(-1, self.multi_task[label_name]["num_classes"])
                    label = label_value.view(-1).to(predict.device)
                    loss = loss_fn(predict, label)
                    multi_losses[label_name] = loss * self.multi_task[label_name]["weight"] / weights

        loss = sum(v for k, v in multi_losses.items() if v is not None)
        if not return_dict:
            multi_logits = (v for k, v in multi_logits.items() if v is not None)
            output = (multi_logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return MultiTaskSequenceClassifierOutput(
            loss=loss,
            multi_loss=multi_losses,
            logits=None,
            multi_logits=multi_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def _get_task_model(self, task_type: str, hidden_size: int = None, num_classes: int = None, device: str = None):
        if task_type == "class":
            if isinstance(hidden_size, int):
                task_model = nn.Sequential(
                    nn.Linear(self.embedding.get_embedding_dim, hidden_size,
                              dtype=self.embedding.get_embedding_dtype, device=device),
                    nn.ReLU(),
                    nn.Linear(hidden_size, num_classes,
                              dtype=self.embedding.get_embedding_dtype, device=device),
                )
            else:
                task_model = nn.Sequential(
                    nn.Linear(self.embedding.get_embedding_dim, num_classes,
                              dtype=self.embedding.get_embedding_dtype, device=device),
                )
        elif task_type == "mask":
            task_model = nn.Linear(self.embedding.get_embedding_dim, num_classes,
                                   dtype=self.embedding.get_embedding_dtype, device=device)
        else:
            raise ValueError(f"task type {task_type} is not supported. [class, mask]")
        return task_model

    @staticmethod
    def _get_loss_fn(problem_type: str = "cross_entropy"):
        if problem_type == "mse":
            loss_fct = MSELoss()
        elif problem_type == "cross_entropy":
            loss_fct = CrossEntropyLoss()
        elif problem_type == "bce_with_logits":
            loss_fct = BCEWithLogitsLoss()
        else:
            raise ValueError(f"problem_type {problem_type} is not supported.")
        return loss_fct


class BaseLoraForClassifier(BaseForClassifier):
    _init_weights = True

    def __init__(
            self,
            embedding: BaseEmbedding,
            lora_config: LoraConfig,
            device: str = None,
            num_classes: int = 2,
            hidden_size: int = None,
            freeze_embedding: bool = True,
            pooling: str = "cls",
    ):
        super().__init__(
            embedding=embedding,
            device=device,
            num_classes=num_classes,
            hidden_size=hidden_size,
            freeze_embedding=freeze_embedding,
            pooling=pooling,
        )
        self.embedding = get_peft_model(self.embedding, lora_config)
        self.embedding.print_trainable_parameters()
