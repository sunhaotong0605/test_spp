import os
import os.path as osp
import socket
import torch

from abc import ABC, abstractmethod
from dataclasses import field
from pathlib import Path
from datetime import timedelta, datetime
from typing import Optional, Union

import torch.nn.functional as F
from accelerate import Accelerator, InitProcessGroupKwargs
from accelerate.utils import set_seed
from torch.optim.lr_scheduler import LRScheduler
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from safetensors import safe_open
from transformers import Trainer
from transformers.modeling_utils import unwrap_model
from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES
from transformers.trainer import _is_peft_model

from ..dataloaders.datasets.evo_tokenizer import EvoCharLevelTokenizer
from ..dataloaders.datasets.gue_dataset import GueDataset


class BaseTrainer(ABC):
    model: Optional[Union[torch.nn.Module]] = field(default=None, metadata={
        "help": "protein sequence AA classifier model"})
    train_loader: DataLoader = field(default=None, metadata={"help": "train loader"})
    test_loader: DataLoader = field(default=None, metadata={"help": "test loader"})
    valid_loader: DataLoader = field(default=None, metadata={"help": "valid loader"})
    optimizer: Optimizer = field(default=None, metadata={"help": "optimizer for training model"})
    lr_scheduler: LRScheduler = field(default=None, metadata={"help": "Learning rate decay course schedule"})
    batch_size: int = field(default=1, metadata={"help": "batch size"})
    loss_weight: float = field(default=1., metadata={"help": "loss weight"})
    max_epoch: int = field(default=100, metadata={"help": "max epoch"})
    learning_rate: float = field(default=1e-3, metadata={"help": "learning rate"})
    is_trainable: bool = field(default=True, metadata={"help": "whether the model to be train or not"})
    reuse: bool = field(default=False, metadata={"help": "whether the model parameters to be reuse or not"})
    accelerator: Accelerator = field(default=None)

    def __init__(self, **kwargs):

        set_seed(kwargs.get('seed', 42))
        process_group_kwargs = InitProcessGroupKwargs(
            timeout=timedelta(seconds=5400)
        )  # 1.5 hours
        self.accelerator = Accelerator(
            # mixed_precision='fp16',
            log_with='wandb',
            gradient_accumulation_steps=kwargs.get('k', 1),
            kwargs_handlers=[process_group_kwargs]
        )

        # output home
        output_home = kwargs.get('output_home', '.')
        self.save_in_batch = kwargs.get('save_in_batch', True)
        # ckpt
        self.best_ckpt_home = self.register_dir(output_home, 'best_ckpt')
        if self.save_in_batch:
            self.batch_ckpt_home = self.register_dir(output_home, 'batch_ckpt')
        # wandb
        self.wandb_home = self.register_dir(output_home, 'wandb')
        # result
        self.result_home = self.register_dir(output_home, 'result')

    def register_dir(self, parent_path, folder):
        new_path = osp.join(parent_path, folder)

        with self.accelerator.main_process_first():
            Path(new_path).mkdir(parents=True, exist_ok=True)

        return new_path

    def register_wandb(
            self,
            user_name: str,
            project_name: str = 'EvoTune',
            group: str = 'Classifier',
            task: str = "LinearHead",
            timestamp: str = datetime.now().strftime("%d_%m_%Y_%H_%M_%S")
    ):
        """
        register wandb to ProteintTrainer

        :param task: wandb name for current task
        :param user_name:
            username or team name where you're sending runs.
            This entity must exist before you can send runs there,
            so make sure to create your account or team in the UI before starting to log runs.
            If you don't specify an entity, the run will be sent to your default entity,
            which is usually your username. Change your default entity in [your settings](https://wandb.ai/settings)
            under "default location to create new projects".
        :param project_name:
            The name of the project where you're sending the new run. If the project is not specified, the run is put in an "Uncategorized" project.
        :param group:
            Specify a group to organize individual runs into a larger experiment.
            For example, you might be doing cross validation, or you might have multiple jobs that train and evaluate
            a model against different test sets. Group gives you a way to organize runs together into a larger whole, and you can toggle this
            on and off in the UI. For more details, see our [guide to grouping runs](https://docs.wandb.com/guides/runs/grouping).
            wandb ouput file save path
        :param timestamp:
        """

        self.accelerator.init_trackers(
            project_name=project_name,
            config={
                'learning_rate': self.learning_rate,
                'batch_size': self.batch_size,
                'loss_weight': self.loss_weight,
                'max_epoch': self.max_epoch
            },
            init_kwargs={
                'wandb': {
                    'entity': user_name,
                    'notes': socket.gethostname(),
                    'name': f'{task}_{timestamp}',
                    'group': group,
                    'dir': self.wandb_home,
                    'job_type': 'training',
                    'reinit': True
                }
            }
        )

    def register_dataset(
            self,
            data_files,
            mode,
            dataset_type='class',
            **kwargs
    ):
        """

        :param data_files: pickle files path list of protein sequence
        :param mode: data loader type, optional, only support `train`, `test` and `valid`
        :param dataset_type: dataset type, optional, only support `class` and `embed`

        :return:
        """
        self.batch_size = kwargs.get('batch_size', 2)
        if dataset_type == 'class':
            tokenizer = kwargs.get("tokenizer", EvoCharLevelTokenizer(vocab_size=512))
            assert os.path.exists(data_files), f"{data_files} is not existed"
            dataset = GueDataset(
                split=mode,
                dest_path=data_files,
                tokenizer=tokenizer,
                dataset_name="H3",
            )
        else:
            raise ValueError('Got an invalid dataset mode, ONLY SUPPORT: `class` and `embed`')

        if mode == 'train':
            self.train_loader = DataLoader(
                dataset=dataset,
                batch_size=self.batch_size,
                collate_fn=dataset.data_collator,
                shuffle=True,
            )
        elif mode == 'test':
            self.test_loader = DataLoader(
                dataset=dataset,
                batch_size=self.batch_size,
                collate_fn=dataset.data_collator,
                shuffle=False,
            )
        elif mode == 'valid':
            self.valid_loader = DataLoader(
                dataset=dataset,
                batch_size=self.batch_size,
                collate_fn=dataset.data_collator,
                shuffle=False,
            )
        else:
            raise ValueError('Got an invalid data loader mode, ONLY SUPPORT: `train`, `test` and `valid`')

    def register_model(self, model, **kwargs):
        reuse = kwargs.get('reuse', False)
        is_trainable = kwargs.get('is_trainable', True)
        self.learning_rate = kwargs.get('learning_rate', 5e-3)
        mode = kwargs.get('mode', 'best')

        self.model = model
        if is_trainable:
            self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate)
            self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, 1, gamma=0.9)

        if reuse:
            self.load_ckpt(mode=mode, is_trainable=is_trainable)

    def save_ckpt(self, mode):
        if self.accelerator.main_process_first():
            unwrapped_model = self.accelerator.unwrap_model(self.model)
            trainer_dict = {
                "optimizer": self.optimizer.state_dict(),
                "lr_scheduler": self.lr_scheduler.state_dict(),
            }
            classifier_dict = {"state_dict": unwrapped_model.classifier.state_dict()}

            if mode == 'batch':
                self.accelerator.save(trainer_dict, osp.join(self.batch_ckpt_home, 'trainer.bin'))
                self.accelerator.save(classifier_dict, osp.join(self.batch_ckpt_home, 'classifier.bin'))
                self._save_lora_weight(unwrapped_model, self.batch_ckpt_home)
                # unwrapped_model.embedding.lora_embedding.save_pretrained(self.batch_ckpt_home)
            elif mode in ['epoch', 'best']:
                self.accelerator.save(trainer_dict, osp.join(self.best_ckpt_home, 'trainer.bin'))
                self.accelerator.save(classifier_dict, osp.join(self.best_ckpt_home, 'classifier.bin'))
                self._save_lora_weight(unwrapped_model, self.best_ckpt_home)
                # unwrapped_model.embedding.lora_embedding.save_pretrained(self.best_ckpt_home)
            else:
                raise ValueError('Got an invalid dataset mode, ONLY SUPPORT: `batch`, `epoch` or `best`')

    def load_ckpt(self, mode, is_trainable=False):
        if mode == 'batch':
            path = self.batch_ckpt_home
        elif mode in ['epoch', 'best']:
            path = self.best_ckpt_home
        else:
            raise ValueError('Got an invalid dataset mode, ONLY SUPPORT: `batch`, `epoch` or `best`')

        if is_trainable:
            trainer_ckpt = torch.load(osp.join(path, 'trainer.bin'), map_location=torch.device('cuda'))
            self.optimizer.load_state_dict(trainer_ckpt['optimizer'])
            self.lr_scheduler.load_state_dict(trainer_ckpt['lr_scheduler'])

        # loading LoRA model
        if osp.exists(os.path.join(path, 'adapter_model.bin')):
            self.model = self._load_lora_weight(self.model, path, is_trainable)
        # loading token classifier
        classifier_ckpt = torch.load(osp.join(path, 'classifier.bin'), map_location=torch.device('cuda'))
        classifier_state_dict = self.model.classifier.state_dict()
        classifier_trained_dict = {k: v for k, v in classifier_ckpt['state_dict'].items() if k in classifier_state_dict}
        classifier_state_dict.update(classifier_trained_dict)
        self.model.classifier.load_state_dict(classifier_state_dict)

    def print_trainable_parameters(self):
        total = 0
        trainable = 0

        for k, v in self.model.named_parameters():
            total += v.numel()
            if v.requires_grad:
                trainable += v.numel()
        self.accelerator.log({
            'Trainable Params': f'trainable params: {trainable} || all params: {total} || trainable%: {trainable / total:.15f}'
        })

    @staticmethod
    def _save_lora_weight(unwrapped_model, path: str):
        if hasattr(unwrapped_model, "embedding"):
            if hasattr(unwrapped_model.embedding, "lora_embedding"):
                unwrapped_model.embedding.lora_embedding.save_pretrained(path)
                print(f"load lora weights in :{path}")
        return unwrapped_model

    @staticmethod
    def _load_lora_weight(unwrapped_model, path: str, is_trainable=False):
        lora_weight_tensor = {}
        with safe_open(osp.join(path, 'adapter_model.safetensors'), framework='pt') as file:
            for key in file.keys():
                lora_weight_tensor[key.replace('weight', 'default.weight')] = file.get_tensor(key)
        for name, weight in unwrapped_model.lora_embedding.named_parameters():
            if name not in lora_weight_tensor.keys():
                continue
            if weight.requires_grad:
                assert weight.data.size() == lora_weight_tensor[name].size(), f'Got an invalid key: `{name}`!'
                weight.data.copy_(lora_weight_tensor[name])
                if not is_trainable:
                    weight.requires_grad = False
        return unwrapped_model

    @abstractmethod
    def train(self, **kwargs):
        """train model"""

    @abstractmethod
    def inference(self, **kwargs):
        """inference model"""


class CustomTrainer(Trainer):

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """
        if "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None
        output = model(inputs["input_ids"])
        loss = F.nll_loss(output.last_hidden_state, labels)
        outputs = {"output": output, "loss": loss, "logits": output.last_hidden_state}
        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if labels is not None:
            unwrapped_model = unwrap_model(model)
            if _is_peft_model(unwrapped_model):
                model_name = unwrapped_model.base_model.model._get_name()
            else:
                model_name = unwrapped_model._get_name()
            if model_name in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values():
                loss = self.label_smoother(outputs, labels, shift_labels=True)
            else:
                loss = self.label_smoother(outputs, labels)
        else:
            if isinstance(outputs, dict) and "loss" not in outputs:
                raise ValueError(
                    "The model did not return a loss from the inputs, only the following keys: "
                    f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}."
                )
            # We don't use .loss here since the model may return tuples instead of ModelOutput.
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        return (loss, outputs) if return_outputs else loss
