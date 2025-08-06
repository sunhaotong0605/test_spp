import os.path as osp
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from safetensors import safe_open
from tqdm import tqdm

from .base_trainer import BaseTrainer
from .eval_fn import EvalFunc
from .loss_fn import GeneLoss
from ..self_logger import logger
from ..utils.util import EarlyStopper

CKPT_SAVE_STEP = 10


class FTClassifierTrainer(BaseTrainer):
    def __init__(self, **kwargs):
        super(FTClassifierTrainer, self).__init__(**kwargs)

    def train(self, **kwargs):
        self.loss_weight = kwargs.get('loss_weight', 1.)

        early_stopper = EarlyStopper(patience=kwargs.get('patience', 4))
        self.register_wandb(
            user_name=kwargs.get('username'),
            project_name=kwargs.get('project'),
            group=kwargs.get('group'),
            task=kwargs.get('task'),
        )

        self.model = self.accelerator.prepare_model(self.model)
        self.optimizer = self.accelerator.prepare_optimizer(self.optimizer)
        self.train_loader = self.accelerator.prepare_data_loader(self.train_loader)
        self.lr_scheduler = self.accelerator.prepare_scheduler(self.lr_scheduler)

        for eph in range(kwargs.get('epoch', 100)):
            self.model.train()
            batch_iterator = tqdm(self.train_loader,
                                  desc=f'Pid: {self.accelerator.process_index} Eph: {eph:03d} ({early_stopper.counter} / {early_stopper.patience})')
            eph_loss = []
            for idx, sample in enumerate(batch_iterator):
                input_ids, batch_label = sample["input_ids"], sample["labels"]
                logger.info(f"idx:{idx},my_batch_label：{batch_label}")
                with self.accelerator.accumulate(self.model):
                    with self.accelerator.autocast():
                        logger.info(f"my_input_ids:{input_ids.dtype, input_ids.shape}")
                        logger.info(f"my_batch_label:{batch_label, batch_label.dtype, batch_label.shape}")
                        logger.info("************")
                        output = self.model(input_ids)
                        logist = output.last_hidden_state
                        with self.accelerator.autocast():
                            loss = GeneLoss.cross_entropy_loss(
                                logist=logist,
                                target=batch_label,
                            )
                        logger.info(f"Loss:{loss}")
                    self.accelerator.backward(loss)
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                batch_iterator.set_postfix({'Loss': f'{loss.item():.4f}'})
                self.accelerator.log({'loss': loss.item()})
                eph_loss.append(loss.item())
                if self.save_in_batch and idx % CKPT_SAVE_STEP == 0:
                    self.save_ckpt('batch')

                with self.accelerator.main_process_first():
                    self.accelerator.log({'learning rate': self.optimizer.state_dict()['param_groups'][0]['lr']})
            self.lr_scheduler.step()

            if early_stopper(np.mean(eph_loss)):
                if np.isnan(np.mean(eph_loss)):
                    self.accelerator.print(
                        f"\n{datetime.now().strftime('%d_%m_%Y_%H_%M_%S')} Model training ended unexpectedly!")

                self.accelerator.print(
                    f"\nPid: {self.accelerator.process_index}: {datetime.now().strftime('%d_%m_%Y_%H_%M_%S')} Model Training Finished!")

                self.accelerator.wait_for_everyone()
                with self.accelerator.main_process_first():
                    self.accelerator.print(
                        f'\n\nPid: {self.accelerator.process_index}: The best `ckpt` file has saved in {self.best_ckpt_home}')
                self.accelerator.end_training()
                break
            elif early_stopper.counter == 0:
                self.save_ckpt(mode='best')

    def register_model(self, model, **kwargs):
        reuse = kwargs.get('reuse', False)
        is_trainable = kwargs.get('is_trainable', True)
        self.learning_rate = kwargs.get('learning_rate', 5e-3)
        mode = kwargs.get('mode', 'best')

        self.model = model
        if is_trainable:
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    print(f"optimizer param: {name}: {param.shape}")
            self.optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, self.model.parameters()),
                                               lr=self.learning_rate)
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
            elif mode in ['epoch', 'best']:
                self.accelerator.save(trainer_dict, osp.join(self.best_ckpt_home, 'trainer.bin'))
                self.accelerator.save(classifier_dict, osp.join(self.best_ckpt_home, 'classifier.bin'))
                self._save_lora_weight(unwrapped_model, self.batch_ckpt_home)
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

        # loading token classifier
        classifier_ckpt = torch.load(osp.join(path, 'classifier.bin'), map_location=torch.device('cuda'))
        classifier_state_dict = self.model.classifier.state_dict()
        classifier_trained_dict = {k: v for k, v in classifier_ckpt['state_dict'].items() if k in classifier_state_dict}
        classifier_state_dict.update(classifier_trained_dict)
        self.model.classifier.load_state_dict(classifier_state_dict)

    @staticmethod
    def _save_lora_weight(unwrapped_model, path: str):
        if hasattr(unwrapped_model, 'lora_embedding'):
            unwrapped_model.lora_embedding.save_pretrained(path)
            logger.info(f"load lora weights from: {path}")
        else:
            logger.info(f"no lora weights in: {path}")

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

    @torch.no_grad()
    def valid_model_performance(self, **kwargs):
        self.inference(*kwargs)

    @torch.no_grad()
    def inference(self, **kwargs):
        model = self.accelerator.prepare_model(self.model)
        data_loader = self.accelerator.prepare_data_loader(self.test_loader)

        self.model.eval()

        eval_func = EvalFunc()
        labels, predicts = [], []
        for sample in tqdm(data_loader):
            input_ids, batch_label = sample["input_ids"], sample["labels"]
            output = model(input_ids)
            logist = output.last_hidden_state
            pred = logist.argmax(-1).cpu().tolist()
            labels.extend(batch_label.cpu().tolist())
            predicts.extend(pred)
        df = pd.DataFrame({'labels': labels, 'predicts': predicts})
        timestamp = datetime.now().strftime("%d_%m_%Y_%H_%M_%S")
        df.to_csv(osp.join(self.result_home, f"predict_{timestamp}.csv"), index=False)
        print(f"acc:{eval_func.get_acc(labels, predicts)}")
