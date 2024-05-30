import gc
import random
import time
import logging
import datetime
import os.path as osp
from tqdm import tqdm

import numpy as np
from sklearn.metrics import confusion_matrix

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import DataLoader

from methods._trainer import _Trainer
from utils.train_utils import select_optimizer, select_scheduler
from utils.memory import MemoryBatchSampler

logger = logging.getLogger()


class AdapterCLIP(_Trainer):

    def __init__(self, **kwargs):
        super(AdapterCLIP, self).__init__(**kwargs)
        self.batch_exposed_classes = []
        self.batch_exposed_classes_names = []
        self.visible_classes = self.args.get('visible_classes', 'batch')

    def online_step(self, images, labels, idx):
        self.add_new_class(labels)
        self.model.update_class_names(self.exposed_classes_names)

        self.memory_sampler = MemoryBatchSampler(
            self.memory, self.memory_batchsize,
            self.temp_batchsize * self.online_iter * self.world_size)
        self.memory_dataloader = DataLoader(self.train_dataset,
                                            batch_size=self.memory_batchsize,
                                            sampler=self.memory_sampler,
                                            num_workers=4)
        self.memory_provider = iter(self.memory_dataloader)
        # train with augmented batches
        _loss, _acc, _iter = 0.0, 0.0, 0
        for _ in range(int(self.online_iter)):
            loss, acc = self.online_train([images.clone(), labels.clone()])
            _loss += loss
            _acc += acc
            _iter += 1
        self.update_memory(idx, labels)
        del (images, labels)
        gc.collect()
        # torch.cuda.empty_cache()
        return _loss / _iter, _acc / _iter

    def online_train(self, data):
        self.model.train()
        total_loss, total_correct, total_num_data = 0.0, 0.0, 0.0

        if self.visible_classes == 'batch':
            # batch
            train_class_list = self.batch_exposed_classes
            train_class_name_list = self.batch_exposed_classes_names

        else:
            # all
            train_class_list = self.exposed_classes
            train_class_name_list = self.exposed_classes_names

        x, y = data

        if len(self.memory) > 0 and self.memory_batchsize > 0:
            memory_images, memory_labels = next(self.memory_provider)
            for i in memory_labels.unique():
                if i not in train_class_list:
                    train_class_list.append(i)
                    train_class_name_list.append(self.exposed_classes_names[
                        self.exposed_classes.index(i)])
            x = torch.cat([x, memory_images], dim=0)
            y = torch.cat([y, memory_labels], dim=0)

        for j in range(len(y)):
            y[j] = train_class_list.index(y[j].item())

        x = x.to(self.device)
        y = y.to(self.device)

        x = self.train_transform(x)

        text_tokens = self.model.labels_tokenize(train_class_name_list)

        self.optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=self.use_amp):
            logit, image_features, text_features = self.model(x, text_tokens)
            loss = self.criterion(logit, y)
        _, preds = logit.topk(self.topk, 1, True, True)

        self.optimizer.zero_grad()
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.update_schedule()

        if self.args.get('grad_analysis', False):
            self._grad_analysis(image_features.clone().detach(),
                                text_features.clone().detach(),
                                y.clone().detach(), train_class_list)

        total_loss += loss.item()
        total_correct += torch.sum(preds == y.unsqueeze(1)).item()
        total_num_data += y.size(0)

        return total_loss, total_correct / total_num_data

    def online_before_task(self, task_id):
        # Freeze some parameters
        for k, v in self.model.named_parameters():
            if "adaptmlp" not in k and "lora" not in k:
                v.requires_grad = False

        logger.info("Total parameters:\t{}".format(
            sum(p.numel() for p in self.model.parameters())))
        logger.info("Trainable parameters:\t{}".format(
            sum(p.numel() for p in self.model.parameters()
                if p.requires_grad)))

        self.reset_opt()

    def online_after_task(self, task_id):
        pass

    def online_evaluate(self, test_loader):
        total_correct, total_num_data, total_loss = 0.0, 0.0, 0.0
        correct_l = torch.zeros(self.n_classes)
        num_data_l = torch.zeros(self.n_classes)
        label = []
        pred_list = []

        self.model.eval()
        with torch.no_grad():
            for i, data in enumerate(test_loader):
                x, y = data
                for j in range(len(y)):
                    y[j] = self.exposed_classes.index(y[j].item())

                x = x.to(self.device)
                y = y.to(self.device)

                logit, _, _ = self.model(x)
                pred = torch.argmax(logit, dim=-1)
                _, preds = logit.topk(self.topk, 1, True, True)
                total_correct += torch.sum(preds == y.unsqueeze(1)).item()
                total_num_data += y.size(0)

                xlabel_cnt, correct_xlabel_cnt = self._interpret_pred(y, pred)
                correct_l += correct_xlabel_cnt.detach().cpu()
                num_data_l += xlabel_cnt.detach().cpu()

                label += y.tolist()
                pred_list += pred.tolist()

        avg_acc = total_correct / total_num_data
        avg_loss = total_loss / len(test_loader)
        cls_acc = (correct_l / (num_data_l + 1e-5)).numpy().tolist()
        cm = confusion_matrix(label, pred_list)

        eval_dict = {
            "avg_loss": avg_loss,
            "avg_acc": avg_acc,
            "cls_acc": cls_acc,
            "confusion_matrix": cm.tolist()
        }
        return eval_dict

    def offline_evaluate(self, test_loader, classes_names):
        total_correct, total_num_data, total_loss = 0.0, 0.0, 0.0
        correct_l = torch.zeros(self.n_classes)
        num_data_l = torch.zeros(self.n_classes)
        label, pred_list = [], []

        text_tokens = self.model.labels_tokenize(classes_names)
        self.model.eval()
        with torch.no_grad():
            for data in tqdm(test_loader):
                x, y = data

                x = x.to(self.device)
                y = y.to(self.device)

                logit, _, _ = self.model(x, text_tokens)
                pred = torch.argmax(logit, dim=-1)
                _, preds = logit.topk(self.topk, 1, True, True)
                total_correct += torch.sum(preds == y.unsqueeze(1)).item()
                total_num_data += y.size(0)

                xlabel_cnt, correct_xlabel_cnt = self._interpret_pred(y, pred)
                correct_l += correct_xlabel_cnt.detach().cpu()
                num_data_l += xlabel_cnt.detach().cpu()

                label += y.tolist()
                pred_list += pred.tolist()

        total_acc = total_correct / total_num_data

        return total_acc

    def update_memory(self, sample, label):
        # Update memory
        if self.distributed:
            sample = torch.cat(self.all_gather(sample.to(self.device)))
            label = torch.cat(self.all_gather(label.to(self.device)))
            sample = sample.cpu()
            label = label.cpu()
        idx = []
        if self.is_main_process():
            for lbl in label:
                self.seen += 1
                if len(self.memory) < self.memory_size:
                    idx.append(-1)
                else:
                    j = torch.randint(0, self.seen, (1, )).item()
                    if j < self.memory_size:
                        idx.append(j)
                    else:
                        idx.append(self.memory_size)
        # Distribute idx to all processes
        if self.distributed:
            idx = torch.tensor(idx).to(self.device)
            size = torch.tensor([idx.size(0)]).to(self.device)
            dist.broadcast(size, 0)
            if dist.get_rank() != 0:
                idx = torch.zeros(size.item(),
                                  dtype=torch.long).to(self.device)
            dist.barrier()  # wait for all processes to reach this point
            dist.broadcast(idx, 0)
            idx = idx.cpu().tolist()
        # idx = torch.cat(self.all_gather(torch.tensor(idx).to(self.device))).cpu().tolist()
        for i, index in enumerate(idx):
            if len(self.memory) >= self.memory_size:
                if index < self.memory_size:
                    self.memory.replace_data([sample[i], label[i].item()],
                                             index)
            else:
                self.memory.replace_data([sample[i], label[i].item()])

    def update_schedule(self, reset=False):
        if reset:
            self.scheduler = select_scheduler(self.sched_name, self.optimizer,
                                              None)
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = self.lr
        else:
            self.scheduler.step()

    def reset_opt(self):
        self.optimizer = select_optimizer(self.opt_name, self.lr, self.model)
        self.scheduler = select_scheduler(self.sched_name, self.optimizer,
                                          None)

    def add_new_batch_class(self, class_name):
        batch_exposed_classes = []
        for label in class_name:
            if label.item() not in self.batch_exposed_classes:
                self.batch_exposed_classes.append(label.item())
        if self.distributed:
            batch_exposed_classes = torch.cat(
                self.all_gather(
                    torch.tensor(self.batch_exposed_classes,
                                 device=self.device))).cpu().tolist()
            self.batch_exposed_classes = []
            for cls in batch_exposed_classes:
                if cls not in self.batch_exposed_classes:
                    self.batch_exposed_classes.append(cls)
        self.batch_exposed_classes_names = [
            self.train_dataset.classes_names[i]
            for i in self.batch_exposed_classes
        ]

    def add_new_class(self, class_name):
        _old_num = len(self.exposed_classes)
        super().add_new_class(class_name)

        self.batch_exposed_classes = []
        self.batch_exposed_classes_names = []
        if self.memory_size > 0:
            self.batch_exposed_classes = self.exposed_classes
            self.batch_exposed_classes_names = self.exposed_classes_names
        else:
            self.add_new_batch_class(class_name)

    def report_training(self, sample_num, train_loss, train_acc):
        print(
            f"Train | Sample # {sample_num} | train_loss {train_loss:.4f} | train_acc {train_acc:.4f} | "
            f"lr {self.optimizer.param_groups[0]['lr']:.6f} | "
            f"Num_Classes {len(self.exposed_classes)} | "
            f"Num_Batch_Classes {len(self.batch_exposed_classes)} | "
            f"running_time {datetime.timedelta(seconds=int(time.time() - self.start_time))} | "
            f"ETA {datetime.timedelta(seconds=int((time.time() - self.start_time) * (self.total_samples-sample_num) / sample_num))}"
        )
