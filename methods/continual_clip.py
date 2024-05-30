import datetime
import gc
import time

from sklearn.metrics import confusion_matrix
import torch
import logging

from tqdm import tqdm

from methods._trainer import _Trainer

logger = logging.getLogger()


class ContinualCLIP(_Trainer):

    def __init__(self, **kwargs):
        super(ContinualCLIP, self).__init__(**kwargs)

    def online_step(self, images, labels, idx):
        self.add_new_class(labels)
        self.model.update_class_names(self.exposed_classes_names)

        # # zero-shot, don't need to train
        # for j in range(len(labels)):
        #     labels[j] = self.exposed_classes.index(labels[j].item())

        # images = images.to(self.device)
        # labels = labels.to(self.device)
        # images = self.train_transform(images)

        # logit = self.model(images)
        # preds = torch.argmax(logit, dim=-1)
        # _, preds = logit.topk(self.topk, 1, True, True)
        # _correct = torch.sum(preds == labels.unsqueeze(1)).item()
        # _num_data = labels.size(0)

        # del (images, labels)
        # gc.collect()
        # return 0, _correct / _num_data
        del (images, labels)
        gc.collect()
        return -1, -1

    def report_training(self, sample_num, train_loss, train_acc):
        pass

    def report_test(self, sample_num, avg_loss, avg_acc):
        logging.info(
            f"Test | Sample # {sample_num} | test_acc {avg_acc:.4f} | "
            f"running_time {datetime.timedelta(seconds=int(time.time() - self.start_time))} | "
            f"ETA {datetime.timedelta(seconds=int((time.time() - self.start_time) * (self.total_samples-sample_num) / sample_num))}"
        )

    def online_train(self, data):
        pass

    def online_before_task(self, task_id):
        pass

    def online_after_task(self, task_id):
        pass

    def online_evaluate(self, test_loader):
        total_correct, total_num_data, total_loss = 0.0, 0.0, 0.0
        correct_l = torch.zeros(self.n_classes)
        num_data_l = torch.zeros(self.n_classes)
        label, pred_list = [], []

        self.model.eval()
        with torch.no_grad():
            for i, data in enumerate(test_loader):
                x, y = data
                for j in range(len(y)):
                    y[j] = self.exposed_classes.index(y[j].item())

                x = x.to(self.device)
                y = y.to(self.device)

                logit = self.model(x)
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

        bk_class_names = self.model.current_class_names
        self.model.current_class_names = []
        self.model.update_class_names(classes_names)

        self.model.eval()
        with torch.no_grad():
            for data in tqdm(test_loader):
                x, y = data

                x = x.to(self.device)
                y = y.to(self.device)

                logit = self.model(x)
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
        self.model.current_class_names = bk_class_names

        return total_acc
