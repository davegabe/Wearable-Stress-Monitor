from data_provider.data_factory import data_provider
from utils.pot import pot_eval
from utils.tools import adjust_learning_rate, adjustment
from utils.tools import NativeScalerWithGradNormCount as NativeScaler
from utils.losses import mape_loss, mase_loss, smape_loss
from utils.dataloader import BalancedDataLoaderIterator
from utils.layer_decay import param_groups_lrd
from utils.ddp import get_world_size, is_main_process, gather_tensors_from_all_gpus

from sklearn.metrics import roc_auc_score, average_precision_score

import torch
import torch.nn as nn
from torch import optim
import torch.distributed as dist

import os
import time
import warnings
import numpy as np
import yaml
import wandb
import sys
import copy
from tqdm import tqdm

warnings.filterwarnings('ignore')


def apply_random_mask_for_imputation(x, patch_len, mask_rate):
    """
    Apply a random mask to the input tensor.

    Parameters:
    x (torch.Tensor): The input tensor with shape [B, T, N].
    patch_len (int): The length of each patch.
    mask_rate (float): The proportion of the tensor to be masked.

    Returns:
    torch.Tensor: The masked input tensor.
    torch.Tensor: The mask tensor.
    """
    B, T, N = x.shape
    num_keep = int((T // patch_len) * (1 - mask_rate))

    # Generate random noise and sort it
    noise = torch.rand(B, T // patch_len, device=x.device)
    ids_shuffle = torch.argsort(noise, dim=1)
    ids_restore = torch.argsort(ids_shuffle, dim=1)

    # Select indices to keep
    ids_keep = ids_shuffle[:, :num_keep]
    mask = torch.zeros([B, T], device=x.device)
    mask[:, :num_keep] = 1
    # unshuffle to get the binary mask
    mask = torch.gather(mask, dim=1, index=ids_restore)

    # Expand the mask to the original shape
    mask = mask.unsqueeze(-1).repeat(1, 1, patch_len).view(B, T)
    mask = mask.unsqueeze(-1).repeat(1, 1, N)

    # Apply the mask
    x_masked = x.masked_fill(mask == 0, 0)

    return x_masked, mask


def custom_print_decorator(func):
    def wrapper(*args, **kwargs):
        text = ' '.join(map(str, args))
        if 'file' not in kwargs or kwargs['file'] is None:
            sys.stdout.write(text + '\n')
        else:
            kwargs['file'].write(text + '\n')

        if 'folder' in kwargs and kwargs['folder']:
            with open(f'{kwargs["folder"]}/finetune_output.log', 'a') as log_file:
                log_file.write(text + '\n')
        if 'folder' in kwargs:
            del kwargs['folder']
        if 'file' in kwargs:
            del kwargs['file']

    return wrapper


# Replace print to save all print into log files
print = custom_print_decorator(print)


def read_task_data_config(config_path, task_name=None):
    with open(config_path, 'r') as config_file:
        config = yaml.load(config_file, Loader=yaml.FullLoader)
    task_dataset_config = config.get('task_dataset', {})
    # Replace the task_name with the real task name
    if task_name is not None:
        dataset_name = list(task_dataset_config.keys())[0]
        print(
            f"WARNING: changing the task name for {dataset_name} to {task_name}")
        task_dataset_config[dataset_name]["task_name"] = task_name
        print(f"New task data: {task_dataset_config}")
    return task_dataset_config


def get_task_data_config_list(task_data_config, default_batch_size=None):
    task_data_config_list = []

    for task_name, task_config in task_data_config.items():
        task_config['max_batch'] = default_batch_size
        task_data_config_list.append([task_name, task_config])

    return task_data_config_list


def get_loss_by_name(loss_name):
    if loss_name == 'MSE':
        return nn.MSELoss()
    elif loss_name == 'MAPE':
        return mape_loss()
    elif loss_name == 'MASE':
        return mase_loss()
    elif loss_name == 'SMAPE':
        return smape_loss()
    elif loss_name == 'CE':
        return nn.CrossEntropyLoss()
    else:
        print("no loss function found!")
        exit()


def init_and_merge_datasets(data_loader_list):
    dataloader = BalancedDataLoaderIterator(data_loader_list)
    train_steps = dataloader.__len__()
    return dataloader, train_steps


class Exp_All_Task(object):
    def __init__(self, args, infer=False, task_data_config=None):
        super(Exp_All_Task, self).__init__()

        self.args = args
        if not infer:
            self.ori_task_data_config = read_task_data_config(
                self.args.task_data_config_path, self.args.task_name)
            device_id = dist.get_rank() % torch.cuda.device_count()
        else:
            self.ori_task_data_config = task_data_config
            device_id = 0

        self.ori_task_data_config_list = get_task_data_config_list(
            self.ori_task_data_config, default_batch_size=self.args.batch_size)

        self.task_data_config = self.ori_task_data_config
        self.task_data_config_list = self.ori_task_data_config_list
        self.device_id = device_id
        print("device id", self.device_id)
        self.model = self._build_model(ddp=not infer)

    def _build_model(self, ddp=True):
        import importlib
        module = importlib.import_module("models."+self.args.model)
        model = module.Model(
            self.args, self.task_data_config_list).to(self.device_id)
        if ddp:
            model = nn.parallel.DistributedDataParallel(model, device_ids=[self.device_id],
                                                        find_unused_parameters=True, gradient_as_bucket_view=True, static_graph=False)
        return model

    def _get_data(self, flag, test_anomaly_detection=False, train_ddp=True):
        this_task_data_config = self.task_data_config

        data_set_list = []
        data_loader_list = []

        for task_data_name, task_config in this_task_data_config.items():
            if test_anomaly_detection and (task_config['task_name'] == 'anomaly_detection'):
                train_data_set, train_data_loader = data_provider(
                    self.args, task_config, flag='train', ddp=False)
                data_set, data_loader = data_provider(
                    self.args, task_config, flag, ddp=False)  # ddp false to avoid shuffle
                data_set_list.append([train_data_set, data_set])
                data_loader_list.append([train_data_loader, data_loader])
                print(task_data_name, len(data_set))
            else:
                data_set, data_loader = data_provider(
                    self.args, task_config, flag, ddp=train_ddp)
                data_set_list.append(data_set)
                data_loader_list.append(data_loader)
                print(task_data_name, len(data_set))
        return data_set_list, data_loader_list

    def _select_optimizer(self):
        eff_batch_size = self.args.batch_size * self.args.acc_it * get_world_size()
        real_learning_rate = self.args.learning_rate * eff_batch_size / 32
        self.real_learning_rate = real_learning_rate
        print("base lr: %.2e" % (self.args.learning_rate * 32 / eff_batch_size))
        print("actual lr: %.2e" % real_learning_rate)

        print("accumulate grad iterations: %d" % self.args.acc_it)
        print("effective batch size: %d" % eff_batch_size)
        if self.args.layer_decay is not None:
            print("layer decay: %.2f" % self.args.layer_decay)
            model_without_ddp = self.model.module
            param_groups = param_groups_lrd(model_without_ddp, self.args.weight_decay,
                                            no_weight_decay_list=[
                                                'prompts', 'mask_tokens', 'cls_tokens', 'category_tokens'],
                                            layer_decay=self.args.layer_decay
                                            )
            model_optim = optim.Adam(param_groups, lr=real_learning_rate)
        else:
            model_optim = optim.Adam(self.model.parameters(
            ), lr=real_learning_rate, weight_decay=self.args.weight_decay)
        return model_optim

    def _select_criterion(self, config_list):
        criterion_list = []
        for each_config in config_list:
            if 'loss' in each_config[1]:
                loss_name = each_config[1]['loss']
            else:
                if each_config[1]['task_name'] == 'anomaly_detection':
                    loss_name = 'MSE'
                else:
                    print("this task has no loss now!", folder=self.path)
                    exit()
            criterion_list.append(get_loss_by_name(loss_name))

        return criterion_list

    def choose_training_parts(self, prompt_tune=False):
        for name, param in self.model.named_parameters():
            if prompt_tune:
                if 'prompt_token' in name or 'mask_prompt' in name or 'cls_prompt' in name or 'mask_token' in name or 'cls_token' in name or 'category_token' in name:
                    param.requires_grad = True
                    print("trainable:", name)
                else:
                    param.requires_grad = False
            else:
                param.requires_grad = True

        # if not prompt_tune:
        #     print("all trainable.")

    def train(self, setting, train_ddp=True):
        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path) and is_main_process():
            os.makedirs(path)
        self.path = path

        # Load pretrained weights (Optional)
        if self.args.pretrained_weight is not None:
            if self.args.pretrained_weight == 'auto':
                pretrain_weight_path = os.path.join(
                    self.path, 'pretrain_checkpoint.pth')
            else:
                pretrain_weight_path = self.args.pretrained_weight
            print('loading pretrained model:',
                  pretrain_weight_path, folder=self.path)
            if 'pretrain_checkpoint.pth' in pretrain_weight_path:
                state_dict = torch.load(
                    pretrain_weight_path, map_location='cpu')['student']
                ckpt = {}
                for k, v in state_dict.items():
                    if not ('cls_prompts' in k):
                        ckpt[k] = v
            else:
                ckpt = torch.load(pretrain_weight_path, map_location='cpu')
            msg = self.model.load_state_dict(ckpt, strict=False)
            print(msg, folder=self.path)

        # Data
        _, train_loader_list = self._get_data(flag='train', train_ddp=train_ddp)
        # Since some datasets do not have val set, we use test set and report the performance of last epoch instead of the best epoch.
        test_data_list, test_loader_list = self._get_data(
            flag='test', test_anomaly_detection=True)
        data_loader_cycle, train_steps = init_and_merge_datasets(
            train_loader_list)

        # Model param check
        pytorch_total_params = sum(p.numel() for p in self.model.parameters())
        print("Parameters number for all {} M".format(
            pytorch_total_params/1e6), folder=self.path)
        model_param = []
        for name, param in self.model.named_parameters():
            if ('prompts' in name and 'prompt2forecat' not in name) or 'prompt_token' in name or \
                    'mask_prompt' in name or 'cls_prompt' in name or 'mask_token' in name or 'cls_token' in name or 'category_token' in name:
                print('skip this:', name)
            else:
                model_param.append(param.numel())
        model_total_params = sum(model_param)
        print("Parameters number for UniTS {} M".format(
            model_total_params/1e6), folder=self.path)

        # Optimizer and Criterion
        model_optim = self._select_optimizer()
        criterion_list = self._select_criterion(self.task_data_config_list)
        scaler = NativeScaler()

        # Set up batch size for each task
        if self.args.memory_check:
            self.memory_check(data_loader_cycle, criterion_list)
            torch.cuda.empty_cache()
        torch.cuda.synchronize()
        if get_world_size() > 1:
            dist.barrier()

        for epoch in tqdm(range(self.args.train_epochs+self.args.prompt_tune_epoch), desc='Epochs', position=0, leave=True):
            adjust_learning_rate(model_optim, epoch,
                                 self.real_learning_rate, self.args)
            # Prompt learning
            if (epoch+1) <= self.args.prompt_tune_epoch:
                self.choose_training_parts(prompt_tune=True)
            else:
                self.choose_training_parts(prompt_tune=False)

            train_loss = self.train_one_epoch(
                model_optim, data_loader_cycle, criterion_list, epoch, train_steps, scaler)

            # we report the results of last epoch and not find the best epoch based on val set, since some datasets do not have val set

            if (epoch % 5 == 0) or (epoch == (self.args.train_epochs+self.args.prompt_tune_epoch - 1)):
                self.test(
                    setting, load_pretrain=False, test_data_list=test_data_list, test_loader_list=test_loader_list)

        return self.model

    def train_one_epoch(self, model_optim, data_loader_cycle, criterion_list, epoch, train_steps, scaler):
        current_device = torch.cuda.current_device()
        train_loss_set = []
        acc_it = self.args.acc_it
        max_norm = self.args.clip_grad

        self.model.train()
        epoch_time = time.time()
        self.model.zero_grad(set_to_none=True)
        loss_sum = 0

        for i, (sample_init, task_id) in enumerate(data_loader_cycle):

            task_name = self.task_data_config_list[task_id][1]['task_name']
            small_batch_size = self.task_data_config_list[task_id][1]['max_batch']
            if small_batch_size != self.args.batch_size:
                sample_list = self.split_batch(
                    sample_init, small_batch_size, task_name)
                len_sample_list = len(sample_list)
            else:
                sample_list = [sample_init]
                len_sample_list = 1

            for sample_idx in range(len_sample_list):
                sample = sample_list[sample_idx]
                if task_name == 'anomaly_detection':
                    loss = self.train_anomaly_detection(
                        self.model, sample, criterion_list[task_id], self.task_data_config_list[task_id][1], task_id)
                    loss_scale = 1.0
                

                loss /= acc_it
                loss /= len_sample_list
                if sample_idx < len_sample_list-1:
                    norm_value = scaler(loss*loss_scale, model_optim, clip_grad=max_norm,
                                        parameters=self.model.parameters(), create_graph=False, update_grad=False)
            loss_display = loss.item()*len_sample_list*acc_it
            train_loss_set.append(loss_display)

            norm_value = scaler(loss*loss_scale, model_optim, clip_grad=max_norm,
                                parameters=self.model.parameters(), create_graph=False, update_grad=((i + 1) % acc_it == 0))

            if (i+1) % acc_it == 0:
                model_optim.zero_grad()
            torch.cuda.synchronize()

            loss_sum += loss_display
            loss_sum_display = loss_sum

            del sample_init
            del sample_list
            if torch.cuda.memory_reserved(current_device) > 30*1e9:
                torch.cuda.empty_cache()

            if is_main_process():
                wandb.log(
                    {'train_loss_'+self.task_data_config_list[task_id][0]: loss_display, 'norm_value': norm_value, "loss_sum": loss_sum_display/(i+1)})

            if (i + 1) % 100 == 0:
                if norm_value == None:
                    norm_value = -1
                # if is_main_process():
                #     print("\titers: {0}, epoch: {1} | norm: {2:.2f} | loss: {3:.7f} | current_loss: {4} |current task: {5}".format(
                #         i + 1, epoch + 1, norm_value, loss_sum_display/(i+1), loss_display, task_name, folder=self.path))

        # print("Epoch: {} cost time: {}".format(
        #     epoch + 1, time.time() - epoch_time), folder=self.path)
        train_loss = np.average(train_loss_set)
        torch.cuda.synchronize()
        if get_world_size() > 1:
            dist.barrier()

        return train_loss

    def train_anomaly_detection(self, model, this_batch, criterion, config, task_id):
        task_name = config['task_name']
        features = config['features']

        batch_x, _ = this_batch

        batch_x = batch_x.float().to(self.device_id)

        with torch.cuda.amp.autocast():
            outputs = model(batch_x, None, None,
                            None, task_id=task_id, task_name=task_name)
            f_dim = -1 if features == 'MS' else 0
            outputs = outputs[:, :, f_dim:]
            loss = criterion(outputs, batch_x)

        return loss


    def test(self, setting, load_pretrain=False, test_data_list=None, test_loader_list=None):
        self.path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(self.path) and is_main_process():
            os.makedirs(self.path)
        if test_data_list is None or test_loader_list is None:
            test_data_list, test_loader_list = self._get_data(
                flag='test', test_anomaly_detection=True)
        if load_pretrain:
            if os.path.exists(self.args.pretrained_weight):
                pretrain_weight_path = self.args.pretrained_weight
                print('loading pretrained model:',
                      pretrain_weight_path, folder=self.path)
                if 'pretrain_checkpoint.pth' in pretrain_weight_path:
                    state_dict = torch.load(
                        pretrain_weight_path, map_location='cpu')['student']
                    ckpt = {}
                    for k, v in state_dict.items():
                        if not ('cls_prompts' in k):
                            ckpt[k] = v
                else:
                    ckpt = torch.load(pretrain_weight_path, map_location='cpu')
                msg = self.model.load_state_dict(ckpt, strict=False)
                print(msg)
            else:
                print("no ckpt found!")
                exit()

        total_dict = {}
        avg_anomaly_f_score = []
        for task_id, (test_data, test_loader) in enumerate(zip(test_data_list, test_loader_list)):
            task_name = self.task_data_config_list[task_id][1]['task_name']
            data_task_name = self.task_data_config_list[task_id][0]
            if task_name == 'anomaly_detection':
                f_score, precision, recall, accuracy = self.test_anomaly_detection(
                    setting, test_data, test_loader, data_task_name, task_id)
                total_dict[data_task_name] = {
                    'f_score': f_score, 'precision': precision, 'recall': recall
                }
                avg_anomaly_f_score.append(f_score)

        avg_anomaly_f_score = np.average(avg_anomaly_f_score)

    def test_anomaly_detection(self, setting, test_data, test_loader_set, data_task_name, task_id, ar=None):
        train_loader, test_loader = test_loader_set
        attens_energy = []
        anomaly_criterion = nn.MSELoss(reduce=False)

        self.model.eval()
        # (1) stastic on the train set
        with torch.no_grad():
            for i, (batch_x, batch_y) in enumerate(train_loader):
                batch_x = batch_x.float().to(self.device_id)
                # reconstruction
                outputs = self.model(
                    batch_x, None, None, None, task_id=task_id, task_name='anomaly_detection')
                # criterion
                score = torch.mean(anomaly_criterion(batch_x, outputs), dim=-1)
                score = score.detach().cpu()
                attens_energy.append(score)

            # If distributed, gather the tensors from all gpus
            if get_world_size() > 1:
                attens_energy = gather_tensors_from_all_gpus(
                    attens_energy, self.device_id, to_numpy=True)
        train_energy = np.concatenate(attens_energy, axis=0).reshape(-1)

        # (2) find the threshold
        attens_energy = []
        test_labels = []
        reconstructed = []
        for i, (batch_x, batch_y) in enumerate(test_loader):
            batch_x = batch_x.float().to(self.device_id)
            # reconstruction
            outputs = self.model(batch_x, None, None, None,
                                 task_id=task_id, task_name='anomaly_detection')
            reconstructed.append(outputs.detach().cpu().numpy())
            
            # criterion
            score = torch.mean(anomaly_criterion(batch_x, outputs), dim=-1)
            score = score.detach().cpu()
            attens_energy.append(score)
            test_labels.append(batch_y)

        # If distributed, gather the tensors from all gpus
        if get_world_size() > 1:
            attens_energy = gather_tensors_from_all_gpus(
                attens_energy, self.device_id, to_numpy=True)
        test_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
        combined_energy = np.concatenate([train_energy, test_energy], axis=0)

        test_labels = np.concatenate(test_labels, axis=0).reshape(-1)
        test_labels = np.array(test_labels)
        gt = test_labels.astype(int)

        # AUC metrics
        auc_roc = roc_auc_score(gt, test_energy)
        auc_pr = average_precision_score(gt, test_energy)

        wandb.log({
            "AUC_ROC": auc_roc,
            "AUC_PR": auc_pr
        })

        # Test for different anomaly thresholds
        for anomaly_ratio in range(1, int(self.args.anomaly_ratio)):
            threshold = np.percentile(combined_energy, 100 - anomaly_ratio)

            # (3) evaluation on the test set
            pred = (test_energy > threshold).astype(int)

            # print("pred:   ", pred.shape)
            # print("gt:     ", gt.shape)

            # (4) detection adjustment
            gt, pred = adjustment(gt, pred)

            pred = np.array(pred)
            gt = np.array(gt)

            tp = np.sum((gt == 1) & (pred == 1))
            tn = np.sum((gt == 0) & (pred == 0))
            fp = np.sum((gt == 0) & (pred == 1))
            fn = np.sum((gt == 1) & (pred == 0))

            # Metrics
            precision = tp / (tp + fp)
            recall = tp / (tp + fn)
            f_score = 2 * precision * recall / (precision + recall)
            accuracy = (tp + tn) / (tp + tn + fp + fn)
            MIoU = 0.5 * (tp / (tp + fp + fn) + tn / (tn + fn + fp))

            # print("Accuracy : {:0.4f}, Precision : {:0.4f}, Recall : {:0.4f}, F-score : {:0.4f} ".format(
            #     accuracy, precision,
            #     recall, f_score))

            wandb.log({
                f"f1_score_ar{anomaly_ratio}": f_score,
                f"precision_ar{anomaly_ratio}": precision,
                f"recall_ar{anomaly_ratio}": recall,
                f"accuracy_ar{anomaly_ratio}": accuracy,
                f"MIoU_ar{anomaly_ratio}": MIoU,

                f"threshold_ar{anomaly_ratio}": threshold,

                f"TP_ar{anomaly_ratio}": tp,
                f"TN_ar{anomaly_ratio}": tn,
                f"FP_ar{anomaly_ratio}": fp,
                f"FN_ar{anomaly_ratio}": fn
            })

            if ar == anomaly_ratio:
                print("WARNING: inference mode is on, only return the results for anomaly ratio", ar)
                reconstructed = np.concatenate(reconstructed, axis=0)
                return f_score, precision, recall, accuracy, gt, pred, reconstructed

        return f_score, precision, recall, accuracy

    def split_batch(self, batch, small_batch_size, task_name):
        def split_tensor(tensor, size):
            return [tensor[i:min(i + size, tensor.size(0))] for i in range(0, tensor.size(0), size)]
        if task_name == 'anomaly_detection':
            batch_x, batch_y = batch
            split_batch_x = split_tensor(batch_x, small_batch_size)
            split_batch_y = split_tensor(batch_y, small_batch_size)
            return list(zip(split_batch_x, split_batch_y))
