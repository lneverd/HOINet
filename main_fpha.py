import argparse
import os
import pickle
import random
import shutil
import sys
import time
from collections import OrderedDict
import csv
import numpy as np
import glob
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import yaml
from tqdm import tqdm
from utils.loss import get_loss_func
from matplotlib import pyplot as plt


def init_seed(seed):
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    # torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])  # import return model
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')


def get_parser():
    # parameter priority: command line > config > default
    parser = argparse.ArgumentParser(description='Skeleton-based Action Recgnition')
    parser.add_argument('--seed', type=int, default=1, help='seed')
    parser.add_argument('--work_dir', default='./exp/fpha', help='the work folder for storing results')
    parser.add_argument('--config', default='./config/fpha/fpha.yaml', help='path to the configuration file')

    # processor
    parser.add_argument('--run_mode', default='train', help='must be train or test')
    parser.add_argument('--save_score', type=str2bool, default=False,
                        help='if ture, the classification score will be stored')

    # visulize and debug
    parser.add_argument('--save_epoch', type=int, default=80, help='the start epoch to save model (#iteration)')
    parser.add_argument('--eval_interval', type=int, default=2, help='the interval for evaluating models (#iteration)')
    parser.add_argument('--print_log', type=str2bool, default=True, help='print logging or not')
    parser.add_argument('--show_topk', type=int, default=[1, 5], nargs='+', help='which Top K accuracy will be shown')

    # feeder
    parser.add_argument('--feeder', default='feeders.feeder_fpha.Feeder', help='data loader will be used')
    parser.add_argument('--num_worker', type=int, default=8, help='the number of worker for data loader')
    parser.add_argument('--train_feeder_args', default=dict(), help='the arguments of data loader for training')
    parser.add_argument('--test_feeder_args', default=dict(), help='the arguments of data loader for test')

    # model
    parser.add_argument('--model', default=None, help='the model will be used')
    parser.add_argument('--model_args', default=dict(), help='the arguments of model')
    parser.add_argument('--weights', default='D:/Downloads/fpha final 93.57/fpha.pt', help='the weights for model testing')
    parser.add_argument('--ignore_weights', type=str, default=[], nargs='+',
                        help='the name of weights which will be ignored in the initialization')

    # optim
    parser.add_argument('--base_lr', type=float, default=0.1, help='initial learning rate')
    parser.add_argument('--step', type=int, default=[60, 80], nargs='+',
                        help='the epoch where optimizer reduce the learning rate')
    parser.add_argument('--cuda_visible_device', default='0', help='')
    parser.add_argument('--device', type=int, default=[0], nargs='+',
                        help='the indexes of GPUs for training or testing')

    parser.add_argument('--optimizer', default='SGD', help='type of optimizer')
    parser.add_argument('--nesterov', type=str2bool, default=False, help='use nesterov or not')
    parser.add_argument('--batch_size', type=int, default=256, help='training batch size')
    parser.add_argument('--test_batch_size', type=int, default=256, help='test batch size')
    parser.add_argument('--start_epoch', type=int, default=0, help='start training from which epoch')
    parser.add_argument('--num_epoch', type=int, default=80, help='stop training in which epoch')
    parser.add_argument('--weight_decay', type=float, default=0.0005, help='weight decay for optimizer')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1, help='decay rate for learning rate')
    parser.add_argument('--warm_up_epoch', type=int, default=5)
    parser.add_argument('--optimizer_betas', type=float, default=[0.9, 0.999])

    parser.add_argument('--loss', default='CrossEntropy', help='the loss will be used')
    parser.add_argument('--loss_args', default=dict(), help='the arguments of loss')

    parser.add_argument('--noun_loss_weight', type=float, default=0, help='the arguments of loss weight')
    parser.add_argument('--verb_loss_weight', type=float, default=0, help='the arguments of loss weight')
    parser.add_argument('--action_loss_weight', type=float, default=0, help='the arguments of loss weight')
    parser.add_argument('--feat_loss_weight', type=float, default=0, help='the arguments of loss weight')

    return parser


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].contiguous().view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def get_total_loss(obj_loss, verb_loss, action_loss, l1_loss, noun_loss_weight, verb_loss_weight, action_loss_weight,
                   feat_loss_weight):
    total_loss = noun_loss_weight * obj_loss + verb_loss_weight * verb_loss + action_loss_weight * action_loss + feat_loss_weight * l1_loss
    return total_loss


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.value = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, value, n=1):
        self.value = value
        self.sum += value * n
        self.count += n
        self.avg = self.sum / self.count


class Processor():
    """ Processor for Skeleton-based Action Recgnition """

    def __init__(self, arg):
        self.arg = arg
        self.global_step = 0
        self.lr = self.arg.base_lr
        self.best_acc = 0
        self.noun_loss_weight = self.arg.noun_loss_weight
        self.verb_loss_weight = self.arg.verb_loss_weight
        self.action_loss_weight = self.arg.action_loss_weight
        self.feat_loss_weight = self.arg.feat_loss_weight
        if not os.path.exists(self.arg.work_dir):
            os.makedirs(self.arg.work_dir)
        self.load_model()
        self.load_data()
        self.action2verb_tensor = torch.tensor([
            0,  # action 0 -> open_juice_bottle -> open (verb 0)
            1,  # action 1 -> close_juice_bottle -> close (verb 1)
            2,  # action 2 -> pour_juice_bottle -> pour (verb 2)
            0,  # action 3 -> open_peanut_butter -> open (verb 0)
            1,  # action 4 -> close_peanut_butter -> close (verb 1)
            3,  # action 5 -> prick -> prick (verb 3)
            4,  # action 6 -> sprinkle -> sprinkle (verb 4)
            5,  # action 7 -> scoop_spoon -> scoop (verb 5)
            6,  # action 8 -> put_sugar -> put (verb 6)
            7,  # action 9 -> stir -> stir (verb 7)
            0,  # action 10 -> open_milk -> open (verb 0)
            1,  # action 11 -> close_milk -> close (verb 1)
            2,  # action 12 -> pour_milk -> pour (verb 2)
            8,  # action 13 -> drink_mug -> drink (verb 8)
            6,  # action 14 -> put_tea_bag -> put (verb 6)
            6,  # action 15 -> put_salt -> put (verb 6)
            0,  # action 16 -> open_liquid_soap -> open (verb 0)
            1,  # action 17 -> close_liquid_soap -> close (verb 1)
            2,  # action 18 -> pour_liquid_soap -> pour (verb 2)
            9,  # action 19 -> wash_sponge -> wash (verb 9)
            10,  # action 20 -> flip_sponge -> flip (verb 10)
            11,  # action 21 -> scratch_sponge -> scratch (verb 11)
            12,  # action 22 -> squeeze_sponge -> squeeze (verb 12)
            0,  # action 23 -> open_soda_can -> open (verb 0)
            13,  # action 24 -> use_flash -> use (verb 13)
            14,  # action 25 -> write -> write (verb 14)
            15,  # action 26 -> tear_paper -> tear (verb 15)
            12,  # action 27 -> squeeze_paper -> squeeze (verb 12)
            0,  # action 28 -> open_letter -> open (verb 0)
            16,  # action 29 -> take_letter_from_enveloppe -> take (verb 16)
            17,  # action 30 -> read_letter -> read (verb 17)
            10,  # action 31 -> flip_pages -> flip (verb 10)
            13,  # action 32 -> use_calculator -> use (verb 13)
            18,  # action 33 -> light_candle -> light (verb 18)
            19,  # action 34 -> charge_cell_phone -> charge (verb 19)
            20,  # action 35 -> unfold_glasses -> unfold (verb 20)
            21,  # action 36 -> clean_glasses -> clean (verb 21)
            0,  # action 37 -> open_wallet -> open (verb 0)
            22,  # action 38 -> give_coin -> give (verb 22)
            23,  # action 39 -> receive_coin -> receive (verb 23)
            22,  # action 40 -> give_card -> give (verb 22)
            2,  # action 41 -> pour_wine -> pour (verb 2)
            24,  # action 42 -> toast_wine -> toast (verb 24)
            25,  # action 43 -> handshake -> handshake (verb 25)
            26  # action 44 -> high_five -> high_five (verb 26)
        ], device=self.output_device)
        self.action2object_tensor = torch.tensor([
            8,  # action 0 -> open_juice_bottle -> juice (object 8)
            8,  # action 1 -> close_juice_bottle -> juice (object 8)
            8,  # action 2 -> pour_juice_bottle -> juice (object 8)
            15,  # action 3 -> open_peanut_butter -> peanut_butter (object 15)
            15,  # action 4 -> close_peanut_butter -> peanut_butter (object 15)
            5,  # action 5 -> prick -> fork (object 5)
            19,  # action 6 -> sprinkle -> spoon (object 19)
            19,  # action 7 -> scoop_spoon -> spoon (object 19)
            19,  # action 8 -> put_sugar -> spoon (object 19)
            19,  # action 9 -> stir -> spoon (object 19)
            12,  # action 10 -> open_milk -> milk (object 12)
            12,  # action 11 -> close_milk -> milk (object 12)
            12,  # action 12 -> pour_milk -> milk (object 12)
            13,  # action 13 -> drink_mug -> mug (object 13)
            22,  # action 14 -> put_tea_bag -> tea_bag (object 22)
            17,  # action 15 -> put_salt -> salt (object 17)
            10,  # action 16 -> open_liquid_soap -> liquid_soap (object 10)
            10,  # action 17 -> close_liquid_soap -> liquid_soap (object 10)
            10,  # action 18 -> pour_liquid_soap -> liquid_soap (object 10)
            21,  # action 19 -> wash_sponge -> sponge (object 21)
            21,  # action 20 -> flip_sponge -> sponge (object 21)
            21,  # action 21 -> scratch_sponge -> sponge (object 21)
            21,  # action 22 -> squeeze_sponge -> sponge (object 21)
            18,  # action 23 -> open_soda_can -> soda_can (object 18)
            20,  # action 24 -> use_flash -> spray (object 20)
            16,  # action 25 -> write -> pen (object 16)
            14,  # action 26 -> tear_paper -> paper (object 14)
            14,  # action 27 -> squeeze_paper -> paper (object 14)
            9,  # action 28 -> open_letter -> letter (object 9)
            9,  # action 29 -> take_letter_from_enveloppe -> letter (object 9)
            14,  # action 30 -> read_letter -> paper (object 14)
            0,  # action 31 -> flip_pages -> book (object 0)
            1,  # action 32 -> use_calculator -> calculator (object 1)
            11,  # action 33 -> light_candle -> match (object 11)
            3,  # action 34 -> charge_cell_phone -> cell_charger (object 3)
            6,  # action 35 -> unfold_glasses -> glasses (object 6)
            6,  # action 36 -> clean_glasses -> glasses (object 6)
            23,  # action 37 -> open_wallet -> wallet (object 23)
            4,  # action 38 -> give_coin -> coin (object 4)
            4,  # action 39 -> receive_coin -> coin (object 4)
            2,  # action 40 -> give_card -> card (object 2)
            24,  # action 41 -> pour_wine -> wine_bottle (object 24)
            25,  # action 42 -> toast_wine -> wine_glass (object 25)
            7,  # action 43 -> handshake -> hand (object 7)
            7  # action 44 -> high_five -> hand (object 7)
        ], device=self.output_device)

        if arg.run_mode == 'train':
            if not arg.train_feeder_args['debug']:
                self.load_optimizer()

        self.model = self.model.cuda(self.output_device)

        if type(self.arg.device) is list:
            if len(self.arg.device) > 1:
                self.model = nn.DataParallel(self.model, device_ids=self.arg.device, output_device=self.output_device)

    def load_data(self):
        Feeder = import_class(self.arg.feeder)
        self.data_loader = dict()
        if self.arg.run_mode == 'train':
            self.data_loader['train'] = DataLoader(
                dataset=Feeder(**self.arg.train_feeder_args),
                batch_size=self.arg.batch_size,
                shuffle=True,
                num_workers=self.arg.num_worker,
                drop_last=True,
                worker_init_fn=init_seed)
        self.data_loader['test'] = DataLoader(
            dataset=Feeder(**self.arg.test_feeder_args),
            batch_size=self.arg.test_batch_size,
            shuffle=False,
            num_workers=self.arg.num_worker,
            drop_last=False,
            worker_init_fn=init_seed)
        self.print_log('Data load finished')

    def load_model(self):
        output_device = self.arg.device[0] if type(self.arg.device) is list else self.arg.device
        self.output_device = output_device
        Model = import_class(self.arg.model)
        self.model = Model(**self.arg.model_args)
        self.action_loss = get_loss_func(self.arg.loss, self.arg.loss_args).cuda(output_device)
        self.obj_loss = get_loss_func('CrossEntropy', None).cuda(output_device)
        self.verb_loss = get_loss_func(self.arg.loss, self.arg.loss_args).cuda(output_device)
        if self.arg.weights:
            # self.global_step = int(arg.weights[:-3].split('-')[-1])
            self.print_log('Load weights from {}'.format(self.arg.weights))
            if '.pkl' in self.arg.weights:
                with open(self.arg.weights, 'r') as f:
                    weights = pickle.load(f)
            else:
                weights = torch.load(self.arg.weights)

            weights = OrderedDict([[k.split('module.')[-1], v.cuda(output_device)] for k, v in weights.items()])

            keys = list(weights.keys())
            for w in self.arg.ignore_weights:
                for key in keys:
                    if w in key:
                        if weights.pop(key, None) is not None:
                            self.print_log('Sucessfully Remove Weights: {}.'.format(key))
                        else:
                            self.print_log('Can Not Remove Weights: {}.'.format(key))

            try:
                self.model.load_state_dict(weights)
            except:
                state = self.model.state_dict()
                diff = list(set(state.keys()).difference(set(weights.keys())))
                print('Can not find these weights:')
                for d in diff:
                    print('  ' + d)
                state.update(weights)
                self.model.load_state_dict(state)
                # 仅在训练模式下处理 ISTANet 内部的 ResNet50
        if self.arg.run_mode == 'train' and hasattr(self.model, 'resnet'):
            self.print_log('Loading ResNet50 weights and modifying its structure.')
            resnet = self.model.resnet
            # 加载 ResNet50 预训练权重并去除 fc 层的权重
            resnet_weights = torch.load("./exp/h2o/resnet50-0676ba61.pth")  # 替换为你的权重路径
            resnet_weights.pop("fc.weight", None)
            resnet_weights.pop("fc.bias", None)
            # 冻结 conv1 到 layer3
            for param in resnet.parameters():
                param.requires_grad = False
            # 仅 layer4 可训练
            for param in resnet.layer4.parameters():
                param.requires_grad = True
            resnet.load_state_dict(resnet_weights, strict=False)
        self.print_log('Model load finished: ' + self.arg.model)

    def load_optimizer(self):
        if self.arg.optimizer == 'SGD':
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=self.arg.base_lr,
                momentum=0.9,
                nesterov=self.arg.nesterov,
                weight_decay=self.arg.weight_decay)
        elif self.arg.optimizer == 'Adam':
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.arg.base_lr,
                weight_decay=self.arg.weight_decay,
                betas=(self.arg.optimizer_betas[0], self.arg.optimizer_betas[1]))
        elif self.arg.optimizer == 'AdamW':
            self.optimizer = optim.AdamW(
                self.model.parameters(),
                lr=self.arg.base_lr,
                weight_decay=self.arg.weight_decay,
                betas=(self.arg.optimizer_betas[0], self.arg.optimizer_betas[1]))
        else:
            raise ValueError()
        self.print_log('Optimizer load finished: ' + self.arg.optimizer)

    def adjust_learning_rate(self, epoch):
        self.print_log('adjust learning rate, using warm up, epoch: {}'.format(self.arg.warm_up_epoch))
        if self.arg.optimizer == 'SGD' or self.arg.optimizer == 'Adam' or self.arg.optimizer == 'AdamW':
            if epoch < self.arg.warm_up_epoch:
                lr = self.arg.base_lr * (epoch + 1) / self.arg.warm_up_epoch
            else:
                lr = self.arg.base_lr * (self.arg.lr_decay_rate ** np.sum(epoch >= np.array(self.arg.step)))
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
            return lr
        else:
            raise ValueError()

    def print_log(self, str, print_time=True):
        if print_time:
            localtime = time.strftime('%Y-%m-%d %H:%M', time.localtime(time.time()))
            str = "[ " + localtime + ' ] ' + str
        print(str)
        if self.arg.print_log:
            with open('{}/log.txt'.format(self.arg.work_dir), 'a') as f:
                print(str, file=f)

    def train(self, epoch, save_model=False):
        losses = AverageMeter()
        obj_losses = AverageMeter()
        verb_losses = AverageMeter()
        l1_losses = AverageMeter()
        top1 = AverageMeter()
        obj_prec_top1 = AverageMeter()
        verb_prec_top1 = AverageMeter()
        self.model.train()
        self.adjust_learning_rate(epoch)

        for batch, (data, rgb_data, label, sample) in enumerate(
                tqdm(self.data_loader['train'], desc="Training", ncols=100)):
            self.global_step += 1
            with torch.no_grad():
                data = data.float().cuda(self.output_device)
                label = label.long().cuda(self.output_device)
                object_label = self.action2object_tensor[label]
                rgb_data = rgb_data.cuda(self.output_device)
                verb_label = self.action2verb_tensor[label]
            # forward
            obj_logits, verb_logits, output, l1_loss = self.model(data, rgb_data)
            obj_loss = self.obj_loss(obj_logits, object_label)
            verb_loss = self.verb_loss(verb_logits, verb_label)
            action_loss = self.action_loss(output, label)
            l1_loss = l1_loss.mean()
            loss = get_total_loss(obj_loss, verb_loss, action_loss, l1_loss, self.noun_loss_weight,
                                  self.verb_loss_weight, self.action_loss_weight, self.feat_loss_weight)
            # backward
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            prec = accuracy(output.data, label, topk=(1,))
            prec_obj = accuracy(obj_logits.data, object_label, topk=(1,))
            prec_verb = accuracy(verb_logits.data, verb_label, topk=(1,))
            top1.update(prec[0].item(), data.size(0))
            obj_prec_top1.update(prec_obj[0].item(), obj_logits.size(0))
            verb_prec_top1.update(prec_verb[0].item(), verb_logits.size(0))
            losses.update(loss.item())
            obj_losses.update(obj_loss.item())
            verb_losses.update(verb_loss.item())
            l1_losses.update(l1_loss.item())

            self.lr = self.optimizer.param_groups[0]['lr']

        self.print_log(
            'training: epoch: {}, loss: {:.4f}, top1: {:.2f}%, lr: {:.6f}, obj_loss: {:.4f}, verb_loss:{:.4f},l1_loss:{:.4f}, obj_acc:{:.4f}%, verb_acc:{:.4f}%'.format(
                epoch + 1, losses.avg, top1.avg, self.lr, obj_losses.avg, verb_losses.avg, l1_losses.avg,
                obj_prec_top1.avg, verb_prec_top1.avg))

    def eval(self, epoch, save_score=False, loader_name=['test'], wrong_file=None, result_file=None):
        losses = AverageMeter()
        obj_losses = AverageMeter()
        verb_losses = AverageMeter()
        l1_losses = AverageMeter()
        top1 = AverageMeter()
        obj_prec_top1 = AverageMeter()
        verb_prec_top1 = AverageMeter()
        if wrong_file is not None:
            f_w = open(wrong_file, 'w')
        if result_file is not None:
            f_r = open(result_file, 'w')
        self.model.eval()
        for ln in loader_name:
            score_frag = []
            label_list = []
            pred_list = []
            for batch, (data, rgb_data, label, sampie) in enumerate(
                    tqdm(self.data_loader[ln], desc="Evaluating", ncols=100)):
                label_list.append(label)
                with torch.no_grad():
                    data = data.float().cuda(self.output_device)
                    label = label.long().cuda(self.output_device)
                    rgb_data = rgb_data.cuda(self.output_device)
                    object_label = self.action2object_tensor[label]
                    verb_label = self.action2verb_tensor[label]
                    obj_logits, verb_logits, output, l1_loss = self.model(data, rgb_data)
                    obj_loss = self.obj_loss(obj_logits, object_label)
                    verb_loss = self.verb_loss(verb_logits, verb_label)
                    action_loss = self.action_loss(output, label)
                    l1_loss = l1_loss.mean()
                    loss = get_total_loss(obj_loss, verb_loss, action_loss, l1_loss, self.noun_loss_weight,
                                          self.verb_loss_weight, self.action_loss_weight, self.feat_loss_weight)

                    score_frag.append(output.data.cpu().numpy())
                    _, predict_label = torch.max(output.data, 1)
                    pred_list.append(predict_label.data.cpu().numpy())

                prec = accuracy(output.data, label, topk=(1,))
                prec_obj = accuracy(obj_logits.data, object_label, topk=(1,))
                prec_verb = accuracy(verb_logits.data, verb_label, topk=(1,))
                top1.update(prec[0].item(), data.size(0))
                obj_prec_top1.update(prec_obj[0].item(), obj_logits.size(0))
                verb_prec_top1.update(prec_verb[0].item(), verb_logits.size(0))
                losses.update(loss.item())
                obj_losses.update(obj_loss.item())
                verb_losses.update(verb_loss.item())
                l1_losses.update(l1_loss.item())

                if wrong_file is not None or result_file is not None:
                    predict = list(predict_label.cpu().numpy())
                    true = list(label.data.cpu().numpy())
                    for i, x in enumerate(predict):
                        if result_file is not None:
                            f_r.write(str(x) + ',' + str(true[i]) + '\n')
                        if x != true[i] and wrong_file is not None:
                            f_w.write(str(sampie[i]) + ',' + str(x) + ',' + str(true[i]) + '\n')

            score = np.concatenate(score_frag)
            score_dict = dict(zip(self.data_loader[ln].dataset.sample_name, score))

            if top1.avg >= self.best_acc and self.arg.run_mode == 'train':
                state_dict = self.model.state_dict()
                weights = OrderedDict([[k.split('module.')[-1], v.cpu()] for k, v in state_dict.items()])
                torch.save(weights, self.arg.work_dir + '/' + self.arg.work_dir.split('/')[-1] + '.pt')

            self.best_acc = top1.avg if top1.avg > self.best_acc else self.best_acc

            self.print_log(
                'evaluating: loss: {:.4f}, top1: {:.2f}%, best_acc: {:.2f}%,obj_loss: {:.4f},verb_loss:{:.4f},l1_loss:{:.4f},obj_acc:{:.4f}%, verb_acc:{:.4f}%'.format
                (losses.avg, top1.avg, self.best_acc, obj_losses.avg, verb_losses.avg, l1_losses.avg, obj_prec_top1.avg,
                 verb_prec_top1.avg))

            if save_score:
                with open('{}/score.pkl'.format(self.arg.work_dir), 'wb') as f:
                    pickle.dump(score_dict, f)

    def h2o_get_results(self, loader_name=['test'], result_file=None):

        res = {"modality": "train: hand+obj, test: hand+obj", }
        obj = {"modality": "train: RGB, test: RGB", }
        verb = {"modality": "train: hand+obj, test: hand+obj", }

        self.model.eval()
        for ln in loader_name:
            for batch, (data, rgb_data, index) in enumerate(
                    tqdm(self.data_loader[ln], desc="Evaluating", ncols=100)):
                with torch.no_grad():
                    data = data.float().cuda(self.output_device)
                    rgb_data = rgb_data.cuda(self.output_device)
                    obj_logits, verb_logits, output, l1_loss = self.model(data, rgb_data)
                    _, predict_label = torch.max(output.data, 1)
                    _, predict_obj = torch.max(obj_logits.data, 1)
                    _, verb_logits = torch.max(verb_logits.data, 1)
                    pred = predict_label.data.cpu().numpy()
                    pred_obj = predict_obj.data.cpu().numpy()
                    pred_verb = verb_logits.data.cpu().numpy()
                    for i in range(len(pred)):
                        res[str(index[i].data.cpu().numpy() + 1)] = int(pred[i] + 1)
                    for i in range(len(pred_obj)):
                        obj[str(index[i].data.cpu().numpy() + 1)] = int(pred_obj[i] + 1)
                    for i in range(len(pred_verb)):
                        verb[str(index[i].data.cpu().numpy() + 1)] = int(pred_verb[i] + 1)

        out = open(result_file, 'w')
        out_obj = open('./exp/h2o/obj_labels.json', 'w')
        out_verb = open('./exp/h2o/verb_labels.json', 'w')
        json.dump(res, out)
        json.dump(obj, out_obj)
        json.dump(verb, out_verb)

    def asb_get_results(self, loader_name=['test'], result_file=None):

        res = {"task": "recognition", "results": {}}

        softmax = nn.Softmax(dim=1)

        type_name = "default"
        if self.arg.model_args['num_classes'] == 1380:
            type_name = "action"
        elif self.arg.model_args['num_classes'] == 24:
            type_name = "verb"
        elif self.arg.model_args['num_classes'] == 90:
            type_name = "object"
        else:
            raise ValueError('Label type is not action/verb/object.')

        self.model.eval()
        for ln in loader_name:
            for batch, (data, rgb_data, index) in enumerate(
                    tqdm(self.data_loader[ln], desc="Evaluating", ncols=100)):
                with torch.no_grad():
                    data = data.float().cuda(self.output_device)
                    rgb_data = rgb_data.cuda(self.output_device)
                    output = self.model(data, rgb_data)
                    predict_label = softmax(output.data)
                    pred = predict_label.data.cpu().tolist()
                    for i in range(len(pred)):
                        res["results"][str(index[i].data.cpu().numpy())] = {type_name: pred[i]}

        out = open(result_file, 'w')
        json.dump(res, out)

    def start(self):

        if self.arg.run_mode == 'train':

            for argument, value in sorted(vars(self.arg).items()):
                self.print_log('{}: {}'.format(argument, value))

            self.global_step = self.arg.start_epoch * len(self.data_loader['train']) / self.arg.batch_size

            def count_parameters(model):
                return sum(p.numel() for p in model.parameters() if p.requires_grad)

            self.print_log(f'# Parameters: {count_parameters(self.model)}')

            self.print_log('###***************start training***************###')

            for epoch in range(self.arg.start_epoch, self.arg.num_epoch):

                save_model = (epoch + 1 == self.arg.num_epoch)
                self.train(epoch, save_model=save_model)

                if ((epoch + 1) % self.arg.eval_interval == 0):
                    self.eval(epoch, save_score=self.arg.save_score, loader_name=['test'])
            self.print_log('Done.\n')

        elif self.arg.run_mode == 'test':
            if not self.arg.test_feeder_args['debug']:
                weights_path = self.arg.work_dir + '.pt'
                wf = self.arg.work_dir + '/wrong.txt'
                rf = self.arg.work_dir + '/right.txt'
            else:
                wf = rf = None

            if self.arg.weights is None:
                raise ValueError('Please appoint --weights.')
            self.arg.print_log = False
            self.print_log('Model:   {}'.format(self.arg.model))
            self.print_log('Weights: {}'.format(self.arg.weights))
            self.eval(epoch=0, save_score=self.arg.save_score, loader_name=['test'], wrong_file=wf, result_file=rf)
            self.print_log('Done.\n')

        elif self.arg.run_mode == 'h2o_test_get_results':
            if self.arg.weights is None:
                raise ValueError('Please appoint --weights.')
            self.arg.print_log = False
            self.print_log('Model:   {}'.format(self.arg.model))
            self.print_log('Weights: {}'.format(self.arg.weights))
            self.h2o_get_results(loader_name=['test'],
                                 result_file=os.path.join(self.arg.work_dir, 'action_labels.json'))
            self.print_log('Done.\n')

        elif self.arg.run_mode == 'asb_test_get_results':
            if self.arg.weights is None:
                raise ValueError('Please appoint --weights.')
            self.arg.print_log = False
            self.print_log('Model:   {}'.format(self.arg.model))
            self.print_log('Weights: {}'.format(self.arg.weights))
            self.asb_get_results(loader_name=['test'], result_file=os.path.join(self.arg.work_dir, 'preds.json'))
            self.print_log('Done.\n')


if __name__ == '__main__':
    parser = get_parser()

    # load arg form config file
    p = parser.parse_args()
    if p.config is not None:
        with open(p.config, 'r') as f:
            default_arg = yaml.load(f, yaml.FullLoader)
        key = vars(p).keys()
        for k in default_arg.keys():
            if k not in key:
                print('WRONG ARG: {}'.format(k))
                assert (k in key)
        parser.set_defaults(**default_arg)

    arg = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = arg.cuda_visible_device
    init_seed(arg.seed)
    processor = Processor(arg)
    processor.start()
