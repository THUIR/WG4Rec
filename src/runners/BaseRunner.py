# coding=utf-8

import torch
import logging
from time import time
from utils import utils
from utils.global_p import *
from tqdm import tqdm
import gc
import numpy as np
import copy
import os
import random
from collections import defaultdict


class BaseRunner(object):
    @staticmethod
    def parse_runner_args(parser):
        """
        跑模型的命令行参数
        :param parser:
        :return:
        """
        parser.add_argument('--load', type=int, default=0,
                            help='Whether load model and continue to train')
        parser.add_argument('--epoch', type=int, default=100,
                            help='Number of epochs.')
        parser.add_argument('--check_epoch', type=float, default=1,
                            help='Check every xxx epochs.')
        parser.add_argument('--check_train', type=int, default=0,
                            help='Whether evaluate train when each check')
        parser.add_argument('--early_stop', type=int, default=1,
                            help='whether to early-stop.')
        parser.add_argument('--es_worse', type=int, default=10,
                            help='keep worse for es_worse results then early stop.')
        parser.add_argument('--es_long', type=int, default=40,
                            help='keep no better for es_long results then early stop.')
        parser.add_argument('--lr', type=float, default=0.001,
                            help='Learning rate.')
        parser.add_argument('--batch_size', type=int, default=128,
                            help='Batch size during training.')
        parser.add_argument('--eval_batch_size', type=int, default=32,
                            help='Batch size during testing.')
        parser.add_argument('--dropout', type=float, default=0.2,
                            help='Dropout probability for each deep layer')
        parser.add_argument('--l2_bias', type=int, default=0,
                            help='Whether add l2 regularizer on bias.')
        parser.add_argument('--l2', type=float, default=1e-6,
                            help='Weight of l2_regularize in pytorch optimizer.')
        parser.add_argument('--l2s', type=float, default=0.0,
                            help='Weight of l2_regularize in loss.')
        parser.add_argument('--grad_clip_n', type=float, default=10,
                            help='clip_grad_norm_ para, -1 means, no clip')
        parser.add_argument('--grad_clip_v', type=float, default=10,
                            help='clip_grad_value_ para, -1 means, no clip')
        parser.add_argument('--optimizer', type=str, default='Adam',
                            help='optimizer: GD, Adam, Adagrad')
        parser.add_argument('--metrics', type=str, default="RMSE",
                            help='metrics: RMSE, MAE, AUC, F1, Accuracy, Precision, Recall')
        parser.add_argument('--pre_gpu', type=int, default=0,
                            help='Whether put all batches to gpu before run batches. \
                            If 0, dynamically put gpu for each batch.')
        parser.add_argument('--num_workers', type=int, default=16,
                            help='Number of processors when get batches in DataLoader')
        parser.add_argument('--gc_batch', type=int, default=0,
                            help='Run gc.collect after some number of batches')
        parser.add_argument('--gc', type=int, default=0,
                            help='Run gc.collect at some point')
        parser.add_argument('--pin_memory', type=int, default=1,
                            help='pin_memory in DataLoader')
        return parser

    def __init__(self, optimizer, lr, epoch, batch_size, eval_batch_size, dropout, l2, l2s, l2_bias,
                 grad_clip_n, grad_clip_v,
                 metrics, check_epoch, check_train, early_stop, es_worse, es_long,
                 pre_gpu, num_workers, gc_batch, gc, pin_memory):
        """
        初始化
        :param optimizer: 优化器名字
        :param lr: 学习率
        :param epoch: 总共跑几轮
        :param batch_size: 训练batch大小
        :param eval_batch_size: 测试batch大小
        :param dropout: dropout比例
        :param l2: l2权重
        :param metrics: 评价指标，逗号分隔
        :param check_epoch: 每几轮输出check一次模型中间的一些tensor
        :param early_stop: 是否自动提前终止训练
        """
        self.optimizer_name = optimizer
        self.lr = lr
        self.epoch = epoch
        self.batch_size = batch_size
        self.eval_batch_size = eval_batch_size
        self.dropout = dropout
        self.no_dropout = 0.0
        self.l2_weight = l2
        self.l2s_weight = l2s
        self.l2_bias = l2_bias
        self.grad_clip_n = grad_clip_n
        self.grad_clip_v = grad_clip_v
        self.pre_gpu = pre_gpu
        self.num_workers = num_workers
        self.gc_batch = gc_batch
        self.gc = gc
        self.pin_memory = pin_memory

        # 把metrics转换为list of str
        self.metrics = metrics.lower().split(',')
        self.check_epoch = 0 if check_epoch <= 0 else int(check_epoch) if check_epoch >= 1 else check_epoch
        self.check_train = check_train
        self.early_stop = early_stop
        self.es_worse = es_worse
        self.es_long = es_long
        self.time = None

        # 用来记录训练集、验证集、测试集每一轮的评价指标
        self.train_results, self.valid_results, self.test_results = [], [], []

    def _build_optimizer(self, model):
        """
        创建优化器
        :param model: 模型
        :return: 优化器
        """
        weight_p, bias_p = [], []
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if 'bias' in name:
                bias_p.append(p)
            else:
                weight_p.append(p)
        if self.l2_bias == 1:
            optimize_dict = [{'params': weight_p + bias_p, 'weight_decay': self.l2_weight}]
        else:
            optimize_dict = [{'params': weight_p, 'weight_decay': self.l2_weight},
                             {'params': bias_p, 'weight_decay': 0.0}]

        optimizer_name = self.optimizer_name.lower()
        if optimizer_name == 'gd':
            logging.info("Optimizer: GD")
            optimizer = torch.optim.SGD(optimize_dict, lr=self.lr)
        elif optimizer_name == 'adagrad':
            logging.info("Optimizer: Adagrad")
            optimizer = torch.optim.Adagrad(optimize_dict, lr=self.lr)
        elif optimizer_name == 'adam':
            logging.info("Optimizer: Adam")
            optimizer = torch.optim.Adam(optimize_dict, lr=self.lr)
        else:
            logging.error("Unknown Optimizer: " + self.optimizer_name)
            assert self.optimizer_name in ['GD', 'Adagrad', 'Adam']
            optimizer = torch.optim.SGD(optimize_dict, lr=self.lr)
        return optimizer

    def _check_time(self, start=False):
        """
        记录时间用，self.time保存了[起始时间，上一步时间]
        :param start: 是否开始计时
        :return: 上一步到当前位置的时间
        """
        if self.time is None or start:
            self.time = [time()] * 2
            return self.time[0]
        tmp_time = self.time[1]
        self.time[1] = time()
        return self.time[1] - tmp_time

    def batch_add_control(self, batch, train):
        """
        向所有batch添加一些控制信息比如DROPOUT
        :param batch: 由DataProcessor产生
        :param train: 是否是训练阶段
        :return: 所有batch的list
        """
        batch[TRAIN] = train
        batch[DROPOUT] = self.dropout if train else self.no_dropout
        return batch

    def predict(self, model, data, target_keys=PREDICTION):
        """
        预测，不训练
        :param model: 模型
        :param data: 数据dict，由DataProcessor的self.get_*_data()和self.format_data_dict()系列函数产生
        :param data_processor: DataProcessor实例
        :return: prediction 拼接好的 np.array
        """
        if self.gc == 1:
            torch.cuda.empty_cache()
            gc.collect()

        if type(target_keys) is str:
            target_keys = [target_keys]

        result_dict = defaultdict(list)

        if data.buffer_dp_b:
            dl = sorted(data.buffer_b.keys())
            dl = [data.buffer_b[k] for k in dl]
        else:
            dl = torch.utils.data.DataLoader(
                data, batch_size=self.eval_batch_size, shuffle=False, num_workers=self.num_workers,
                collate_fn=data.collect_batch, pin_memory=self.pin_memory == 1)
        model.eval()
        sample_ids = []
        for batch_i, batch in tqdm(enumerate(dl), total=len(dl),
                                   leave=False, ncols=100, mininterval=1, desc='Predict'):
            batch = self.batch_add_control(batch=batch, train=False)
            batch = data.batch_to_gpu(batch)
            out_dict = model.predict(batch)
            if TARGET_KEYS in out_dict:
                target_keys = list(set(target_keys + out_dict[TARGET_KEYS]))
            for key in target_keys:
                if key in out_dict:
                    result_dict[key].append(out_dict[key].detach().cpu().data.numpy())
            sample_ids.append(batch[SAMPLE_ID])
            if self.gc == 1 and self.gc_batch > 0 and batch_i % self.gc_batch == 0:
                torch.cuda.empty_cache()
                gc.collect()

        sample_ids = np.concatenate(sample_ids)
        sample_ids_sorted = np.sort(sample_ids)

        for key in result_dict:
            try:
                result_array = np.concatenate(result_dict[key])
            except ValueError as e:
                # logging.warning("run_some_tensors: %s %s" % (key, str(e)))
                result_array = np.array([d for b in result_dict[key] for d in b])
            if len(sample_ids) == len(result_array):
                reorder_dict = dict(zip(sample_ids, result_array))
                result_dict[key] = np.array([reorder_dict[i] for i in sample_ids_sorted])

        if self.gc == 1:
            torch.cuda.empty_cache()
            gc.collect()
        return result_dict

    def logging_metrics(self, prefix='', suffix='', result_index=-1):
        logging.info("")
        logging.info("{} \t train= {} valid= {} test= {} {} \t "
                     .format(prefix,
                             utils.format_metric(self.train_results[result_index]),
                             utils.format_metric(self.valid_results[result_index]),
                             utils.format_metric(self.test_results[result_index]),
                             suffix)
                     + ','.join(self.metrics))
        return

    def eval_during_fit(self, model, output_dict, train_data, valid_data, test_data, epoch):
        self.check(model, output_dict)
        before_time = time()
        if self.check_train == 1:
            self.train_results.append(self.evaluate(model, train_data))
        else:
            self.train_results.append([output_dict['mean_loss'], output_dict['mean_loss_l2']])
        self.valid_results.append(self.evaluate(model, valid_data))
        if test_data is None:
            self.test_results.append(self.valid_results[-1])
        else:
            self.test_results.append(self.evaluate(model, test_data))
        after_time = time()
        self.logging_metrics(prefix="Epoch {:5} [{:.1f} s]".format(float("%.2f" % epoch), self._check_time()),
                             suffix="[{:.1f} s]".format(after_time - before_time))
        if utils.best_result(self.metrics[0], self.valid_results) == self.valid_results[-1]:
            model.save_model()

    def fit(self, model, data, epoch=-1, valid_data=None, test_data=None):  # fit the results for an input set
        """
        训练
        :param model: 模型
        :param data: 数据dict，由DataProcessor的self.get_*_data()和self.format_data_dict()系列函数产生
        :param data_processor: DataProcessor实例
        :param epoch: 第几轮
        :return: 返回最后一轮的输出，可供self.check函数检查一些中间结果
        """
        if self.gc == 1:
            torch.cuda.empty_cache()
            gc.collect()
        if model.optimizer is None:
            model.optimizer = self._build_optimizer(model)

        dl = torch.utils.data.DataLoader(
            data, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers,
            collate_fn=data.collect_batch, pin_memory=self.pin_memory == 1)
        output_dict = None
        predictions, sample_ids = [], []
        loss_list, loss_l2_list = [], []
        for batch_i, batch in tqdm(enumerate(dl), total=len(dl),
                                   leave=False, desc='Epoch %5d' % (epoch + 1), ncols=100, mininterval=1):
            batch = self.batch_add_control(batch=batch, train=True)
            batch = data.batch_to_gpu(batch)
            model.train()
            model.optimizer.zero_grad()
            output_dict = model(batch)
            l2 = output_dict[LOSS_L2]
            loss = output_dict[LOSS] + l2 * self.l2s_weight
            loss.backward()
            loss_list.append(loss.detach().cpu().data.numpy())
            loss_l2_list.append(l2.detach().cpu().data.numpy())
            predictions.append(output_dict[PREDICTION].detach().cpu().data.numpy())
            sample_ids.append(batch[SAMPLE_ID])
            if self.grad_clip_n > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.grad_clip_n)
            if self.grad_clip_v > 0:
                torch.nn.utils.clip_grad_value_(model.parameters(), self.grad_clip_v)
            model.optimizer.step()
            model.eval()
            if 0 < self.check_epoch < 1:
                if (1.0 * max(batch_i - 1, 0) / len(dl)) % self.check_epoch \
                        > (1.0 * batch_i / len(dl)) % self.check_epoch:
                    output_dict['mean_loss'] = np.mean(loss_list)
                    output_dict['mean_loss_l2'] = np.mean(loss_l2_list)
                    self.eval_during_fit(
                        model=model, output_dict=output_dict, train_data=data,
                        valid_data=valid_data, test_data=test_data, epoch=epoch + 1.0 * (batch_i + 1) / len(dl))
            if self.gc == 1 and self.gc_batch > 0 and batch_i % self.gc_batch == 0:
                torch.cuda.empty_cache()
                gc.collect()

        predictions = np.concatenate(predictions)
        sample_ids = np.concatenate(sample_ids)
        sample_ids_sorted = np.sort(sample_ids)
        reorder_dict = dict(zip(sample_ids, predictions))
        predictions = np.array([reorder_dict[i] for i in sample_ids_sorted])

        if self.gc == 1:
            torch.cuda.empty_cache()
            gc.collect()
        return predictions, output_dict, np.mean(loss_list), np.mean(loss_l2_list)

    def eva_termination(self, model):
        """
        检查是否终止训练，基于验证集
        :param model: 模型
        :return: 是否终止训练
        """
        metric = self.metrics[0]
        valid = self.valid_results
        # 如果已经训练超过20轮，且评价指标越小越好，且评价已经连续五轮非减
        if len(valid) > 20 and metric in utils.LOWER_METRIC_LIST \
                and utils.strictly_increasing(valid[-self.es_worse:]):
            return True
        # 如果已经训练超过20轮，且评价指标越大越好，且评价已经连续五轮非增
        elif len(valid) > 20 and metric not in utils.LOWER_METRIC_LIST \
                and utils.strictly_decreasing(valid[-self.es_worse:]):
            return True
        # 训练好结果离当前已经20轮以上了
        elif len(valid) - valid.index(utils.best_result(metric, valid)) > self.es_long:
            return True
        return False

    def train(self, model, train_data, validation_data, test_data):
        """
        训练模型
        :param model: 模型
        :param data_processor: DataProcessor实例
        :return:
        """

        self._check_time(start=True)  # 记录初始时间

        # 训练之前的模型效果
        init_train = self.evaluate(model, train_data) if self.check_train == 1 else None
        init_valid = self.evaluate(model, validation_data)
        init_test = self.evaluate(model, test_data) if test_data is not None else init_valid
        logging.info("Init: \t train= %s validation= %s test= %s [%.1f s] " % (
            utils.format_metric(init_train), utils.format_metric(init_valid), utils.format_metric(init_test),
            self._check_time()) + ','.join(self.metrics))

        self.train_results.append(init_train)
        self.valid_results.append(init_valid)
        self.test_results.append(init_test)

        try:
            for epoch in range(self.epoch):
                # 每一轮需要重新获得训练数据，因为涉及shuffle或者topn推荐时需要重新采样负例
                train_data.prepare_epoch()
                train_predictions, last_batch, mean_loss, mean_loss_l2 = \
                    self.fit(model, train_data, epoch=epoch, valid_data=validation_data, test_data=test_data)

                # 检查模型中间结果
                if (self.check_epoch == 0 and epoch == self.epoch - 1) or (0 < self.check_epoch < 1) \
                        or (self.check_epoch >= 1 and epoch % self.check_epoch == 0):
                    last_batch['mean_loss'] = mean_loss
                    last_batch['mean_loss_l2'] = mean_loss_l2

                    self.eval_during_fit(model=model, output_dict=last_batch, train_data=train_data,
                                         valid_data=validation_data, test_data=test_data, epoch=epoch + 1)

                    # 检查是否终止训练，基于验证集
                    if self.eva_termination(model) and self.early_stop == 1:
                        logging.info("Early stop at %d based on validation result." % (epoch + 1))
                        break
        except KeyboardInterrupt:
            logging.info("Early stop manually")
            save_here = input("Save here? (1/0) (default 0):")
            if str(save_here).lower().startswith('1'):
                model.save_model()

        if self.epoch > 0:
            model.load_model()

        # Find the best validation result across iterations
        best_valid_score_v = utils.best_result(self.metrics[0], self.valid_results)
        best_epoch_v = self.valid_results.index(best_valid_score_v)
        if self.check_epoch > 1:
            show_epoch_v = (best_epoch_v - 1) * self.check_epoch + 1
        elif self.check_epoch <= 0:
            show_epoch_v = self.epoch
        else:
            show_epoch_v = best_epoch_v * self.check_epoch

        best_test_score_t = utils.best_result(self.metrics[0], self.test_results)
        best_epoch_t = self.test_results.index(best_test_score_t)
        if self.check_epoch > 1:
            show_epoch_t = (best_epoch_t - 1) * self.check_epoch + 1
        elif self.check_epoch <= 0:
            show_epoch_t = self.epoch
        else:
            show_epoch_t = best_epoch_t * self.check_epoch

        if self.epoch <= 0:
            show_epoch_v = show_epoch_t = 0
        self._check_time()
        self.logging_metrics(prefix="Best Iter(validation)= {:5}".format(float("%.2f" % show_epoch_v)),
                             suffix="[{:.1f} s]".format(self.time[1] - self.time[0]),
                             result_index=best_epoch_v)
        self.logging_metrics(prefix="Best Iter(test)= {:5}".format(float("%.2f" % show_epoch_t)),
                             suffix="[{:.1f} s]".format(self.time[1] - self.time[0]),
                             result_index=best_epoch_t)

    def evaluate(self, model, data, metrics=None):  # evaluate the results for an input set
        """
        evaluate模型效果
        :param model: 模型
        :param data: 数据dict，由DataProcessor的self.get_*_data()和self.format_data_dict()系列函数产生
        :param data_processor: DataProcessor
        :param metrics: list of str
        :return: list of float 每个对应一个 metric
        """
        if metrics is None:
            metrics = self.metrics
        predictions = self.predict(model, data)
        return model.evaluate_method(predictions, data, metrics=metrics)

    def check(self, model, out_dict):
        """
        检查模型中间结果
        :param model: 模型
        :param out_dict: 某一个batch的模型输出结果
        :return:
        """
        check = out_dict
        logging.info("")
        for i, t in enumerate(check[CHECK]):
            d = np.array(t[1].detach().cpu())
            logging.info(os.linesep.join([t[0] + '\t' + str(d.shape), np.array2string(d, threshold=20)]))

        logging.info("")
        loss, l2 = check['mean_loss'], check['mean_loss_l2']
        logging.info('mean loss = %.4f, l2 = %.4f, %.4f' % (loss, l2 * self.l2_weight, l2 * self.l2s_weight))
