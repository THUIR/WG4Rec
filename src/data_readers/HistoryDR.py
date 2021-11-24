# -*- coding: UTF-8 -*-

import os
import json
import pandas as pd
import numpy as np
import logging
from collections import defaultdict

from data_readers.RecDR import RecDR
from utils.dataset import group_user_interactions_df
from utils.global_p import *


class HistoryDR(RecDR):
    """
    带序列历史信息的DataReader
    """

    @staticmethod
    def parse_data_args(parser):
        parser.add_argument('--all_his', type=int, default=0,
                            help='Append all history in the training set')
        parser.add_argument('--max_his', type=int, default=10,
                            help='Max history length. All his if max_his <= 0')
        parser.add_argument('--neg_his', type=int, default=1,
                            help='Whether keep negative interactions in the history')
        parser.add_argument('--neg_column', type=int, default=1,
                            help='Whether keep negative interactions in the history as a single column')
        parser.add_argument('--drop_no_his', type=int, default=1,
                            help='If drop_first > 0, drop the user-item pair with no previous history')
        return RecDR.parse_data_args(parser)

    # TODO: 如果有负反馈，可能要划分session，存储每个session的impression列表，再处理每个交互的session号，供查询对应impression
    def __init__(self, all_his, max_his, neg_his, neg_column, drop_no_his, *args, **kwargs):
        self.all_his = all_his
        self.max_his = max_his
        self.neg_his = neg_his
        self.neg_column = neg_column
        self.drop_no_his = drop_no_his
        RecDR.__init__(self, *args, **kwargs)
        self.append_his()
        if self.drop_no_his == 1:
            self.drop_no_history()

    def append_his(self):
        logging.info('Append history...')

        def get_his_lr(x, his_dict, add_key=False):
            if add_key and x not in his_dict:
                his_dict[x] = np.array([], dtype=np.int64)
            hi = len(his_dict[x])
            lo = 0 if self.max_his <= 0 or self.max_his >= hi else hi - self.max_his
            return [lo, hi]

        if self.all_his == 1 and (self.neg_his == 0 or self.neg_column == 1):
            self.c_history = self.train_user_pos
            for df in [self.train_df, self.validation_df, self.test_df]:
                if df is None:  # 空集合跳过
                    continue
                c_history = [get_his_lr(uid, self.c_history, add_key=True) for uid in df[UID]]
                df[C_HISTORY] = np.array(c_history, dtype=np.int64)

        if self.all_his == 1 and self.neg_his == 1 and self.neg_column == 1:
            self.c_history_neg = self.train_user_neg
            for df in [self.train_df, self.validation_df, self.test_df]:
                if df is None:  # 空集合跳过
                    continue
                c_history_neg = [get_his_lr(uid, self.c_history_neg, add_key=True) for uid in df[UID]]
                df[C_HISTORY_NEG] = np.array(c_history_neg, dtype=np.int64)

        if self.all_his == 0:
            # 存储用户历史和负向历史的列表
            self.c_history = defaultdict(list)
            if self.neg_his == 1 and self.neg_column == 1:
                self.c_history_neg = defaultdict(list)
            for df in [self.train_df, self.validation_df, self.test_df]:
                if df is None:  # 空集合跳过
                    continue
                uids, iids, labels = df[UID], df[IID], df[LABEL]
                c_history, c_history_neg = [], []
                for record in range(len(uids)):
                    uid, iid, label = uids[record], iids[record], labels[record]
                    c_history.append(get_his_lr(uid, self.c_history))
                    if self.neg_his == 1 and self.neg_column == 1:
                        c_history_neg.append(get_his_lr(uid, self.c_history_neg))
                    if label > 0:  # 正例
                        self.c_history[uid].append(iid)
                    elif self.neg_his == 1 and self.neg_column == 0:  # 如果要把正负例放在一起
                        self.c_history[uid].append(-iid)
                    elif self.neg_his == 1 and self.neg_column == 1:  # 如果要把负例单独放在一列
                        self.c_history_neg[uid].append(iid)
                df[C_HISTORY] = np.array(c_history, dtype=np.int64)
                if self.neg_his == 1 and self.neg_column == 1:
                    df[C_HISTORY_NEG] = np.array(c_history_neg, dtype=np.int64)
            self.c_history = dict(self.c_history)
            for k in self.c_history:
                self.c_history[k] = np.array(self.c_history[k], dtype=np.int64)
            if self.neg_his == 1 and self.neg_column == 1:
                self.c_history_neg = dict(self.c_history_neg)
                for k in self.c_history_neg:
                    self.c_history_neg[k] = np.array(self.c_history_neg[k], dtype=np.int64)

    def drop_no_history(self, train=True, validation=True, test=True):
        logging.info('Drop samples with no history...')
        if train and self.train_df is not None:
            index = np.where(self.train_df[C_HISTORY][:, 0] < self.train_df[C_HISTORY][:, 1])[0]
            for k in self.train_df:
                self.train_df[k] = self.train_df[k][index]
            logging.info("size of train: %d" % len(index))
        if validation and self.validation_df is not None:
            index = np.where(self.validation_df[C_HISTORY][:, 0] < self.validation_df[C_HISTORY][:, 1])[0]
            for k in self.validation_df:
                self.validation_df[k] = self.validation_df[k][index]
            logging.info("size of validation: %d" % len(index))
        if test and self.test_df is not None:
            if self.label in self.test_df:
                index = np.where(self.test_df[C_HISTORY][:, 0] < self.test_df[C_HISTORY][:, 1])[0]
                for k in self.test_df:
                    self.test_df[k] = self.test_df[k][index]
                logging.info("size of test: %d" % len(index))
