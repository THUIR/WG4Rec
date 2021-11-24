# -*- coding: UTF-8 -*-

import os
import json
import pandas as pd
import numpy as np
import logging
from collections import Counter

from data_readers.DataReader import DataReader
from utils.dataset import group_user_interactions_df, group_user_interactions_csv
from utils.global_p import *
from utils import utils


class RecDR(DataReader):
    """
    带UserItem信息的DataReader
    """

    @staticmethod
    def parse_data_args(parser):
        parser.add_argument('--drop_neg', type=int, default=1,
                            help='whether drop all negative samples when ranking')
        return DataReader.parse_data_args(parser)

    def __init__(self, path, dataset, *args, **kwargs):
        self.dataset = dataset
        self.path = os.path.join(path, dataset)
        self.user_file = os.path.join(self.path, dataset + USER_SUFFIX)
        self.item_file = os.path.join(self.path, dataset + ITEM_SUFFIX)
        self.train_pos_file = os.path.join(self.path, dataset + TRAIN_POS_SUFFIX)
        self.validation_pos_file = os.path.join(self.path, dataset + VALIDATION_POS_SUFFIX)
        self.test_pos_file = os.path.join(self.path, dataset + TEST_POS_SUFFIX)
        self.train_neg_file = os.path.join(self.path, dataset + TRAIN_NEG_SUFFIX)
        self.validation_neg_file = os.path.join(self.path, dataset + VALIDATION_NEG_SUFFIX)
        self.test_neg_file = os.path.join(self.path, dataset + TEST_NEG_SUFFIX)

        self.user_df, self.item_df = None, None
        self.user_columns, self.item_columns = None, None

        self.train_user_pos, self.train_user_neg = None, None
        self.validation_user_pos, self.validation_user_neg = None, None
        self.test_user_pos, self.test_user_neg = None, None

        DataReader.__init__(self, path=path, dataset=dataset, *args, **kwargs)

    def load_data(self):
        DataReader.load_data(self)
        self.load_user_item()
        self.load_his()

    def load_user_item(self):
        """
        载入用户和物品的csv特征文件
        :return:
        """
        if os.path.exists(self.user_file) and self.load_file:
            logging.info("load user csv...")
            user_df = pd.read_csv(self.user_file, sep='\t').set_index(UID, drop=False)
            user_columns = [c for c in user_df.columns if c != UID]
            self.user_df = self.data_df2dict(user_df)
            for c in user_columns + [UID]:
                self.user_df[c] = utils.pad_id_array(self.user_df[UID], self.user_df[c])

        if os.path.exists(self.item_file) and self.load_file:
            logging.info("load item csv...")
            item_df = pd.read_csv(self.item_file, sep='\t').set_index(IID, drop=False)
            item_columns = [c for c in item_df.columns if c != IID]
            self.item_df = self.data_df2dict(item_df)
            for c in item_columns:
                self.item_df[c] = utils.pad_id_array(self.item_df[IID], self.item_df[c])
            self.item_df[IID] = utils.pad_id_array(self.item_df[IID], self.item_df[IID], sup=None)

    def load_his(self):
        """
        载入数据集按uid合并的历史交互记录，两列 'uid' 和 'iids'，没有则创建
        :return:
        """

        if not self.load_file:
            return

        def build_his(his_df, seqs_sep):
            uids = his_df[UID].tolist()
            iids = his_df[IIDS].astype(str).str.split(seqs_sep).values
            # iids = [i.split(self.seq_sep) for i in his_df['iids'].tolist()]
            iids = [np.array([int(j) for j in i], dtype=np.int64) for i in iids]
            user_his = dict(zip(uids, iids))
            return user_his

        logging.info("load history csv...")
        if not os.path.exists(self.train_pos_file) or self.regenerate == 1:
            logging.info("building train pos history csv...")
            train_pos_df = group_user_interactions_csv(
                in_csv=self.train_file, out_csv=self.train_pos_file, pos_neg=1,
                label=self.label, sep=self.sep, seq_sep=self.seq_sep)
        else:
            train_pos_df = pd.read_csv(self.train_pos_file, sep=self.sep)
        if not os.path.exists(self.validation_pos_file) or self.regenerate == 1:
            logging.info("building validation pos history csv...")
            validation_pos_df = group_user_interactions_csv(
                in_csv=self.validation_file, out_csv=self.validation_pos_file, pos_neg=1,
                label=self.label, sep=self.sep, seq_sep=self.seq_sep)
        else:
            validation_pos_df = pd.read_csv(self.validation_pos_file, sep=self.sep)
        if not os.path.exists(self.test_pos_file) or self.regenerate == 1:
            logging.info("building test pos history csv...")
            test_pos_df = group_user_interactions_csv(
                in_csv=self.test_file, out_csv=self.test_pos_file, pos_neg=1,
                label=self.label, sep=self.sep, seq_sep=self.seq_sep)
        else:
            test_pos_df = pd.read_csv(self.test_pos_file, sep=self.sep)

        if not os.path.exists(self.train_neg_file) or self.regenerate == 1:
            logging.info("building train neg history csv...")
            train_neg_df = group_user_interactions_csv(
                in_csv=self.train_file, out_csv=self.train_neg_file, pos_neg=0,
                label=self.label, sep=self.sep, seq_sep=self.seq_sep)
        else:
            train_neg_df = pd.read_csv(self.train_neg_file, sep=self.sep)
        if not os.path.exists(self.validation_neg_file) or self.regenerate == 1:
            logging.info("building validation neg history csv...")
            validation_neg_df = group_user_interactions_csv(
                in_csv=self.validation_file, out_csv=self.validation_neg_file, pos_neg=0,
                label=self.label, sep=self.sep, seq_sep=self.seq_sep)
        else:
            validation_neg_df = pd.read_csv(self.validation_neg_file, sep=self.sep)
        if not os.path.exists(self.test_neg_file) or self.regenerate == 1:
            logging.info("building test neg history csv...")
            test_neg_df = group_user_interactions_csv(
                in_csv=self.test_file, out_csv=self.test_neg_file, pos_neg=0,
                label=self.label, sep=self.sep, seq_sep=self.seq_sep)
        else:
            test_neg_df = pd.read_csv(self.test_neg_file, sep=self.sep)

        self.train_user_pos = build_his(train_pos_df, self.seq_sep)
        self.validation_user_pos = build_his(validation_pos_df, self.seq_sep)
        self.test_user_pos = build_his(test_pos_df, self.seq_sep)
        self.train_user_neg = build_his(train_neg_df, self.seq_sep)
        self.validation_user_neg = build_his(validation_neg_df, self.seq_sep)
        self.test_user_neg = build_his(test_neg_df, self.seq_sep)

    def load_info(self):
        """
        载入数据集信息文件，如果不存在则创建
        :return:
        """
        DataReader.load_info(self)
        self.user_columns = [c for c in self.user_df if c != UID] if self.user_df is not None else []
        self.item_columns = [c for c in self.item_df if c != IID] if self.item_df is not None else []
        self.user_num, self.item_num = 0, 0
        if UID in self.column_max:
            self.user_num = self.column_max[UID] + 1
        if IID in self.column_max:
            self.item_num = self.column_max[IID] + 1
        logging.info("# of users: %d" % self.user_num)
        logging.info("# of items: %d" % self.item_num)

        all_columns = self.data_columns
        if self.item_columns is not None:
            all_columns = self.item_columns + all_columns
        if self.user_columns is not None:
            all_columns = self.user_columns + all_columns

        # 数据集的特征数目
        self.user_features = [f for f in all_columns if f.startswith('u_')]
        logging.info("# of user features: %d" % len(self.user_features))
        self.item_features = [f for f in all_columns if f.startswith('i_')]
        logging.info("# of item features: %d" % len(self.item_features))
        self.context_features = [f for f in all_columns if f.startswith('c_')]
        logging.info("# of context features: %d" % len(self.context_features))
        self.features = self.user_features + self.item_features + self.context_features
        logging.info("# of features: %d" % len(self.features))

    def feature_info(self, model_name):
        """
        生成模型需要的特征数目、维度等信息，特征最终会在DataProcesso中r转换为multi-hot的稀疏标示，
        例:uid(0-2),iid(0-2),u_age(0-2),i_xx(0-1)，
        那么uid=0,iid=1,u_age=1,i_xx=0会转换为100 010 010 10的稀疏表示0,4,7,9
        :param include_id: 模型是否将uid,iid当成普通特征看待，将和其他特征一起转换到multi-hot embedding中，否则是单独两列
        :param include_item_features: 模型是否包含物品特征
        :param include_user_features: 模型是否包含用户特征
        :param include_context_features: 模型是否包含上下文特征
        :return: 所有特征，例['uid', 'iid', 'u_age', 'i_xx']
                 所有特征multi-hot之后的总维度，例 11
                 每个特征在mult-hot中所在范围的最小index，例[0, 3, 6, 9]
                 每个特征在mult-hot中所在范围的最大index，例[2, 5, 8, 10]
        """
        features = []
        if model_name.include_id:
            if UID in self.column_max:
                features.append(UID)
            if IID in self.column_max:
                features.append(IID)
        if model_name.include_user_features:
            features.extend(self.user_features)
        if model_name.include_item_features:
            features.extend(self.item_features)
        if model_name.include_context_features:
            features.extend(self.context_features)
        feature_dims = 0
        feature_min, feature_max = [], []
        for f in features:
            feature_min.append(feature_dims)
            feature_dims += int(self.column_max[f] + 1)
            feature_max.append(feature_dims - 1)
        logging.info('Model # of features %d' % len(features))
        logging.info('Model # of feature dims %d' % feature_dims)
        return features, feature_dims, feature_min, feature_max

    def drop_neg(self, train=True, validation=True, test=True):
        """
        如果是top n推荐，只保留正例，负例是训练过程中采样得到
        :return:
        """
        logging.info('Drop Neg Samples...')
        if train and self.train_df is not None:
            index = np.where(self.train_df[self.label] > 0)[0]
            for k in self.train_df:
                self.train_df[k] = self.train_df[k][index]
            logging.info("size of train: %d" % len(index))
        if validation and self.validation_df is not None:
            index = np.where(self.validation_df[self.label] > 0)[0]
            for k in self.validation_df:
                self.validation_df[k] = self.validation_df[k][index]
            logging.info("size of validation: %d" % len(index))
        if test and self.test_df is not None:
            if self.label in self.test_df:
                index = np.where(self.test_df[self.label] > 0)[0]
                for k in self.test_df:
                    self.test_df[k] = self.test_df[k][index]
                logging.info("size of test: %d" % len(index))

    def label_01(self, train=True, validation=True, test=True):
        """
        讲label转换为01二值
        :return:
        """
        logging.info("Transform label to 0-1")
        if train and self.train_df is not None:
            self.train_df[self.label] = (self.train_df[self.label] > 0).astype(int)
            logging.info("train label: " + str(dict(
                Counter(self.train_df[self.label]).most_common())))
        if validation and self.validation_df is not None:
            self.validation_df[self.label] = (self.validation_df[self.label] > 0).astype(int)
            logging.info("validation label: " + str(dict(
                Counter(self.validation_df[self.label]).most_common())))
        if test and self.test_df is not None and self.label in self.test_df:
            self.test_df[self.label] = (self.test_df[self.label] > 0).astype(int)
            logging.info("test label: " + str(dict(
                Counter(self.test_df[self.label]).most_common())))
        self.label_min = 0
        self.label_max = 1
