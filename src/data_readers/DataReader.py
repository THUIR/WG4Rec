# coding=utf-8
import os
import pandas as pd
import numpy as np
from collections import Counter
import logging
from utils.dataset import group_user_interactions_df
from utils.global_p import *
import json
import gc


class DataReader(object):
    """
    只负责load数据集文件，记录一些数据集信息
    """

    @staticmethod
    def parse_data_args(parser):
        """
        data loader 的数据集相关的命令行参数
        :param parser:
        :return:
        """
        parser.add_argument('--path', type=str, default=DATASET_DIR,
                            help='Input data dir.')
        parser.add_argument('--dataset', type=str, default='ml100k-1-5',
                            help='Choose a dataset.')
        # parser.add_argument('--sep', type=str, default='\t',
        #                     help='sep of csv file.')
        # parser.add_argument('--seq_sep', type=str, default=',',
        #                     help='sep of sequences in csv file.')
        parser.add_argument('--label', type=str, default=LABEL,
                            help='name of dataset label column.')
        return parser

    # 所有数据转化为dict，使得index访问更快
    # 但大规模数据必须要转化为column->np.array的形式，不然在多线程里会复制造成内存大量增加
    def __init__(self, path, dataset, regenerate, label=LABEL, load_file=True, sep='\t', seq_sep=','):
        """
        初始化
        :param path: 数据集目录
        :param dataset: 数据集名称
        :param label: 标签column的名称
        :param load_data: 是否要载入数据文件，否则只载入数据集信息
        :param sep: csv的分隔符
        :param seq_sep: 变长column的内部分隔符，比如历史记录可能为'1,2,4'
        """
        self.regenerate = regenerate
        self.dataset = dataset
        self.path = os.path.join(path, dataset)
        self.train_file = os.path.join(self.path, dataset + TRAIN_SUFFIX)
        self.validation_file = os.path.join(self.path, dataset + VALIDATION_SUFFIX)
        self.test_file = os.path.join(self.path, dataset + TEST_SUFFIX)
        self.info_file = os.path.join(self.path, dataset + INFO_SUFFIX)
        self.sep, self.seq_sep = sep, seq_sep
        self.load_file = load_file
        self.label = label

        self.train_df, self.validation_df, self.test_df = None, None, None
        self.data_columns = None
        self.column_max, self.column_min = {}, {}

        self.load_data()
        self.load_info()
        # gc.collect()
        logging.info('data columns: {}'.format(self.data_columns))

    def data_df2dict(self, df, skip_columns=None):
        if skip_columns is None:
            skip_columns = []
        df = df.to_dict(orient='list')
        for k in df:
            if k in skip_columns:
                continue
            df[k] = np.array(df[k])
            self.add_info(k, df[k].max())
            self.add_info(k, df[k].min())
        return df

    def load_data(self):
        self.load_tvt()

    def load_tvt(self):
        """
        载入训练集、验证集、测试集csv文件
        :return:
        """
        if os.path.exists(self.train_file):
            logging.info("load train csv...")
            train_df = pd.read_csv(self.train_file, sep=self.sep)
            logging.info("size of train: %d" % len(train_df))
            self.train_df = self.data_df2dict(train_df)
            logging.info("train label: " + str(dict(Counter(self.train_df[self.label]).most_common())))
        if os.path.exists(self.validation_file):
            logging.info("load validation csv...")
            validation_df = pd.read_csv(self.validation_file, sep=self.sep)
            logging.info("size of validation: %d" % len(validation_df))
            self.validation_df = self.data_df2dict(validation_df)
            logging.info("validation label: " + str(dict(Counter(self.validation_df[self.label]).most_common())))
        if os.path.exists(self.test_file):
            logging.info("load test csv...")
            test_df = pd.read_csv(self.test_file, sep=self.sep)
            logging.info("size of test: %d" % len(test_df))
            self.test_df = self.data_df2dict(test_df)
            if self.label in self.test_df:
                logging.info("test label: " + str(dict(Counter(self.test_df[self.label]).most_common())))

    def load_info(self):
        self.label_min = self.column_min[self.label]
        self.label_max = self.column_max[self.label]
        self.data_columns = [c for c in self.train_df if c != self.label]

    def add_info(self, column, value):
        if column not in self.column_max:
            self.column_max[column] = value
        else:
            self.column_max[column] = max(value, self.column_max[column])
        if column not in self.column_min:
            self.column_min[column] = value
        else:
            self.column_min[column] = min(value, self.column_min[column])

    def feature_info(self, model_name):
        feature_dims = 0
        feature_min, feature_max = [], []
        for f in self.data_columns:
            feature_min.append(feature_dims)
            feature_dims += int(self.column_max[f] + 1)
            feature_max.append(feature_dims - 1)
        logging.info('Model # of features %d' % len(self.data_columns))
        logging.info('Model # of feature dims %d' % feature_dims)
        return self.data_columns, feature_dims, feature_min, feature_max
