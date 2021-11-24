# coding=utf-8
import os
import pandas as pd
import numpy as np
from collections import Counter
import logging
from utils.dataset import group_user_interactions_df, leave_out_by_time_df
from utils.global_p import *
import json
import pickle
from tqdm import tqdm
from data_readers.DataReader import DataReader
import gc
from utils import utils


class TextDR(DataReader):
    @staticmethod
    def parse_data_args(parser):
        """
        data reader 的数据集相关的命令行参数
        :param parser:
        :return:
        """
        parser.add_argument('--dict_size', type=int, default=200000,
                            help='Max size of word dictionary.')
        parser.add_argument('--text_max', type=int, default=-1,
                            help="Max number of words in text.")
        return DataReader.parse_data_args(parser)

    def __init__(self, path, dataset, dict_size, text_max, *args, **kwargs):
        self.dict_size = dict_size
        self.text_max = text_max
        self.path = os.path.join(path, dataset)
        self.dictionary_file = os.path.join(self.path, dataset + DICT_SUFFIX)
        self.dictionary_df = None
        DataReader.__init__(self, path=path, dataset=dataset, *args, **kwargs)

    def load_data(self):
        DataReader.load_data(self)
        self.load_text()

    def data_df2dict(self, df, skip_columns=None):
        if skip_columns is None:
            skip_columns = []
        df = DataReader.data_df2dict(self, df, skip_columns=skip_columns + [C_TEXT])
        text, length = self.text_column2array(df[C_TEXT], dict_size=self.dict_size, text_max=-1, pad_text=False)
        df[C_TEXT] = text
        df[C_TEXT_LENGTH] = length
        return df

    def load_text(self):
        if os.path.exists(self.dictionary_file):
            logging.info("load dictionary csv...")
            self.dictionary_df = pd.read_csv(self.dictionary_file, sep=self.sep, na_filter=False).fillna('')
            if self.dict_size > 0:
                self.dictionary_df = self.dictionary_df[self.dictionary_df[C_WORD_ID] < self.dict_size]

    def load_info(self):
        DataReader.load_info(self)
        self.data_columns = [c for c in self.data_columns if c != C_TEXT and c != C_TEXT_LENGTH]
        self.dictionary_size = self.dictionary_df[C_WORD_ID].max() + 1

    @staticmethod
    def text_column2array(text_array, dict_size=-1, text_max=-1, pad_text=False):
        sep_text, sep_length = [], []
        for t in tqdm(text_array, ncols=100, mininterval=1,
                      leave=False, desc='split text array {}'.format(dict_size)):
            t = [int(x) for x in t.split(',') if x != '']
            if dict_size > 0:
                t = [w for w in t if w < dict_size]
            sep_text.append(t)
            sep_length.append(len(t))

        if text_max <= 0:
            text_max = max(sep_length)

        text, text_length = [], []
        for t in tqdm(sep_text, ncols=100, mininterval=1,
                      leave=False, desc='pad text array {}'.format(text_max)):
            t = t[:text_max]
            text_length.append(len(t))
            if pad_text:
                text.append(utils.pad_array(t, text_max, 0))
            else:
                text.append(np.array(t, dtype=np.int64))
        return np.array(text), np.array(text_length)
