# coding=utf-8
import copy
from utils import utils
import numpy as np
import logging
import pandas as pd
from tqdm import tqdm
import torch
from collections import defaultdict
from data_processors.HistoryDP import HistoryDP
from utils.global_p import *


class SGNewsDP(HistoryDP):

    @staticmethod
    def parse_dp_args(parser):
        """
        数据处理生成batch的命令行参数
        :param parser:
        :return:
        """
        return HistoryDP.parse_dp_args(parser)

    def __init__(self, max_his, sparse_his, *args, **kwargs):
        self.max_his = max_his
        assert self.max_his > 0 and sparse_his == 0
        HistoryDP.__init__(self, sparse_his=sparse_his, *args, **kwargs)

    def get_item(self, index):
        result = HistoryDP.get_item(self, index)
        c_history = result[C_HISTORY]
        c_history = utils.pad_array(c_history, self.max_his, 0)
        result[C_HISTORY] = np.concatenate([c_history, [result[IID]]])
        result[C_SENT] = np.array(
            [self.data_reader.doc_dict[C_SENT][iid] for iid in result[C_HISTORY]], dtype=np.int64)
        result[C_SENT_LENGTH] = np.array(
            [self.data_reader.doc_dict[C_SENT_LENGTH][iid] for iid in result[C_HISTORY]], dtype=np.int64)
        return result

    def get_batch(self, batch, skip_keys=None, info_keys=None):
        if info_keys is None:
            info_keys = []
        feed_dict = HistoryDP.get_batch(self, batch, skip_keys=skip_keys, info_keys=info_keys + [C_SENT_LENGTH])
        return feed_dict
