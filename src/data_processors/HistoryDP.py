# coding=utf-8

from collections import defaultdict

import numpy as np
from tqdm import tqdm
import torch

from data_processors.RecDP import RecDP
from utils.global_p import *
from utils import utils


class HistoryDP(RecDP):

    @staticmethod
    def parse_dp_args(parser):
        """
        数据处理生成batch的命令行参数
        :param parser:
        :return:
        """
        parser.add_argument('--sparse_his', type=int, default=0,
                            help='Whether use sparse representation of user history.')
        return RecDP.parse_dp_args(parser)

    def __init__(self, sparse_his, *args, **kwargs):
        self.sparse_his = sparse_his
        RecDP.__init__(self, *args, **kwargs)

    def get_item(self, index):
        result = RecDP.get_item(self, index)
        real_index = index
        if index >= len(self):
            real_index = index % len(self)
        uid = result[UID]
        lo, hi = self.df[C_HISTORY][real_index]
        result[C_HISTORY] = self.data_reader.c_history[uid][lo:hi]
        result[C_HISTORY_LENGTH] = hi - lo
        if C_HISTORY_NEG in self.df:  # 如果有负向历史的列
            lo, hi = self.df[C_HISTORY_NEG][real_index]
            result[C_HISTORY_NEG] = self.data_reader.c_history_neg[uid][lo:hi]
            result[C_HISTORY_NEG_LENGTH] = hi - lo
        return result

    def get_batch(self, batch, skip_keys=None, info_keys=None):
        if skip_keys is None:
            skip_keys = []
        if info_keys is None:
            info_keys = []
        feed_dict = RecDP.get_batch(
            self, batch,
            skip_keys=skip_keys + [C_HISTORY, C_HISTORY_NEG],
            info_keys=info_keys + [C_HISTORY_LENGTH, C_HISTORY_NEG_LENGTH])

        rank_n = len(batch[0])
        batches = []
        for rank_i in range(rank_n):
            batches.extend([b[rank_i] for b in batch])

        his_cs = [C_HISTORY]
        if C_HISTORY_NEG in batches[0]:  # 如果有负向历史的列
            his_cs.append(C_HISTORY_NEG)

        for c in his_cs:
            d = [b[c] for b in batches]
            if self.sparse_his == 1:  # 如果是稀疏表示
                x, y, v = [], [], []
                for idx, iids in enumerate(d):
                    x.extend([idx] * len(iids))
                    y.extend([abs(iid) for iid in iids])
                    v.extend([1.0 if iid > 0 else -1.0 if iid < 0 else 0 for iid in iids])
                if len(x) <= 0:
                    i = utils.numpy_to_torch(np.array([[0], [0]], dtype=np.int64), gpu=False)
                    v = utils.numpy_to_torch(np.array([0.0], dtype=np.float32), gpu=False)
                else:
                    i = utils.numpy_to_torch(np.array([x, y], dtype=np.int64), gpu=False)
                    v = utils.numpy_to_torch(np.array(v, dtype=np.float32), gpu=False)
                history = torch.sparse.FloatTensor(
                    i, v, torch.Size([len(d), self.data_reader.item_num]))
                feed_dict[c] = history
            else:
                lengths = [len(iids) for iids in d]
                max_length = max(max(lengths), 1)
                new_d = np.array([utils.pad_array(x, max_length, 0) for x in d], dtype=np.int64)
                feed_dict[c] = utils.numpy_to_torch(new_d, gpu=False)
        return feed_dict
