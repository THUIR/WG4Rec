# coding=utf-8

import random
import gc

import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import Dataset

from utils.global_p import *
from utils.utils import numpy_to_torch


class DataProcessor(Dataset):

    @staticmethod
    def parse_dp_args(parser):
        """
        数据处理生成batch的命令行参数
        :param parser:
        :return:
        """
        parser.add_argument('--buffer_dp', type=int, default=0,
                            help='Whether buffer dp items or not.')
        # parser.add_argument('--buffer_dp_b', type=int, default=1,
        #                     help='Whether buffer dp batches or not for evaluation.')
        return parser

    def __init__(self, df, procedure, data_reader, model_name,
                 buffer_dp, batch_size, eval_batch_size):
        self.data_reader = data_reader
        self.model_name = model_name
        self.procedure = procedure
        self.train = self.procedure == 0
        self.buffer_dp_i = buffer_dp == 1 and self.train
        self.buffer_dp_b = buffer_dp == 1 and not self.train
        self.batch_size = batch_size
        self.eval_batch_size = eval_batch_size

        self.df = df
        self.buffer_i, self.buffer_b = {}, {}
        self.prepare()

    def prepare(self):
        self.prepare_epoch()
        if self.buffer_dp_i:
            self.build_buffer_i()
        if self.buffer_dp_b:
            self.build_buffer_b()

    def prepare_epoch(self):
        pass

    def build_buffer_i(self):
        for i in tqdm(range(len(self)),
                      leave=False, ncols=100, mininterval=1, desc=str(self.procedure) + '-Build Buffer I'):
            self.buffer_i[i] = self.get_item(i)

    def build_buffer_b(self):
        buffer_b = {}
        batch_size = self.eval_batch_size
        total_batch = int((len(self) + batch_size - 1) / batch_size)
        for batch_i in tqdm(range(total_batch),
                            leave=False, ncols=100, mininterval=1, desc=str(self.procedure) + '-Build Buffer B'):
            batch_start = batch_i * batch_size
            batch_end = min(len(self), batch_start + batch_size)
            batch = self.get_batch([self.__getitem__(i) for i in range(batch_start, batch_end)])
            buffer_b['{}-{}'.format(batch_start, batch_end - 1)] = batch
        self.buffer_b = buffer_b

    def __len__(self):
        for k in self.df:
            return len(self.df[k])

    def __getitem__(self, index):
        if self.buffer_dp_b and len(self.buffer_b) > 0:
            return index
        if self.buffer_dp_i and index in self.buffer_i:
            return self.buffer_i[index]
        return self.get_item(index)

    def collect_batch(self, batch):
        if self.buffer_dp_b and len(self.buffer_b) > 0:
            return self.buffer_b['{}-{}'.format(batch[0], batch[-1])]
        return self.get_batch(batch)

    def get_item(self, index):
        feature_columns = self.data_reader.data_columns
        x = [self.df[c][index] for c in feature_columns]
        y = self.df[self.data_reader.label][index]
        base = 0
        for i, feature in enumerate(feature_columns):
            x[i] += base
            base += int(self.data_reader.column_max[feature] + 1)
        feed_dict = {Y: y, SAMPLE_ID: index}
        if len(x) > 0:
            feed_dict[X] = x
        return feed_dict

    def get_batch(self, batch, skip_keys=None, info_keys=None):
        if info_keys is None:
            info_keys = [SAMPLE_ID]
        else:
            info_keys += [SAMPLE_ID]
        result_dict = {}
        for key in batch[0]:
            if skip_keys is not None and key in skip_keys:
                continue
            tmp_d = [b[key] for b in batch]
            if type(tmp_d[0]) is torch.Tensor:
                result_dict[key] = torch.stack(tmp_d, dim=0)
            elif key not in info_keys:
                result_dict[key] = numpy_to_torch(np.array(tmp_d), gpu=False)
            else:
                result_dict[key] = tmp_d
        result_dict[TOTAL_BATCH_SIZE] = len(batch)
        return result_dict

    def get_column(self, column):
        return self.df[column]

    def get_column_names(self):
        return sorted(list(self.df.keys()))

    def batch_to_gpu(self, batch):
        if torch.cuda.device_count() > 0:
            new_batch = {}
            for c in batch:
                if type(batch[c]) is torch.Tensor:
                    new_batch[c] = batch[c].cuda()
                else:
                    new_batch[c] = batch[c]
            return new_batch
        return batch
