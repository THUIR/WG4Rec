# coding=utf-8

from collections import defaultdict, Counter

import numpy as np
from tqdm import tqdm
import pandas as pd
import logging

from data_processors.DataProcessor import DataProcessor
from utils.global_p import *


class RecDP(DataProcessor):

    @staticmethod
    def parse_dp_args(parser):
        """
        数据处理生成batch的命令行参数
        :param parser:
        :return:
        """
        parser.add_argument('--test_sample_n', type=int, default=100,
                            help='Negative sample num for each instance in test/validation set when ranking.')
        parser.add_argument('--train_sample_n', type=int, default=1,
                            help='Negative sample num for each instance in train set when ranking.')
        parser.add_argument('--sample_un_p', type=float, default=1.0,
                            help='Sample from neg/pos with 1-p or unknown+neg/pos with p.')
        parser.add_argument('--sample_pop', type=int, default=100,
                            help='Whether sample according to popularity')
        parser.add_argument('--sample_given_p', type=float, default=0.3,
                            help='Sample from given neg/pos with 1-p or others with p.')
        return DataProcessor.parse_dp_args(parser)

    def __init__(self, rank, test_sample_n, train_sample_n, sample_un_p, sample_pop, sample_given_p, *args, **kwargs):
        self.rank = rank
        self.test_sample_n = test_sample_n
        self.train_sample_n = train_sample_n
        self.sample_un_p = sample_un_p
        self.sample_pop = sample_pop
        self.sample_given_p = sample_given_p
        self.neg_dict = None
        DataProcessor.__init__(self, *args, **kwargs)

    def prepare_epoch(self):
        self.neg_dict = self.sample_negative()

    def sample_negative(self):
        if self.rank != 1:
            return None

        def multidimensional_shifting(num_samples, sample_size, elements, probabilities):
            # replicate probabilities as many times as `num_samples`
            replicated_probabilities = np.tile(probabilities, (num_samples, 1))
            # get random shifting numbers & scale them correctly
            random_shifts = np.random.random(replicated_probabilities.shape)
            random_shifts /= random_shifts.sum(axis=1)[:, np.newaxis]
            # shift by numbers & find largest (by finding the smallest of the negative)
            shifted_probabilities = random_shifts - replicated_probabilities
            return np.argpartition(shifted_probabilities, sample_size, axis=1)[:, :sample_size]

        train_history_pos = defaultdict(set)
        for uid in self.data_reader.train_user_pos.keys():
            train_history_pos[uid] = set(self.data_reader.train_user_pos[uid])
        validation_history_pos = defaultdict(set)
        for uid in self.data_reader.validation_user_pos.keys():
            validation_history_pos[uid] = set(self.data_reader.validation_user_pos[uid])
        test_history_pos = defaultdict(set)
        for uid in self.data_reader.test_user_pos.keys():
            test_history_pos[uid] = set(self.data_reader.test_user_pos[uid])
        train_history_neg = defaultdict(set)
        for uid in self.data_reader.train_user_neg.keys():
            train_history_neg[uid] = set(self.data_reader.train_user_neg[uid])
        validation_history_neg = defaultdict(set)
        for uid in self.data_reader.validation_user_neg.keys():
            validation_history_neg[uid] = set(self.data_reader.validation_user_neg[uid])
        test_history_neg = defaultdict(set)
        for uid in self.data_reader.test_user_neg.keys():
            test_history_neg[uid] = set(self.data_reader.test_user_neg[uid])

        sample_p = None
        if self.sample_pop > 0:
            item_popularity = Counter(self.data_reader.train_df[IID])
            # print(item_popularity.most_common(100))
            sample_p = np.array([item_popularity[i] for i in range(1, self.data_reader.item_num)]) + self.sample_pop
            sample_p = sample_p / sample_p.sum()
            # print(sample_p, sample_p.min(), sample_p.max())
        all_iids = np.array(range(1, self.data_reader.item_num), dtype=np.int)
        sample_buffer, sample_buf_p = np.random.choice(all_iids, size=10000000, p=sample_p), 0

        neg_dict = []
        sample_n = self.train_sample_n if self.train else self.test_sample_n
        for df_i in tqdm(range(len(self)),
                         leave=False, ncols=100, mininterval=1, desc=str(self.procedure) + '-Sample Negative'):
            uid, label = self.df[UID][df_i], self.df[LABEL][df_i]
            all_given_iids = []
            if NEG_IIDS in self.df:
                tmp_set = set()
                for i in self.df[NEG_IIDS][df_i]:
                    if i not in tmp_set:
                        all_given_iids.append(i)
                        tmp_set.add(i)
            assert len(all_given_iids) == len(set(all_given_iids))

            sample_given_n = self.sample_given_p * sample_n
            if sample_given_n >= len(all_given_iids):
                given_iids = [i for i in all_given_iids]
            elif sample_given_n >= 1:
                # given_ps = np.array([sample_p[i - 1] for i in all_given_iids])
                # given_ps = given_ps / given_ps.sum()
                given_iids = list(np.random.choice(all_given_iids, size=int(sample_given_n), replace=False))
            elif np.random.rand() < self.sample_given_p:
                given_iids = list(np.random.choice(all_given_iids, size=1, replace=False))
            else:
                given_iids = []

            if label > 0:
                # 避免采中已知的正例
                train_history = train_history_pos
                validation_history, test_history = validation_history_pos, test_history_pos
            else:
                assert self.train
                # 避免采中已知的负例
                train_history = train_history_neg
                validation_history, test_history = validation_history_neg, test_history_neg
            if self.train:
                # 训练集采样避免采训练集中已知的正例或负例
                inter_iids = train_history[uid]
                if label > 0:
                    known_iids = train_history_neg[uid]
                else:
                    known_iids = train_history_pos[uid]
            else:
                # 测试集采样避免所有已知的正例或负例
                inter_iids = train_history[uid] | validation_history[uid] | test_history[uid]
                if label > 0:
                    known_iids = validation_history_neg[uid] | test_history_neg[uid]
                else:
                    known_iids = validation_history_pos[uid] | test_history_pos[uid]

            # 检查所剩可以采样的数目
            item_num = self.data_reader.item_num
            remain_iids_num = item_num - len(inter_iids)
            # 所有可采数目不够则报错
            assert remain_iids_num >= sample_n

            # 如果数目不多则列出所有可采样的item采用np.choice
            sampled = set(given_iids)
            assert len(sampled) == len(given_iids)
            remain_iids, remain_ps = None, None
            if 1.0 * remain_iids_num / item_num < 0.2:
                remain_iids = [i for i in range(1, item_num) if i not in inter_iids and i not in sampled]
                if self.sample_pop == 1:
                    remain_ps = np.array([sample_p[i - 1] for i in remain_iids])
                    remain_ps = remain_ps / remain_ps.sum()
            if remain_iids is None:
                unknown_iid_list = given_iids
                while len(sampled) != sample_n:
                    if len(sample_buffer) <= sample_buf_p:
                        sample_buffer = np.random.choice(all_iids, size=10000000, p=sample_p)
                        sample_buf_p = 0
                    iid = sample_buffer[sample_buf_p]
                    sample_buf_p += 1
                    if iid in inter_iids or iid in sampled:
                        continue
                    unknown_iid_list.append(iid)
                    sampled.add(iid)
            else:
                unknown_iid_list = list(
                    np.random.choice(remain_iids, sample_n - len(sampled), p=remain_ps, replace=False))
                unknown_iid_list = given_iids + unknown_iid_list
            assert len(unknown_iid_list) == sample_n

            # 如果训练时候，有可能从已知的负例或正例中采样
            if self.train and self.sample_un_p < 1:
                known_iid_list = list(np.random.choice(
                    list(known_iids), min(sample_n, len(known_iids)), replace=False)) \
                    if len(known_iids) != 0 else []
                known_iid_list = known_iid_list + unknown_iid_list
                tmp_iid_list = []
                sampled = set()
                for i in range(sample_n):
                    p = np.random.rand()
                    if p < self.sample_un_p or len(known_iid_list) == 0:
                        iid = unknown_iid_list.pop(0)
                        while iid in sampled:
                            iid = unknown_iid_list.pop(0)
                    else:
                        iid = known_iid_list.pop(0)
                        while iid in sampled:
                            iid = known_iid_list.pop(0)
                    tmp_iid_list.append(iid)
                    sampled.add(iid)
                neg_dict.append(np.array(tmp_iid_list, dtype=np.int64))
            else:
                neg_dict.append(np.array(unknown_iid_list, dtype=np.int64))
        return np.array(neg_dict)

    def __getitem__(self, index):
        if self.buffer_dp_b and len(self.buffer_b) > 0:
            return index
        if self.rank == 0 and self.buffer_dp_i and index in self.buffer_i:
            return self.buffer_i[index]
        indexes = [index]
        if self.rank == 1:
            sample_n = self.train_sample_n if self.train else self.test_sample_n
            indexes += [i * len(self) + index for i in range(1, sample_n + 1)]
        results = []
        for idx in indexes:
            if self.buffer_dp_i and idx in self.buffer_i:
                results.append(self.buffer_i[idx])
            else:
                results.append(self.get_item(idx))
        return results

    def get_item(self, index):
        real_index = index
        if index >= len(self):
            real_index = index % len(self)
            neg_index = int(index / len(self)) - 1
            iid = self.neg_dict[real_index, neg_index]
        else:
            iid = self.df[IID][real_index]
        uid, label = self.df[UID][real_index], self.df[LABEL][real_index]
        x, feature_columns = [], []
        if self.model_name.include_id:
            x.extend([uid, iid])
            feature_columns.extend([UID, IID])
        if self.model_name.include_user_features and self.data_reader.user_df is not None:
            user_features = self.data_reader.user_features
            uf_row = self.data_reader.user_df
            x.extend([uf_row[f][uid] for f in user_features])
            feature_columns.extend(user_features)
        if self.model_name.include_item_features and self.data_reader.item_df is not None:
            item_features = self.data_reader.item_features
            if_row = self.data_reader.item_df
            x.extend([if_row[f][iid] for f in item_features])
            feature_columns.extend(item_features)
        if self.model_name.include_context_features:
            context_features = self.data_reader.context_features
            x.extend([self.df[f][real_index] for f in context_features])
            feature_columns.extend(context_features)
        base = 0
        for i, feature in enumerate(feature_columns):
            x[i] += base
            base += int(self.data_reader.column_max[feature] + 1)
        feed_dict = {UID: uid, IID: iid, Y: label, SAMPLE_ID: index}
        if len(feature_columns) > 0:
            feed_dict[X] = x
        return feed_dict

    def get_batch(self, batch, skip_keys=None, info_keys=None):
        real_batch_size = len(batch)
        rank_n = len(batch[0])
        batches = []
        for rank_i in range(rank_n):
            batches.extend([b[rank_i] for b in batch])
        feed_dict = DataProcessor.get_batch(self, batch=batches, skip_keys=skip_keys, info_keys=info_keys)
        feed_dict[REAL_BATCH_SIZE] = real_batch_size
        feed_dict[RANK] = self.rank
        feed_dict[Y] = feed_dict[Y][:real_batch_size]
        return feed_dict

    def save_neg_dict(self, csv_file):
        if self.neg_dict is None:
            return
        df = pd.DataFrame({UID: self.df[UID], IID: self.df[IID]})
        neg_iids = [','.join([str(iid) for iid in iids]) for iids in self.neg_dict]
        df[NEG_IIDS] = neg_iids
        df.to_csv(csv_file, sep='\t', index=False)
        logging.info('save neg iids to {}'.format(csv_file))
        return df

    def load_neg_dict(self, csv_file):
        logging.info('load neg iids from {}'.format(csv_file))
        df = pd.read_csv(csv_file, sep='\t')
        neg_iids = df[NEG_IIDS].astype(str).fillna('').tolist()
        neg_iids = np.array([[int(i) for i in l.split(',')] for l in neg_iids], np.int64)
        self.neg_dict = neg_iids
        return neg_iids
