# coding=utf-8
import copy
from utils import utils
import numpy as np
import logging
import pandas as pd
from tqdm import tqdm
import torch
from collections import defaultdict
from data_processors.NewsDP import SGNewsDP
from utils.global_p import *


class AdressaDP(SGNewsDP):

    @staticmethod
    def parse_dp_args(parser):
        """
        数据处理生成batch的命令行参数
        :param parser:
        :return:
        """
        parser.add_argument('--graph_l', type=int, default=1,
                            help='Layers of word graph.')
        parser.add_argument('--graph_sample', type=int, default=1,
                            help='Whether sample nodes or not')
        parser.add_argument('--word_wn', type=int, default=10,
                            help='Number of neighbours in relation word-word.')
        parser.add_argument('--word_tn', type=int, default=1,
                            help='Number of neighbours in relation word-topic.')
        parser.add_argument('--nb_type', type=int, default=0,
                            help='Whether sample neighbours according to different types')
        parser.add_argument('--nb_topic', type=int, default=0,
                            help='Whether sample topic neighbours as a different type')
        return SGNewsDP.parse_dp_args(parser)

    def __init__(self, graph_l, graph_sample, word_wn, word_tn, nb_type, nb_topic, *args, **kwargs):
        self.graph_sample = graph_sample
        self.graph_l = graph_l
        self.word_wn = word_wn
        self.word_tn = word_tn
        self.nb_type = nb_type
        self.nb_topic = nb_topic
        self.word_graph, self.graph_cn, self.graph_split = None, None, None
        SGNewsDP.__init__(self, *args, **kwargs)

    def prepare(self):
        SGNewsDP.prepare(self)
        self.prepare_word_graph()

    def prepare_word_graph(self):
        topic_cn = [(c, self.word_tn) for c in sorted(self.data_reader.word_graph.keys())
                    if not c.endswith('_length') and 'topic' in c]
        word_cn = [(c, self.word_wn) for c in sorted(self.data_reader.word_graph.keys())
                   if not c.endswith('_length') and 'topic' not in c]
        if self.nb_type == 1:
            word_graph = self.data_reader.word_graph
            graph_cn = topic_cn + word_cn
        else:
            topic_g, word_g = [], []
            for wid in tqdm(range(self.data_reader.dictionary_size),
                            leave=False, ncols=100, mininterval=1, desc='{}-Prepare Word Graph'.format(self.procedure)):
                topic_nb, word_nb = [], []
                for c, cn in topic_cn:
                    graph, length = self.data_reader.word_graph[c], self.data_reader.word_graph[c + '_length']
                    if wid >= len(graph):
                        topic_nb.append([])
                    elif self.train and self.graph_sample == 1:
                        topic_nb.append(graph[wid][:length[wid]])
                    else:
                        topic_nb.append(graph[wid][:min(length[wid], cn)])
                for c, cn in word_cn:
                    graph, length = self.data_reader.word_graph[c], self.data_reader.word_graph[c + '_length']
                    if wid >= len(graph):
                        word_nb.append([])
                    elif self.train and self.graph_sample == 1:
                        word_nb.append(graph[wid][:length[wid]])
                    else:
                        word_nb.append(graph[wid][:min(length[wid], cn)])
                if self.nb_topic == 1:
                    topic_g.append(np.concatenate(topic_nb, axis=0))
                    word_g.append(np.concatenate(word_nb, axis=0))
                else:
                    word_g.append(np.concatenate(topic_nb + word_nb, axis=0))
            word_l = [len(g) for g in word_g]
            max_wl = max(word_l)
            word_g = [utils.pad_array(g, max_wl, 0) for g in word_g]
            word_graph = {'word_word': np.array(word_g, dtype=np.int64),
                          'word_word_length': np.array(word_l, dtype=np.int64)}
            if self.nb_topic == 1:
                topic_l = [len(g) for g in topic_g]
                max_tl = max(topic_l)
                topic_g = [utils.pad_array(g, max_tl, 0) for g in topic_g]
                word_graph['word_topic'] = np.array(topic_g, dtype=np.int64)
                word_graph['word_topic_length'] = np.array(topic_l, dtype=np.int64)
                graph_cn = [('word_topic', self.word_tn * len(topic_cn)), ('word_word', self.word_wn * len(word_cn))]
            else:
                graph_cn = [('word_word', self.word_tn * len(topic_cn) + self.word_wn * len(word_cn))]
        graph_split = []
        for i in range(self.graph_l):
            split = [1] + [g[1] for g in graph_cn if g[1] > 0]
            graph_split.append(split)
        self.word_graph, self.graph_cn, self.graph_split = word_graph, graph_cn, graph_split
        return

    def get_item(self, index):
        result = SGNewsDP.get_item(self, index)
        result[C_WORD_GRAPH] = np.array(
            [self.data_reader.doc_dict[C_SENT][iid] for iid in result[C_HISTORY]], dtype=np.int64)  # H * L
        return result

    def get_batch(self, batch, skip_keys=None, info_keys=None):
        if skip_keys is None:
            skip_keys = []
        if self.graph_l > 0:
            feed_dict = SGNewsDP.get_batch(self, batch, skip_keys=skip_keys + [C_WORD_GRAPH], info_keys=info_keys)
            rank_n = len(batch[0])
            batches = [b[rank_i] for rank_i in range(rank_n) for b in batch]
            word_graph = []
            for b in batches:
                his = b[C_WORD_GRAPH]  # H * L
                graph = [self.get_words_graph(s, layer=1) for s in his]
                word_graph.append(graph)
            feed_dict[C_WORD_GRAPH] = utils.numpy_to_torch(
                np.array(word_graph, dtype=np.int64), gpu=False)  # B * H * L * G?
        else:
            feed_dict = SGNewsDP.get_batch(self, batch, skip_keys=skip_keys, info_keys=info_keys)
        feed_dict[C_GRAPH_SPLIT] = self.graph_split
        return feed_dict

    def get_words_graph(self, words, layer):
        result = [words.reshape(-1, 1)]  # L * 1
        for g_name, sample_n in self.graph_cn:
            graph = self.word_graph[g_name][words]  # L * NB
            if not self.train or self.graph_sample != 1:
                word_r = graph[:, :sample_n]  # L * S
            else:
                length = self.word_graph[g_name + '_length'][words]  # L
                high = np.stack([length] * sample_n, axis=1)  # L * S
                index = np.floor(np.random.uniform(high=high)).astype(np.int64)  # L * S
                r_index = np.stack([np.arange(len(words))] * sample_n, axis=1)  # L * S
                word_r = graph[r_index, index]  # L * S
            result.append(word_r)
        result = np.concatenate(result, axis=1)  # L * G
        if layer < self.graph_l:
            result = [self.get_words_graph(w, layer + 1) for w in result]
        return result
