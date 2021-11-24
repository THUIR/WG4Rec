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
from data_readers.HistoryDR import HistoryDR
from data_readers.TextDR import TextDR
import gc
from utils import utils


class WG4RecDR(HistoryDR):
    @staticmethod
    def parse_data_args(parser):
        """
        data reader 的数据集相关的命令行参数
        :param parser:
        :return:
        """
        parser.add_argument('--dict_size', type=int, default=500000,
                            help='Max size of word dictionary.')
        parser.add_argument('--sent_max', type=int, default=10,
                            help="Max number of words in sentences.")
        parser.add_argument('--topic_head', type=int, default=0,
                            help="Topic words at the first of column.")
        parser.add_argument('--tag_head', type=int, default=0,
                            help="tag words at the first of column.")
        parser.add_argument('--entity_tail', type=int, default=1,
                            help="entity words at the last of column.")
        parser.add_argument('--keyword_head', type=int, default=1,
                            help="Keywords at the first of column.")
        parser.add_argument('--graph_wn', type=int, default=100,
                            help='Number of neighbours of word-word.')
        parser.add_argument('--graph_tn', type=int, default=2,
                            help='Number of neighbours of word-topic.')
        return HistoryDR.parse_data_args(parser)

    def __init__(self, path, dataset, dict_size, sent_max,
                 topic_head, tag_head, keyword_head, graph_wn, graph_tn,
                 entity_tail,
                 *args, **kwargs):
        self.topic_head = topic_head
        self.tag_head = tag_head
        self.keyword_head = keyword_head
        self.entity_tail = entity_tail
        self.dict_size = int(dict_size)
        self.sent_max = sent_max
        assert self.sent_max > 0
        self.path = os.path.join(path, dataset)
        self.dictionary_file = os.path.join(self.path, dataset + DICT_SUFFIX)
        self.doc_text_file = os.path.join(self.path, dataset + DOC_TEXT_SUFFIX)
        self.url_text_file = os.path.join(self.path, dataset + URL_TEXT_SUFFIX)
        self.possible_text_cs = ['topic', 'tag', 'keywords', 'keywords_content',
                                 'title_cut', 'title_cut_search', 'query_cut', 'query_cut_search',
                                 'entity']
        self.dictionary_df = None
        self.doc_dict, self.url_dict = None, None

        self.word_graph_file = os.path.join(self.path, dataset + WORD_GRAPH_SUFFIX)
        self.word_graph = None
        self.graph_wn, self.graph_tn = graph_wn, graph_tn
        HistoryDR.__init__(self, path=path, dataset=dataset, *args, **kwargs)

    def load_data(self):
        HistoryDR.load_data(self)
        self.load_text()
        self.load_graph()

    def data_df2dict(self, df, skip_columns=None):
        if skip_columns is None:
            skip_columns = []
        if NEG_IIDS in df:
            df[NEG_IIDS] = df[NEG_IIDS].fillna('')
        df = HistoryDR.data_df2dict(self, df, skip_columns=skip_columns + self.possible_text_cs + [NEG_IIDS])
        for c in self.possible_text_cs:
            if c in df:
                text, length = TextDR.text_column2array(
                    df[c], dict_size=self.dict_size, text_max=-1, pad_text=False)
                df[c] = text
        if NEG_IIDS in df:
            df[NEG_IIDS], _ = TextDR.text_column2array(df[NEG_IIDS])
        return df

    def add_text_column(self, text_dict):
        text_arrays = []
        if self.tag_head == 1 and 'topic' in text_dict:
            text_arrays.append(text_dict['topic'])
        if self.tag_head == 1 and 'tag' in text_dict:
            text_arrays.append(text_dict['tag'])
        if self.keyword_head == 1 and 'keywords' in text_dict:
            text_arrays.append(text_dict['keywords'])
        text_arrays.append(text_dict['title_cut'])
        if self.entity_tail == 1 and 'entity' in text_dict:
            text_arrays.append(text_dict['entity'])
        result_array = []
        for i in range(len(text_arrays[-1])):
            text_array = np.concatenate([a[i] for a in text_arrays])
            result_array.append(text_array[:self.sent_max])
        length = np.array([len(a) if len(a) < self.sent_max else self.sent_max for a in result_array])
        text = utils.pad2same_length(result_array, max_len=self.sent_max, v=0, dtype=np.int64)
        text_dict[C_SENT] = text
        text_dict[C_SENT_LENGTH] = length
        return

    def load_text(self):
        if os.path.exists(self.dictionary_file):
            logging.info("load dictionary csv...")
            self.dictionary_df = pd.read_csv(self.dictionary_file, sep=self.sep, na_filter=False).fillna('')
            if self.dict_size > 0:
                self.dictionary_df = self.dictionary_df[self.dictionary_df[C_WORD_ID] < self.dict_size]
        if os.path.exists(self.doc_text_file):
            logging.info("load doc text...")
            doc_text_df = pd.read_csv(self.doc_text_file, sep='\t', na_filter=False).fillna('')
            self.doc_dict = self.data_df2dict(doc_text_df)
            text_columns = [c for c in self.doc_dict if c != IID]
            for c in text_columns:
                self.doc_dict[c] = utils.pad_id_array(self.doc_dict[IID], self.doc_dict[c],
                                                      sup=np.array([], dtype=np.int64))
            self.doc_dict[IID] = utils.pad_id_array(self.doc_dict[IID], self.doc_dict[IID], sup=None)
            self.add_text_column(self.doc_dict)
            self.column_max[IID] = self.doc_dict[IID].max()
        if os.path.exists(self.url_text_file):
            logging.info("load url text...")
            url_text_df = pd.read_csv(self.url_text_file, sep='\t', na_filter=False).fillna('')
            self.url_dict = self.data_df2dict(url_text_df)
            text_columns = [c for c in self.url_dict if c != URL_ID]
            for c in text_columns:
                self.url_dict[c] = utils.pad_id_array(self.url_dict[URL_ID], self.url_dict[c],
                                                      sup=np.array([], dtype=np.int64))
            self.url_dict[URL_ID] = utils.pad_id_array(self.url_dict[URL_ID], self.url_dict[URL_ID], sup=None)
            self.add_text_column(self.url_dict)
            self.column_max[URL_ID] = self.url_dict[URL_ID].max()

    def load_graph(self):
        if self.load_file and os.path.exists(self.word_graph_file):
            logging.info("load word graph...")
            self.word_graph = pickle.load(open(self.word_graph_file, 'rb'))
            for column in self.word_graph:
                self.word_graph[column] = np.array(self.word_graph[column], dtype=np.int64)
            self.graph_top_nb()
            logging.info("word graph: {}".format([c for c in self.word_graph.keys() if not c.endswith('_length')]))

    def graph_top_nb(self):
        topic_cn = [(c, self.graph_tn) for c in sorted(self.word_graph.keys())
                    if not c.endswith('_length') and 'topic' in c]
        word_cn = [(c, self.graph_wn) for c in sorted(self.word_graph.keys())
                   if not c.endswith('_length') and 'topic' not in c]
        for c, nb_n in topic_cn + word_cn:
            graph = self.word_graph[c][:self.dict_size]
            c_graph, c_length = [], []
            for w_graph in tqdm(graph, ncols=100, mininterval=1,
                                leave=False, desc='graph {} {}'.format(c, nb_n)):
                w_graph = [w for w in w_graph if w < self.dict_size][:nb_n]
                c_length.append(len(w_graph))
                c_graph.append(utils.pad_array(w_graph, nb_n, v=0))
            self.word_graph[c] = np.array(c_graph)
            self.word_graph[c + '_length'] = np.array(c_length)
        return

    def load_info(self):
        HistoryDR.load_info(self)
        self.data_columns = [c for c in self.data_columns if c not in self.possible_text_cs]
        self.dictionary_size = self.dictionary_df[C_WORD_ID].max() + 1

    def drop_no_history(self, train=True, validation=True, test=True):
        logging.info('Drop samples with no history...')
        if train and self.train_df is not None:
            index = np.where((self.train_df[C_HISTORY][:, 0] < self.train_df[C_HISTORY][:, 1])
                             | (self.train_df[C_HISTORY_NEG][:, 0] < self.train_df[C_HISTORY_NEG][:, 1]))[0]
            for k in self.train_df:
                self.train_df[k] = self.train_df[k][index]
            logging.info("size of train: %d" % len(index))
        if validation and self.validation_df is not None:
            index = np.where((self.validation_df[C_HISTORY][:, 0] < self.validation_df[C_HISTORY][:, 1])
                             | (self.validation_df[C_HISTORY_NEG][:, 0] < self.validation_df[C_HISTORY_NEG][:, 1]))[0]
            for k in self.validation_df:
                self.validation_df[k] = self.validation_df[k][index]
            logging.info("size of validation: %d" % len(index))
        if test and self.test_df is not None:
            if self.label in self.test_df:
                index = np.where((self.test_df[C_HISTORY][:, 0] < self.test_df[C_HISTORY][:, 1])
                                 | (self.test_df[C_HISTORY_NEG][:, 0] < self.test_df[C_HISTORY_NEG][:, 1]))[0]
                for k in self.test_df:
                    self.test_df[k] = self.test_df[k][index]
                logging.info("size of test: %d" % len(index))
