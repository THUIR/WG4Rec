# coding=utf-8
import sys

sys.path.insert(0, '../')
sys.path.insert(0, './')

import os
import math
import pickle
import socket
from collections import defaultdict, Counter
from shutil import copyfile
import itertools
from datetime import datetime
import time
import string

import jieba
import jsonlines
import numpy as np
import pandas as pd
from tqdm import tqdm
from gensim.models import KeyedVectors
from pandarallel import pandarallel

from utils.global_p import *
from utils.dataset import leave_out_by_time_csv

# pandarallel.initialize(nb_workers=5, progress_bar=True)
# pandarallel.initialize(nb_workers=5, progress_bar=False)
jieba.enable_paddle()
jieba.enable_parallel(5)

np.random.seed(DEFAULT_SEED)
print(socket.gethostname())

RAW_DATA = '/work/shisy13/Dataset/Sogou/SogouNews/20200403/'
OUT_DIR = os.path.join(DATA_DIR, 'SogouNews0403')
WORD2VEC_DIR = '/work/shisy13/Library/word2vec/'
WORD2VEC_FILE = os.path.join(WORD2VEC_DIR, 'sgns.sogounews.bigram-char')
STOPWORDS_FILE = os.path.join(OUT_DIR, 'stopwords.txt')

RAW_REC_FILES = [os.path.join(RAW_DATA, 'online_feed_data/', 'onlineone_data_{:02d}'.format(i)) for i in range(21)]
RAW_REC_FILE = os.path.join(RAW_DATA, 'online_feed_data/', 'onlineone_data.all')
RAW_DOC_FILE = os.path.join(RAW_DATA, 'all_docid_info')
RAW_SEARCH_FILE = os.path.join(RAW_DATA, 'search_data/', 'user_search_action')
RAW_URL_FILES = [os.path.join(RAW_DATA, 'search_data/', 'topic_url_detail_{:02d}'.format(i)) for i in range(8)]
RAW_URL_FILE = os.path.join(RAW_DATA, 'search_data/', 'topic_url_detail.all')
RAW_DOC_TAG_FILE = os.path.join(RAW_DATA, 'feed_class_result')
RAW_URL_TAG_FILE = os.path.join(RAW_DATA, 'search_class_result')

REC_FILE = os.path.join(RAW_DATA, 'online_feed_data/', 'onlineone_data.utf8')
DOC_FILE = os.path.join(RAW_DATA, 'all_docid_info.utf8')
SEARCH_FILE = os.path.join(RAW_DATA, 'search_data/', 'user_search_action.utf8')
URL_FILE = os.path.join(RAW_DATA, 'search_data/', 'topic_url_detail.utf8')

ITEM_ID_DICT = os.path.join(OUT_DIR, 'item_id_dict.csv')
USER_ID_DICT = os.path.join(OUT_DIR, 'user_id_dict.csv')
URL_ID_DICT = os.path.join(OUT_DIR, 'url_id_dict.csv')
QUERY_ID_DICT = os.path.join(OUT_DIR, 'query_id_dict.csv')
REC_INTERACTION_ORIGIN_FILE = os.path.join(OUT_DIR, 'rec_interactions.origin.csv')
REC_INTERACTION_FILE = os.path.join(OUT_DIR, 'rec_interactions.csv')
REC_INTER_GROUP_FILE = os.path.join(OUT_DIR, 'rec_inter_group.csv')
SEARCH_INTERACTION_FILE = os.path.join(OUT_DIR, 'search_interactions.csv')
SEARCH_INTER_GROUP_FILE = os.path.join(OUT_DIR, 'search_inter_group.csv')
# QUERY_INTERACTION_FILE = os.path.join(OUT_DIR, 'query_interactions.csv')
QUERY_INTER_GROUP_FILE = os.path.join(OUT_DIR, 'query_inter_group.csv')
USER_INTER_CNT = os.path.join(OUT_DIR, 'user_inter_cnt.csv')
DOC_TEXT_ORIGIN_FILE = os.path.join(OUT_DIR, 'doc_text.origin.csv')
URL_TEXT_ORIGIN_FILE = os.path.join(OUT_DIR, 'url_text.origin.csv')
QUERY_TEXT_FILE = os.path.join(OUT_DIR, 'query_text.csv')
DOC_TOPIC_TAG0624 = os.path.join(OUT_DIR, 'doc_topic_tag0624.csv')
URL_TOPIC_TAG0624 = os.path.join(OUT_DIR, 'url_topic_tag0624.csv')
DOC_TEXT_FILE = os.path.join(OUT_DIR, 'doc_text.csv')
URL_TEXT_FILE = os.path.join(OUT_DIR, 'url_text.csv')

TOPIC_TAG_DATASET_DIR = os.path.join(OUT_DIR, 'topic_tag_dataset')
DOC_TOPIC_TAG_TAXONOMY = os.path.join(TOPIC_TAG_DATASET_DIR, 'doc_topic_tag_taxonomy.csv')

REC_COLUMNS = ['click', 'user_type', 'm_user_id', 'doc_id', 'topic',
               'time', 'read_duration', 'position', 'account', 'url',
               'channel', 'location', 'rec_reason', 'ip', 'recall_word',
               'video', 'source']
DOC_COLUMNS = ['doc_id', 'title', 'topic', 'account', 'account_weight',
               'keywords', 'keywords_content', 'tag', 'video', 'porn',
               'quality', 'text_feature_list', 'page_time']
SEARCH_COLUMNS = ['user_type', 'm_user_id', 'time', 'query', 'url']
URL_COLUMNS = ['url', 'abstract', 'title', 'content', 'topic', 'tag', 'keyword']

TOPIC_T, TAG_T, KEYWORD_T, WORD_T, NAN_T = 3, 2, 1, 0, -1
DEFAULT_DICT_SIZE = 500000


def transfer2utf8(in_file, out_file, encoding='gbk', errors='strict', sep=None, column_n=None):
    print(in_file)
    in_f = open(in_file, 'r', encoding=encoding, errors=errors)
    out_f = open(out_file, 'w')
    pre_line = None
    for line_no, line in enumerate(in_f):
        if pre_line is not None:
            line = pre_line + line
        if sep is not None:
            line_split = line.split(sep)
            # assert len(line_split) <= column_n
            if len(line_split) < column_n:
                print(line_no, len(line_split))
                pre_line = line.strip()
                continue
            elif len(line_split) > column_n:
                print(line_no, len(line_split))
                print(line_split)
        out_f.write(line.strip() + os.linesep)
        pre_line = None
    assert pre_line is None
    in_f.close()
    out_f.close()
    return


def id2set(in_file, id_set, sep, index):
    in_f = open(in_file, 'r')
    for line_no, line in enumerate(in_f):
        line = line.strip().split(sep)
        new_id = line[index]
        id_set.add(new_id.strip())
    in_f.close()
    return id_set


def id_set2df(id_set, raw_name, id_name):
    df = pd.DataFrame({raw_name: list(id_set)})
    df = df.sort_values(raw_name).reset_index(drop=True)
    df[id_name] = df.index + 1
    return df


def build_item_id_dict():
    raw_name, id_name = 'doc_id', IID
    doc_ids = {''}
    doc_ids = id2set(DOC_FILE, doc_ids, sep='\t', index=0)
    df = id_set2df(doc_ids, raw_name, id_name)
    df.to_csv(ITEM_ID_DICT, sep='\t', index=False)
    return


def get_columns_df(in_file, sep, column_index, skip_error=False):
    print('get_columns_df:', column_index)
    data = {}
    for k in column_index:
        data[k] = []
    in_f = open(in_file, 'r')
    for line_no, line in enumerate(in_f):
        line = line.strip().split(sep)
        tmp_d = []
        try:
            for k in column_index:
                tmp_d.append((k, line[column_index[k]].strip()))
        except Exception as e:
            if skip_error:
                continue
            raise e
        for k, d in tmp_d:
            data[k].append(d)
    in_f.close()
    df = pd.DataFrame(data)
    return df


def df_transfer_id(df, column, dict_csv, key_c, value_c, nan_v=-1):
    id_df = pd.read_csv(dict_csv, sep='\t')
    id_dict = dict(zip(id_df[key_c], id_df[value_c]))
    df[column] = df[column].apply(lambda x: id_dict[x.strip()] if x in id_dict else nan_v)
    return df


def build_user_id_dict():
    raw_name, id_name = 'user_id', UID
    mid_set = {''}
    mid_set = id2set(REC_FILE, mid_set, sep='\a', index=2)
    mid_set = id2set(SEARCH_FILE, mid_set, sep='\t', index=1)
    df = id_set2df(mid_set, raw_name, id_name)
    print(df)
    df.to_csv(USER_ID_DICT, sep='\t', index=False)
    return


def build_url_id_dict():
    raw_name, id_name = 'url', 'url_id'
    url_id_set = {''}
    url_id_set = id2set(URL_FILE, url_id_set, sep='\t', index=0)
    df = id_set2df(url_id_set, raw_name, id_name)
    print(df)
    df.to_csv(URL_ID_DICT, sep='\t', index=False)
    return


def build_query_id_dict():
    raw_name, id_name = 'query', 'query_id'
    query_id_set = {''}
    query_id_set = id2set(SEARCH_FILE, query_id_set, sep='\t', index=-2)
    df = id_set2df(query_id_set, raw_name, id_name)
    print(df)
    df.to_csv(QUERY_ID_DICT, sep='\t', index=False)
    return


def build_rec_interactions_origin():
    print('build_rec_interactions_origin')
    uid_df = pd.read_csv(USER_ID_DICT, sep='\t')
    uid_dict = dict(zip(uid_df['user_id'], uid_df[UID]))
    iid_df = pd.read_csv(ITEM_ID_DICT, sep='\t')
    iid_dict = dict(zip(iid_df['doc_id'], iid_df[IID]))

    data = {UID: [], IID: [], LABEL: [], TIME: [], 'position': []}
    invalid_uid, invalid_iid, invalid_pos = 0, 0, 0
    in_f = open(REC_FILE, 'r')
    for line_no, line in enumerate(in_f):
        line = line.strip().split('\a')
        uid, iid, label, time, position = \
            line[2].strip(), line[3].strip(), int(line[0].strip()), int(line[5].strip()), line[7].strip()
        if uid not in uid_dict:
            invalid_uid += 1
            continue
        if iid not in iid_dict:
            invalid_iid += 1
            continue
        try:
            position = int(position)
        except:
            position = -1
            invalid_pos += 1
        data[UID].append(uid_dict[uid])
        data[IID].append(iid_dict[iid])
        data[LABEL].append(label)
        data[TIME].append(time)
        data['position'].append(position)
    in_f.close()

    df = pd.DataFrame(data).astype(int)
    df = df.drop_duplicates()
    # df = pd.read_csv(REC_INTERACTION_ORIGIN_FILE, sep='\t')
    df = df.sort_values(by=[UID, TIME, 'position', IID]).reset_index(drop=True)
    print(df)
    print('invalid user id: {}'.format(invalid_uid))
    print('invalid item id: {}'.format(invalid_iid))
    print('invalid position id: {}'.format(invalid_pos))
    df[[UID, IID, LABEL, TIME, 'position']].to_csv(REC_INTERACTION_ORIGIN_FILE, sep='\t', index=False)
    return


def build_rec_interactions():
    df = pd.read_csv(REC_INTERACTION_ORIGIN_FILE, sep='\t')
    data = {UID: [], IID: [], LABEL: [], TIME: [], 'position': []}
    uids, iids, labels, timestps, positions = \
        df[UID].values, df[IID].values, df[LABEL].values, df[TIME].values, df['position'].values
    pre_uid, pre_time, uid_list = -1, -1, []
    for idx in tqdm(range(len(uids) + 1), desc="build_rec_interactions", leave=False, ncols=100, mininterval=1):
        uid, iid, label, timestp, position = None, None, None, None, -1
        if idx < len(uids):
            uid, iid, label, timestp, position = uids[idx], iids[idx], labels[idx], timestps[idx], positions[idx]
        if uid != pre_uid or uid is None:
            for time_list in uid_list:  # 对每一个时间的展示列表
                if time_list[1] <= 0:  # 如果没有点击的新闻，跳过该列表
                    continue
                out_time = time_list[2]  # 列表展示时间
                for out_pos in range(len(time_list[0])):  # 对列表中每个位置
                    for out_iid, out_label in time_list[0][out_pos]:  # 有的位置有多个新闻
                        data[UID].append(pre_uid)
                        data[IID].append(out_iid)
                        data[LABEL].append(out_label)
                        data[TIME].append(out_time)
                        data['position'].append(out_pos)
            pre_uid, pre_time, uid_list = -1, -1, []
        if position < 0:  # 位置异常，则跳过该条
            continue
        if label == 1:  # 是新闻点击记录
            found = False
            for i in range(1, len(uid_list) + 1):  # 对在该点击之前的每个用户展示列表
                if position < len(uid_list[-i][0]):  # 如果该列表长度达到了点击位置
                    records = uid_list[-i][0][position]  # 该位置的所有新闻记录
                    for record in records:  # 对该位置的每一个展示
                        if record[0] == iid:  # 如果是对应的点击新闻
                            record[1] = label  # 则将点击标签修正为1
                            found = True
                if found:  # 如果该列表中找到了所点击的新闻
                    uid_list[-i][1] += 1  # 列表点击数+1
                    break  # 跳过其它列表
            if found:  # 如果找到了所点击的新闻
                continue  # 则不再将点击新闻作为新的列表
        if timestp != pre_time:  # 如果时间戳是新的
            pre_time = timestp  # 更新时间戳
            uid_list.append([[], label, timestp])  # 添加新的列表
        time_list = uid_list[-1]  # 当前时间的展示列表
        while len(time_list[0]) <= position:  # 填充直到长度有当前位置
            time_list[0].append([])
        time_list[0][position].append([iid, label])  # 当前位置加入新的新闻记录
    df = pd.DataFrame(data).astype(int)
    print(df)
    df.to_csv(REC_INTERACTION_FILE, sep='\t', index=False)
    return


def build_search_interactions():
    print('build_search_interactions')
    df = get_columns_df(in_file=SEARCH_FILE, sep='\t',
                        column_index={UID: 1, 'query_id': -2, 'url_id': -1, TIME: 2})

    df = df_transfer_id(df, UID, dict_csv=USER_ID_DICT, key_c='user_id', value_c=UID)
    invalid_uid = df[df[UID] < 0].index
    df = df.drop(invalid_uid)

    df = df_transfer_id(df, 'query_id', dict_csv=QUERY_ID_DICT, key_c='query', value_c='query_id')
    invalid_query_id = df[df['query_id'] < 0].index
    df = df.drop(invalid_query_id)

    df = df_transfer_id(df, 'url_id', dict_csv=URL_ID_DICT, key_c='url', value_c='url_id')
    invalid_url_id = df[df['url_id'] < 0].index
    df = df.drop(invalid_url_id)

    df = df.astype(int)
    df = df.drop_duplicates()
    df = df.sort_values(by=[TIME, UID, 'query_id', 'url_id']).reset_index(drop=True)
    print(df)
    print('invalid user id: {}'.format(len(invalid_uid)))
    print('invalid query id: {}'.format(len(invalid_query_id)))
    print('invalid url id: {}'.format(len(invalid_url_id)))
    df[[UID, 'query_id', 'url_id', TIME]].to_csv(SEARCH_INTERACTION_FILE, sep='\t', index=False)
    return


# def build_query_interactions():
#     print('build_query_interactions')
#     df = get_columns_df(in_file=SEARCH_FILE, sep='\t', column_index={UID: 1, 'query_id': -2, TIME: 2})
#
#     df = df_transfer_id(df, UID, dict_csv=USER_ID_DICT, key_c='user_id', value_c=UID)
#     invalid_uid = df[df[UID] < 0].index
#     df = df.drop(invalid_uid)
#
#     df = df_transfer_id(df, 'query_id', dict_csv=QUERY_ID_DICT, key_c='query', value_c='query_id')
#     invalid_query_id = df[df['query_id'] < 0].index
#     df = df.drop(invalid_query_id)
#
#     df = df.astype(int)
#     df = df.drop_duplicates()
#     df = df.sort_values(by=[TIME, UID, 'query_id']).reset_index(drop=True)
#     print(df)
#     print('invalid user id: {}'.format(len(invalid_uid)))
#     print('invalid query id: {}'.format(len(invalid_query_id)))
#     df[[UID, 'query_id', TIME]].to_csv(QUERY_INTERACTION_FILE, sep='\t', index=False)
#     return


def group_user_interactions(in_csv, out_csv, group_c, list_c, pos_neg=None):
    in_df = pd.read_csv(in_csv, sep='\t')
    if pos_neg is not None:
        if pos_neg > 0:
            in_df = in_df[in_df[LABEL] > 0]
        else:
            in_df = in_df[in_df[LABEL] <= 0]
    group_ids, inters = [], []
    for name, group in in_df.groupby(group_c):
        group_ids.append(name)
        inters.append(','.join(group[list_c].astype(str).tolist()))
    group_inters = pd.DataFrame()
    group_inters[group_c] = group_ids
    group_inters[list_c] = inters
    group_inters = group_inters.sort_values(by=group_c).reset_index(drop=True)
    print(group_inters)
    group_inters.to_csv(out_csv, sep='\t', index=False)
    return


def count_user_interactions():
    inters = [(REC_INTER_GROUP_FILE, IID, 'click_cnt'),
              (SEARCH_INTER_GROUP_FILE, 'url_id', 'search_cnt'),
              (QUERY_INTER_GROUP_FILE, 'query_id', 'query_cnt')]
    dfs = []
    for inter_file, group_c, cnt_c in inters:
        inter_df = pd.read_csv(inter_file, sep='\t')
        inter_df[cnt_c] = inter_df[group_c].fillna('').apply(lambda x: len(x.split(',')))
        inter_df = inter_df.drop(columns=[group_c])
        dfs.append(inter_df)

    inter_cnt = dfs[0]
    for df in dfs[1:]:
        inter_cnt = pd.merge(inter_cnt, df, on=UID, how='outer')
    inter_cnt = inter_cnt.fillna(0).astype(int)
    inter_cnt['user_type'] = inter_cnt.apply(
        lambda r: 3 if r['search_cnt'] >= 10 and r['click_cnt'] < 3 else
        2 if r['search_cnt'] >= 10 and r['click_cnt'] >= 3 else 1,
        axis=1)
    inter_cnt = inter_cnt.sort_values(by=UID).reset_index(drop=True)
    print(inter_cnt)
    print(Counter(inter_cnt['user_type']))
    inter_cnt.to_csv(USER_INTER_CNT, sep='\t', index=False)
    return


def build_doc_text_df():
    df = get_columns_df(DOC_FILE, sep='\t', column_index={
        IID: 0, 'title': 1, 'topic': 2, 'keywords': 5, 'keywords_content': 6, 'tag': 7})
    df = df.fillna('')
    for c in df.columns:
        df[c] = df[c].apply(lambda x: x.strip())
    df = df_transfer_id(df, IID, dict_csv=ITEM_ID_DICT, key_c='doc_id', value_c=IID)

    def filter_w(words):
        words = [w.strip() for w in words]
        return [w for w in words if w != '']

    tqdm.pandas(desc="title cut", leave=False, ncols=100, mininterval=1)
    df['title_cut'] = df['title'].progress_apply(lambda x: ' '.join(filter_w(jieba.lcut(x))))
    tqdm.pandas(desc="title cut search", leave=False, ncols=100, mininterval=1)
    df['title_cut_search'] = df['title'].progress_apply(lambda x: ' '.join(filter_w(jieba.lcut_for_search(x))))
    tqdm.pandas(desc="keywords", leave=False, ncols=100, mininterval=1)
    df['keywords'] = df['keywords'].progress_apply(lambda x: ' '.join(filter_w(x.split('\a'))))
    tqdm.pandas(desc="keywords_content", leave=False, ncols=100, mininterval=1)
    df['keywords_content'] = df['keywords_content'].progress_apply(lambda x: ' '.join(filter_w(x.split('\a'))))
    tqdm.pandas(desc="topic", leave=False, ncols=100, mininterval=1)
    df['topic'] = df['topic'].progress_apply(lambda x: ' '.join(filter_w(x.split('\a'))))
    tqdm.pandas(desc="tag", leave=False, ncols=100, mininterval=1)
    df['tag'] = df['tag'].progress_apply(lambda x: ' '.join(filter_w(x.split('\a'))))
    print(df)
    df.info()
    df.to_csv(DOC_TEXT_ORIGIN_FILE, sep='\t', index=False)
    return


def build_url_text_df():
    df = get_columns_df(URL_FILE, sep='\t', column_index={
        'url_id': 0, 'title': 2, 'topic': 4, 'keywords': -1, 'tag': -2},
                        skip_error=True)
    df = df.fillna('')
    for c in df.columns:
        df[c] = df[c].apply(lambda x: x.strip())
    df = df_transfer_id(df, 'url_id', dict_csv=URL_ID_DICT, key_c='url', value_c='url_id')

    def filter_w(words):
        words = [w.strip() for w in words]
        return [w for w in words if w != '']

    tqdm.pandas(desc="title cut", leave=False, ncols=100, mininterval=1)
    df['title_cut'] = df['title'].progress_apply(lambda x: ' '.join(filter_w(jieba.lcut(x))))
    tqdm.pandas(desc="title cut search", leave=False, ncols=100, mininterval=1)
    df['title_cut_search'] = df['title'].progress_apply(lambda x: ' '.join(filter_w(jieba.lcut_for_search(x))))
    tqdm.pandas(desc="keywords", leave=False, ncols=100, mininterval=1)
    df['keywords'] = df['keywords'].progress_apply(lambda x: ' '.join(filter_w(x.split(','))))
    tqdm.pandas(desc="topic", leave=False, ncols=100, mininterval=1)
    df['topic'] = df['topic'].progress_apply(lambda x: ' '.join(filter_w(x.split(','))))
    tqdm.pandas(desc="tag", leave=False, ncols=100, mininterval=1)
    df['tag'] = df['tag'].progress_apply(lambda x: ' '.join(filter_w(x.split(','))))
    print(df)
    df.info()
    df.to_csv(URL_TEXT_ORIGIN_FILE, sep='\t', index=False)
    return


def build_query_text_df():
    df = pd.read_csv(QUERY_ID_DICT, sep='\t')
    df = df.fillna('')

    def filter_w(words):
        words = [w.strip() for w in words]
        return [w for w in words if w != '']

    tqdm.pandas(desc="query cut", leave=False, ncols=100, mininterval=1)
    df['query_cut'] = df['query'].progress_apply(lambda x: ' '.join(filter_w(jieba.lcut(x))))
    tqdm.pandas(desc="query cut search", leave=False, ncols=100, mininterval=1)
    df['query_cut_search'] = df['query'].progress_apply(lambda x: ' '.join(filter_w(jieba.lcut_for_search(x))))
    print(df)
    df.to_csv(QUERY_TEXT_FILE, sep='\t', index=False)
    return


def build_topic_tag_taxonomy(out_file):
    doc_df = pd.read_csv(DOC_TOPIC_TAG0624, sep='\t')[['topic', 'tag']]
    url_df = pd.read_csv(URL_TOPIC_TAG0624, sep='\t')[['topic', 'tag']]
    df = pd.concat([doc_df, url_df], ignore_index=True).reset_index(drop=True)
    df = df.dropna(how='any')
    df_group = df.groupby(['topic', 'tag'])
    pair_df = pd.DataFrame({'cnt': df_group.size()}).sort_values('cnt', ascending=False).reset_index()
    print(pair_df)
    pair_df.to_csv(out_file, sep='\t', index=False)
    return


def read_topic_tag_taxonomy(taxonomy_file):
    taxonomy_df = pd.read_csv(taxonomy_file, sep='\t')
    taxonomy_df = taxonomy_df[taxonomy_df['cnt'] >= 100]
    taxonomy = defaultdict(set)
    for topic, tag in zip(taxonomy_df['topic'].tolist(), taxonomy_df['tag'].tolist()):
        taxonomy[topic].add(tag)
    return taxonomy


def build_topic_tag_tvt(taxonomy_file, dataset_dir, dataset_name='sgnews'):
    def filter_words(words):
        result = []
        for word in words:
            flag = 0
            for uchar in word:
                if not '\u4e00' <= uchar <= '\u9fa5':
                    flag = 1
            if flag == 0:
                result.append(word)
        return result

    taxonomy = read_topic_tag_taxonomy(taxonomy_file)
    topics = sorted(taxonomy.keys())
    with open(os.path.join(dataset_dir, dataset_name + '.taxonomy'), 'w') as out_f:
        out_f.write('\t'.join(['Root'] + topics) + os.linesep)
        for topic in topics:
            out_f.write('\t'.join([topic] + sorted(taxonomy[topic])) + os.linesep)

    doc_df = pd.read_csv(DOC_TOPIC_TAG0624, sep='\t')[['topic', 'tag', 'title_cut_search']].dropna(how='any')
    url_df = pd.read_csv(URL_TOPIC_TAG0624, sep='\t')[['topic', 'tag', 'title_cut_search']].dropna(how='any')
    df = pd.concat([url_df, doc_df], ignore_index=True).reset_index(drop=True)

    tqdm.pandas(desc="filter sents", leave=False, ncols=100, mininterval=1)
    df['drop'] = df.progress_apply(lambda r: 1 if r['topic'] in taxonomy
                                                  and r['tag'] in taxonomy[r['topic']]
                                                  and len(r['title_cut_search']) > 0 else 0, axis=1)
    train = df[df['drop'] > 0][['topic', 'tag', 'title_cut_search']]
    tv_n = 100000
    train_n = 500000
    test = train.sample(n=tv_n)
    train = train.drop(test.index)
    valid = train.sample(n=tv_n)
    train = train.drop(valid.index).sample(n=train_n)
    dfs = [(train, os.path.join(dataset_dir, dataset_name + '_train.json')),
           (valid, os.path.join(dataset_dir, dataset_name + '_valid.json')),
           (test, os.path.join(dataset_dir, dataset_name + '_test.json'))]
    for df, out_file in dfs:
        with jsonlines.open(out_file, 'w') as out_f:
            for idx, row in tqdm(df.iterrows(), total=len(df), leave=False, ncols=100, mininterval=1,
                                 desc=out_file.split('/')[-1]):
                doc_token = filter_words(row['title_cut_search'].split(' '))
                if len(doc_token) == 0:
                    continue
                doc_label = [row['topic'], row['tag']]
                doc = {'doc_token': doc_token, 'doc_label': doc_label, 'doc_keyword': [], 'doc_topic': []}
                out_f.write(doc)

    doc_df = pd.read_csv(DOC_TOPIC_TAG0624, sep='\t')[[IID, 'title_cut_search']].dropna(how='any')
    doc_df['url_id'] = -1
    url_df = pd.read_csv(URL_TOPIC_TAG0624, sep='\t')[['url_id', 'title_cut_search']].dropna(how='any')
    url_df[IID] = -1
    predict_df = pd.concat([doc_df, url_df], ignore_index=True).reset_index(drop=True)
    with jsonlines.open(os.path.join(dataset_dir, dataset_name + '_predict.json'), 'w') as out_f:
        for idx, row in tqdm(predict_df.iterrows(), total=len(predict_df), leave=False, ncols=100, mininterval=1,
                             desc=dataset_name + '_predict.json'):
            # doc_token = filter_words(row['title_cut_search'].split(' '))
            doc_token = row['title_cut_search'].split(' ')
            if len(doc_token) == 0:
                continue
            doc = {'url_id': row['url_id'], IID: row[IID],
                   'doc_token': doc_token, 'doc_label': [], 'doc_keyword': [], 'doc_topic': []}
            out_f.write(doc)
    return


def dataset_word_dict(dataset_name):
    dataset_dir = os.path.join(DATASET_DIR, dataset_name)
    out_file = os.path.join(dataset_dir, dataset_name + DICT_SUFFIX)

    word_dict = {'': [0, 0, 0, 0, 0, 0, NAN_T]}  # wid, df, f0, f1, f2, f3, type
    wid_idx, df_idx, f0_idx, f1_idx, f2_idx, f3_idx, type_idx = 0, 1, 2, 3, 4, 5, 6
    for file_path in [DOC_TEXT_FILE, URL_TEXT_FILE, QUERY_TEXT_FILE]:
        in_df = pd.read_csv(file_path, sep='\t').fillna('')
        for idx, row in tqdm(in_df.iterrows(),
                             total=len(in_df), leave=False, ncols=100, mininterval=1, desc=file_path.split('/')[-1]):
            all_words = []
            if 'topic' in row:
                topics = row['topic'].strip().split(' ')
                all_words.extend([(t, TOPIC_T) for t in topics if t != ''])
            if 'tag' in row:
                tag = row['tag'].strip().split(' ')
                all_words.extend([(t, TAG_T) for t in tag if t != ''])
            if 'keywords' in row:
                keywords = row['keywords'].strip().split(' ')
                all_words.extend([(t, KEYWORD_T) for t in keywords if t != ''])
            if 'keywords_content' in row:
                keywords_content = row['keywords_content'].strip().split(' ')
                all_words.extend([(t, KEYWORD_T) for t in keywords_content if t != ''])
            if 'title_cut_search' in row:
                title_words = row['title_cut_search'].strip().split(' ')
                all_words.extend([(t, WORD_T) for t in title_words if t != ''])
            if 'query_cut_search' in row:
                query_words = row['query_cut_search'].strip().split(' ')
                all_words.extend([(t, WORD_T) for t in query_words if t != ''])

            visited = set([])
            for word, w_type in all_words:
                if word not in word_dict:
                    word_dict[word] = [len(word_dict), 0, 0, 0, 0, 0, w_type]
                word_dict[word][w_type + 2] += 1
                if word not in visited:
                    word_dict[word][df_idx] += 1
                if w_type > word_dict[word][type_idx]:
                    word_dict[word][type_idx] = w_type
                visited.add(word)
    out_df = pd.DataFrame.from_dict(word_dict, orient='index')
    columns = [C_WORD, C_WORD_ID, 'df', 'f0', 'f1', 'f2', 'f3', 'word_type']
    out_df.columns = columns[1:]
    out_df[C_WORD] = out_df.index
    out_df = out_df.reset_index(drop=True)
    out_df.loc[1:] = out_df[1:].sort_values(by=['df', 'word_type'], ascending=False).values
    out_df[C_WORD_ID] = range(len(out_df))
    out_df = out_df[columns].reset_index(drop=True)
    print(out_df)
    out_df.to_csv(out_file, sep='\t', index=False)
    return


def dataset_read_word_dict(dataset_name, dict_size):
    dataset_dir = os.path.join(DATASET_DIR, dataset_name)
    word_dict_file = os.path.join(dataset_dir, dataset_name + DICT_SUFFIX)
    word_dict_df = pd.read_csv(word_dict_file, sep='\t', na_filter=False).fillna('')
    if dict_size > 0:
        word_dict_df = word_dict_df[word_dict_df[C_WORD_ID] < dict_size]
    # print(word_dict_df)
    # word_dict = dict(zip(word_dict_df[key_c], word_dict_df[value_c]))
    return word_dict_df


def split_dataset(dataset_name, start_time, end_time, leave_n=1, warm_n=0):
    dataset_dir = os.path.join(DATASET_DIR, dataset_name)

    def dataset_id(df, c):
        id_list = [i for i in sorted(df[c].unique()) if i > 0]
        id_df = pd.DataFrame({c: id_list})
        d_c = c + '_dataset'
        id_df[d_c] = id_df.index + 1
        id_df.to_csv(os.path.join(dataset_dir, dataset_name + '.{}.csv'.format(d_c)), sep='\t', index=False)
        id_dict = dict(zip(id_df[c], id_df[d_c]))
        df[c] = df[c].apply(lambda x: id_dict[x] if x in id_dict else x)
        return df

    rec_df = pd.read_csv(REC_INTERACTION_FILE, sep='\t')
    # 2020-02-11 23:59:58
    # 2020-03-12 23:59:59
    # print(datetime.fromtimestamp(rec_df[TIME].min()).strftime('%Y-%m-%d %H:%M:%S'))
    # print(datetime.fromtimestamp(rec_df[TIME].max()).strftime('%Y-%m-%d %H:%M:%S'))
    rec_df = rec_df[rec_df[LABEL] > 0]
    rec_df = rec_df[(rec_df[TIME] >= start_time) & (rec_df[TIME] < end_time)]
    uids = Counter(rec_df[UID]).most_common()
    uids = [u[0] for u in uids if u[1] < 100]
    uids_set = set(uids)
    rec_df = rec_df[rec_df[UID].isin(uids_set)]
    rec_df = dataset_id(rec_df, IID)  # new IID

    search_df = pd.read_csv(SEARCH_INTERACTION_FILE, sep='\t')
    search_df = search_df[(search_df[TIME] >= start_time) & (search_df[TIME] < end_time)]
    search_df = search_df[search_df[UID].isin(uids_set)]
    search_df = dataset_id(search_df, 'url_id')  # new url id
    search_df = dataset_id(search_df, 'query_id')  # new query_id
    search_df = search_df.rename(columns={'url_id': IID})
    search_df[LABEL] = -1

    columns = [UID, IID, LABEL, TIME]
    inter_df = pd.concat([rec_df, search_df], ignore_index=True)
    inter_df = inter_df.fillna(-1).astype(int)
    inter_df = inter_df.sort_values(by=[TIME, UID, 'position', IID]).reset_index(drop=True)
    inter_df = dataset_id(inter_df, UID)  # new UID

    print(inter_df)
    inter_df.info()
    print(datetime.fromtimestamp(inter_df[TIME].max()).strftime('%Y-%m-%d %H:%M:%S'))
    print(datetime.fromtimestamp(inter_df[TIME].min()).strftime('%Y-%m-%d %H:%M:%S'))

    all_inter_file = os.path.join(dataset_dir, dataset_name + '.all.csv')
    inter_df.to_csv(all_inter_file, sep='\t', index=False)
    leave_out_by_time_csv(all_data_file=all_inter_file, dataset_name=dataset_name,
                          leave_n=leave_n, warm_n=warm_n, max_user=10000)
    return


def split_dataset_by_time(dataset_name, start_time, train_time, vt_time, drop_neg=True, max_click=100):
    dataset_dir = os.path.join(DATASET_DIR, dataset_name)

    def dataset_id(df, c):
        id_list = [i for i in sorted(df[c].unique()) if i > 0]
        id_df = pd.DataFrame({c: id_list})
        d_c = c + '_dataset'
        id_df[d_c] = id_df.index + 1
        id_df.to_csv(os.path.join(dataset_dir, dataset_name + '.{}.csv'.format(d_c)), sep='\t', index=False)
        id_dict = dict(zip(id_df[c], id_df[d_c]))
        df[c] = df[c].apply(lambda x: id_dict[x] if x in id_dict else x)
        return df, id_dict

    def print_df_time(df, des):
        print(des, end=':')
        print(datetime.fromtimestamp(df[TIME].min()).strftime('%Y-%m-%d %H:%M:%S'), end='->')
        print(datetime.fromtimestamp(df[TIME].max()).strftime('%Y-%m-%d %H:%M:%S'), end=os.linesep)
        return

    rec_df = pd.read_csv(REC_INTERACTION_FILE, sep='\t')
    rec_df = rec_df[(rec_df[TIME] >= start_time) & (rec_df[TIME] < vt_time)]
    rec_pos_df = rec_df[rec_df[LABEL] > 0]
    uids = Counter(rec_pos_df[UID]).most_common()

    if drop_neg:
        uids = [u[0] for u in uids if u[1] < max_click]
        uids_set = set(uids)
        vt_uids = np.random.choice(uids, size=20000, replace=False)
        valid_uids = set(vt_uids[:10000])
        test_uids = set(vt_uids[10000:])
    else:
        uids = [u[0] for u in uids if 10000 >= u[1] >= 3]
        uids = np.random.choice(uids, size=10000, replace=False)
        uids_set = set(uids)
        vt_uids = np.random.choice(uids, size=2000, replace=False)
        valid_uids = set(vt_uids[:1000])
        test_uids = set(vt_uids[1000:])

    rec_df = rec_df[rec_df[UID].isin(uids_set)]
    rec_df, _ = dataset_id(rec_df, IID)  # new IID

    uids, iids, timestps, labels = rec_df[UID].values, rec_df[IID].values, rec_df[TIME].values, rec_df[LABEL].values
    neg_dict = defaultdict(list)
    for idx in tqdm(range(len(uids)), leave=False, ncols=100, mininterval=1, desc='neg_dict'):
        uid, iid, timestp, label = uids[idx], iids[idx], timestps[idx], labels[idx]
        if label == 0:
            neg_dict[(uid, timestp)].append(str(iid))
    if drop_neg:
        rec_pos_df = rec_df[rec_df[LABEL] > 0]
    else:
        rec_pos_df = rec_df
    rec_train_df = rec_pos_df[(rec_pos_df[TIME] >= start_time) & (rec_pos_df[TIME] < train_time)]
    print('rec train: {}'.format(len(rec_train_df)))
    rec_valid_df = rec_pos_df[(rec_pos_df[TIME] >= train_time) & (rec_pos_df[TIME] < vt_time)
                              & (rec_pos_df[UID].isin(valid_uids))]
    print('rec valid: {}'.format(len(rec_valid_df)))
    rec_test_df = rec_df[(rec_df[TIME] >= train_time) & (rec_df[TIME] < vt_time)
                         & (rec_df[UID].isin(test_uids))]
    print('rec test: {}'.format(len(rec_test_df)))
    rec_df = pd.concat([rec_train_df, rec_valid_df, rec_test_df])

    search_df = pd.read_csv(SEARCH_INTERACTION_FILE, sep='\t')
    search_df = search_df[(search_df[TIME] >= start_time) & (search_df[TIME] < vt_time)]
    search_df = search_df[search_df[UID].isin(uids_set)]
    search_df, _ = dataset_id(search_df, 'url_id')  # new url id
    search_df, _ = dataset_id(search_df, 'query_id')  # new query_id
    search_df = search_df.rename(columns={'url_id': IID})
    search_df[LABEL] = -1

    inter_df = pd.concat([rec_df, search_df], ignore_index=True)
    inter_df = inter_df.fillna(-1).astype(int)
    inter_df = inter_df.sort_values(by=[UID, TIME, 'position', IID]).reset_index(drop=True)

    tqdm.pandas(desc='neg_iids', leave=False, ncols=100, mininterval=1)
    inter_df['neg_iids'] = inter_df.progress_apply(
        lambda r: ','.join(neg_dict[(r[UID], r[TIME])]) if r[LABEL] > 0 else '', axis=1)
    inter_df, uid_dict = dataset_id(inter_df, UID)  # new UID

    print(inter_df)
    inter_df.info()
    print_df_time(inter_df, 'inter_df')

    valid_uids = set([uid_dict[uid] for uid in valid_uids])
    test_uids = set([uid_dict[uid] for uid in test_uids])
    train_df = inter_df[(inter_df[TIME] >= start_time) & (inter_df[TIME] < train_time)]
    valid_df = inter_df[(inter_df[TIME] >= train_time) & (inter_df[TIME] < vt_time) & (inter_df[UID].isin(valid_uids))]
    system_df = inter_df[(inter_df[TIME] >= train_time) & (inter_df[TIME] < vt_time) & (inter_df[UID].isin(test_uids))]
    if drop_neg:
        test_df = system_df[system_df[LABEL] != 0]
    else:
        test_df = system_df

    print_df_time(train_df, 'train_df')
    train_df.info(null_counts=True)
    print_df_time(valid_df, 'valid_df')
    valid_df.info(null_counts=True)
    print_df_time(system_df, 'system_df')
    system_df.info(null_counts=True)
    print_df_time(test_df, 'test_df')
    test_df.info(null_counts=True)
    if drop_neg:
        assert len(train_df[train_df[LABEL] == 0]) == 0
        assert len(valid_df[valid_df[LABEL] == 0]) == 0
        assert len(test_df[test_df[LABEL] == 0]) == 0

    train_file = os.path.join(dataset_dir, dataset_name + TRAIN_SUFFIX)
    valid_file = os.path.join(dataset_dir, dataset_name + VALIDATION_SUFFIX)
    test_file = os.path.join(dataset_dir, dataset_name + TEST_SUFFIX)
    system_file = os.path.join(dataset_dir, dataset_name + '.system.csv')
    train_df.to_csv(train_file, sep='\t', index=False)
    valid_df.to_csv(valid_file, sep='\t', index=False)
    test_df.to_csv(test_file, sep='\t', index=False)
    system_df.to_csv(system_file, sep='\t', index=False)
    return


def dataset_text(dataset_name):
    dataset_dir = os.path.join(DATASET_DIR, dataset_name)
    # word_dict = dataset_read_word_dict(dataset_name=dataset_name, dict_size=DEFAULT_DICT_SIZE)
    word_dict = dataset_filter_words(dataset_name=dataset_name)
    word2id = dict(zip(word_dict[C_WORD], word_dict[C_WORD_ID]))

    text_files = [(DOC_TEXT_FILE, DOC_TEXT_SUFFIX, IID), (URL_TEXT_FILE, URL_TEXT_SUFFIX, 'url_id'),
                  (QUERY_TEXT_FILE, QUERY_TEXT_SUFFIX, 'query_id')]
    # text_files = [(QUERY_TEXT_FILE, QUERY_TEXT_SUFFIX, 'query_id')]
    possible_columns = [
        'topic', 'tag', 'keywords', 'keywords_content',
        'title_cut', 'title_cut_search', 'query_cut', 'query_cut_search']
    for text_file, dataset_suffix, id_c in text_files:
        text_df = pd.read_csv(text_file, sep='\t').fillna('')
        columns = [c for c in possible_columns if c in text_df]
        for c in columns:
            tqdm.pandas(desc=c, leave=False, ncols=100, mininterval=1)
            text_df[c] = text_df[c].progress_apply(
                lambda x: ','.join([str(word2id[w]) for w in x.split(' ') if w != '' and w in word2id]))
        text_df = text_df[[id_c] + columns]
        print(text_df)
        text_df.info()
        text_df.to_csv(os.path.join(dataset_dir, dataset_name + dataset_suffix.replace('.csv', '_all.csv')),
                       sep='\t', index=False)

        id_cd = id_c + '_dataset'
        id_dict = pd.read_csv(os.path.join(dataset_dir, dataset_name + '.{}.csv'.format(id_cd)), sep='\t')
        id_dict = dict(zip(id_dict[id_c], id_dict[id_cd]))
        text_df[id_c] = text_df[id_c].apply(lambda x: id_dict[x] if x in id_dict else -1)
        text_df = text_df[text_df[id_c] > 0]
        text_df.to_csv(os.path.join(dataset_dir, dataset_name + dataset_suffix), sep='\t', index=False)
    return


def dataset_filter_words(dataset_name):
    word_dict = dataset_read_word_dict(dataset_name=dataset_name, dict_size=DEFAULT_DICT_SIZE)
    df = word_dict[word_dict[C_WORD].str.len() > 1]

    def check_number(s):
        try:
            f = float(s)
        except:
            return False
        return True

    df = df[~df[C_WORD].apply(check_number)]
    stopwords_f = open(STOPWORDS_FILE, 'r')
    stopwords = stopwords_f.readlines()
    stopwords_f.close()
    stopwords = set([w.strip() for w in stopwords])
    df = df[~df[C_WORD].apply(lambda x: x in stopwords)]
    return df


def dataset_word2vec(dataset_name, top_n=100):
    out_file = os.path.join(DATASET_DIR, dataset_name, dataset_name + DICT_WORD2VEC_NET)
    model = KeyedVectors.load_word2vec_format(WORD2VEC_FILE, binary=False)

    words_df = dataset_filter_words(dataset_name=dataset_name)
    word2type = dict(zip(words_df[C_WORD].tolist(), words_df['word_type'].tolist()))
    word2id = dict(zip(words_df[C_WORD].tolist(), words_df[C_WORD_ID].tolist()))
    id2word = dict(zip(words_df[C_WORD_ID].tolist(), words_df[C_WORD].tolist()))
    words_list = words_df[C_WORD].tolist()
    words_set = set(words_list)

    result_dict = {}
    for w in tqdm(words_list, leave=False, ncols=100, mininterval=1, desc=out_file.split('/')[-1]):
        wid = word2id[w]
        result_dict[wid] = []
        if w in model:
            top_ws = model.most_similar(positive=w, topn=top_n * 10)
            top_ws = [(word2id[t[0]], word2type[t[0]], t[1]) for t in top_ws if t[0] in words_set][:top_n]
            result_dict[wid] = top_ws
    pickle.dump(result_dict, open(out_file, 'wb'))

    for w in words_list[:10] + words_list[10000:10010]:
        print(os.linesep)
        print(w, word2type[w], len(result_dict[word2id[w]]))
        for t in result_dict[word2id[w]][:100]:
            print('%s\t\t%d\t%.4E' % (id2word[t[0]], t[1], t[2]))
    return


def dataset_read_words_dict(dataset_name, text_file):
    words_df = dataset_filter_words(dataset_name=dataset_name)
    wid_list = words_df[C_WORD_ID].tolist()
    wid_set = set(wid_list)
    text_df = pd.read_csv(text_file, sep='\t', keep_default_na=False)
    tqdm.pandas(desc="title_cut_search", leave=False, ncols=100, mininterval=1)
    text_df['title_cut_search'] = text_df['title_cut_search'].astype(str).progress_apply(
        lambda x: [int(w) for w in x.split(',') if w != ''])
    text_df['title_cut_search'] = text_df['title_cut_search'].progress_apply(
        lambda x: [w for w in x if w in wid_set])
    tqdm.pandas(desc="topic", leave=False, ncols=100, mininterval=1)
    text_df['topic'] = text_df['topic'].astype(str).progress_apply(
        lambda x: [int(w) for w in x.split(',') if w != ''])
    text_df['topic'] = text_df['topic'].progress_apply(
        lambda x: [w for w in x if w in wid_set])
    words_dict = dict(zip(text_df[text_df.columns[0]], text_df['title_cut_search']))
    topic_dict = dict(zip(text_df[text_df.columns[0]], text_df['topic']))
    return words_dict, topic_dict


def dataset_word_net(dataset_name, word_type=KEYWORD_T, min_tfidf=-1.0, max_n=100, url=True):
    dataset_dir = os.path.join(DATASET_DIR, dataset_name)
    out_file = os.path.join(dataset_dir, dataset_name + DICT_WORD_CO_NET)
    if word_type == TOPIC_T:
        out_file = os.path.join(dataset_dir, dataset_name + DICT_WORD_TOPIC_NET)
    words_df = dataset_filter_words(dataset_name=dataset_name)
    wid2type = dict(zip(words_df[C_WORD_ID].tolist(), words_df['word_type'].tolist()))
    wid_list = words_df[C_WORD_ID].tolist()

    words_co_net = {}
    for word in wid_list:
        words_co_net[word] = {}

    text_item_files = [os.path.join(dataset_dir, dataset_name + DOC_TEXT_SUFFIX.replace('.csv', '_all.csv'))]
    if url:
        text_item_files.append(os.path.join(dataset_dir, dataset_name + URL_TEXT_SUFFIX.replace('.csv', '_all.csv')))

    for text_item_file in text_item_files:
        words_dict, topic_dict = dataset_read_words_dict(dataset_name, text_item_file)
        for item in tqdm(words_dict, total=len(words_dict), leave=False, ncols=100, mininterval=1,
                         desc=text_item_file.split('/')[-1]):
            left_words = list(set(words_dict[item] + topic_dict[item]))
            # topic_words = [t for t in left_words if wid2type[t] == TOPIC_T]  # topic words should include that in title
            # right_words = words_dict[item]
            right_words = [w for w in words_dict[item] if wid2type[w] >= word_type]  # do not count regular words
            for l in left_words:
                for r in right_words:
                    if l == r: continue
                    if r not in words_co_net[l]:
                        words_co_net[l][r] = 1
                    else:
                        words_co_net[l][r] += 1

    dataset_prune_word_net(dataset_name, word_net=words_co_net, out_file=out_file, min_tfidf=min_tfidf, max_n=max_n)
    return


def dataset_prune_word_net(dataset_name, word_net, out_file, min_tfidf, max_n):
    words_df = dataset_filter_words(dataset_name=dataset_name)
    wid2type = dict(zip(words_df[C_WORD_ID].tolist(), words_df['word_type'].tolist()))
    id2word = dict(zip(words_df[C_WORD_ID].tolist(), words_df[C_WORD].tolist()))
    wid_list = words_df[C_WORD_ID].tolist()

    doc_lengths = [len(word_net[w]) for w in word_net]
    hist, bin_edges = np.histogram(doc_lengths, bins=[0, 1, 10, 100, 1000, 5000, 10000, 50000, 100000])
    print(hist)
    print(bin_edges)

    docs = list(word_net.keys())
    df, dl = defaultdict(int), {}
    for doc in tqdm(wid_list, leave=False, ncols=100, mininterval=1, desc='df-dl'):
        dl[doc] = 0
        if doc not in word_net:
            # print(doc, 'not in all_dict')
            continue
        for word in word_net[doc]:
            df[word] += 1
            dl[doc] += word_net[doc][word]
    result_dict, doc_lengths = {}, []
    for doc in tqdm(wid_list, leave=False, ncols=100, mininterval=1, desc=out_file.split('/')[-1]):
        # for doc in docs:
        if doc not in word_net or len(word_net[doc]) == 0:
            result_dict[doc] = []
            continue
        words = word_net[doc].keys()
        tfs = [(w, 1.0 * word_net[doc][w] / (dl[doc] + 1)) for w in words]
        idfs = [(w, math.log10(1.0 * (len(docs) + 1) / (df[w] + 1))) for w in words]
        tfidfs = [(w, wid2type[w], tfs[idx][1] * idfs[idx][1]) for idx, w in enumerate(words)]
        tfidfs = sorted(tfidfs, key=lambda x: x[2], reverse=True)
        if min_tfidf > 0:
            tfidfs = [t for t in tfidfs if t[2] > min_tfidf]
        doc_lengths.append(len(tfidfs))
        if max_n > 0:
            result_dict[doc] = tfidfs[:max_n]
        else:
            result_dict[doc] = tfidfs

    hist, bin_edges = np.histogram(doc_lengths, bins=[0, 1, 10, 100, 1000, 5000, 10000, 50000, 100000])
    print(hist)
    print(bin_edges)
    pickle.dump(result_dict, open(out_file, 'wb'))

    result_dict = pickle.load(open(out_file, 'rb'))
    for w in wid_list[:10] + wid_list[10000:10010]:
        print(os.linesep)
        print(id2word[w], wid2type[w], len(result_dict[w]))
        for t in result_dict[w][:100]:
            print('%s\t\t%d\t%.4E' % (id2word[t[0]], t[1], t[2]))
    return result_dict


def dataset_word_cf_net(dataset_name, word_type=KEYWORD_T, max_n=100, url=True):
    dataset_dir = os.path.join(DATASET_DIR, dataset_name)
    out_file = os.path.join(dataset_dir, dataset_name + DICT_WORD_CF_NET)
    if word_type == TOPIC_T:
        out_file = os.path.join(dataset_dir, dataset_name + DICT_TOPIC_CF_NET)

    words_df = dataset_filter_words(dataset_name=dataset_name)
    wid2type = dict(zip(words_df[C_WORD_ID].tolist(), words_df['word_type'].tolist()))

    doc_words_dict, doc_topic_dict = dataset_read_words_dict(
        dataset_name, os.path.join(os.path.join(dataset_dir, dataset_name + DOC_TEXT_SUFFIX)))
    url_words_dict, url_topic_dict = dataset_read_words_dict(
        dataset_name, os.path.join(os.path.join(dataset_dir, dataset_name + URL_TEXT_SUFFIX)))

    train_inter_df = pd.read_csv(os.path.join(dataset_dir, dataset_name + TRAIN_SUFFIX), sep='\t')
    user_his_dict = {}
    uids, iids, labels = train_inter_df[UID].tolist(), train_inter_df[IID].tolist(), train_inter_df[LABEL].tolist()
    for idx in tqdm(range(len(uids)), total=len(uids), leave=False, ncols=100, mininterval=1,
                    desc=dataset_name + TRAIN_SUFFIX):
        uid, iid, label = uids[idx], iids[idx], labels[idx]
        if uid not in user_his_dict:
            user_his_dict[uid] = []
        if label > 0:
            user_his_dict[uid].append(iid)
        elif url:
            user_his_dict[uid].append(-iid)

    word_cf_net = {}
    for user in tqdm(user_his_dict, total=len(user_his_dict), leave=False, ncols=100, mininterval=1,
                     desc=dataset_name + TRAIN_SUFFIX):
        history = list(set(user_his_dict[user]))
        for left in history:
            if left > 0 and left in doc_words_dict:
                lws = list(set(doc_words_dict[left] + doc_topic_dict[left]))
            elif -left in url_words_dict:
                lws = list(set(url_words_dict[-left] + url_words_dict[-left]))
            else:
                continue
            rws = set([])
            for right in history:
                if left == right: continue
                tmp_l = doc_words_dict[right] if right > 0 and right in doc_words_dict else \
                    url_words_dict[-right] if -right in url_words_dict else []
                for w in tmp_l:
                    if wid2type[w] >= word_type:
                        rws.add(w)
            for lw in lws:
                for rw in rws:
                    if lw not in word_cf_net:
                        word_cf_net[lw] = {}
                    if rw not in word_cf_net[lw]:
                        word_cf_net[lw][rw] = 0
                    word_cf_net[lw][rw] += 1
    dataset_prune_word_net(dataset_name, word_net=word_cf_net, out_file=out_file, min_tfidf=-1, max_n=max_n)
    return


def dataset_word_graph(dataset_name):
    dataset_dir = os.path.join(DATASET_DIR, dataset_name)

    words_df = dataset_filter_words(dataset_name=dataset_name)
    max_wid = words_df[C_WORD_ID].max()

    topic_net = pickle.load(open(os.path.join(dataset_dir, dataset_name + DICT_WORD_TOPIC_NET), 'rb'))
    topic_cf_net = pickle.load(open(os.path.join(dataset_dir, dataset_name + DICT_TOPIC_CF_NET), 'rb'))
    word_cf_net = pickle.load(open(os.path.join(dataset_dir, dataset_name + DICT_WORD_CF_NET), 'rb'))
    w2v_net = pickle.load(open(os.path.join(dataset_dir, dataset_name + DICT_WORD2VEC_NET), 'rb'))
    word_co_net = pickle.load(open(os.path.join(dataset_dir, dataset_name + DICT_WORD_CO_NET), 'rb'))

    graph_list = [(topic_net, 2, C_WORD_TOPIC), (topic_cf_net, 2, C_TOPIC_CF),
                  (w2v_net, 100, C_W2V_SIM), (word_co_net, 100, C_WORD_CO),
                  (word_cf_net, 100, C_WORD_CF)]
    graph_dict = {}
    for graph, max_w, column in graph_list:
        column_len = column + '_length'
        graph_array, length_array = [], []
        for wid in tqdm(range(max_wid + 1), total=max_wid, leave=False, ncols=100, mininterval=1, desc=column):
            if wid not in graph:
                graph_array.append([0] * max_w)
                length_array.append(0)
                continue
            connect = graph[wid][:max_w]
            length_array.append(len(connect))
            connect = [c[0] for c in connect] + [0] * (max_w - len(connect))
            graph_array.append(connect)
        graph_dict[column] = graph_array
        graph_dict[column_len] = length_array
    pickle.dump(graph_dict, open(os.path.join(dataset_dir, dataset_name + WORD_GRAPH_SUFFIX), 'wb'), protocol=4)
    return


def tencent_predict_topic_tag():
    taxonomy = read_topic_tag_taxonomy(DOC_TOPIC_TAG_TAXONOMY)

    predict_json = os.path.join(TOPIC_TAG_DATASET_DIR, 'sgnews_predict.json')
    predict_json_df = pd.read_json(predict_json, lines=True)[['url_id', IID]]

    predict_topics, predict_tags = [], []
    with open(os.path.join(TOPIC_TAG_DATASET_DIR, 'predict0627.txt'), 'r') as predict_txt:
        for line in predict_txt:
            words = [w for w in line.strip().split(';') if w != '']
            topic, tag = '', ''
            for word in words:
                if word in taxonomy:
                    topic = word
                    break
            if topic != '':
                for word in words:
                    if word in taxonomy[topic]:
                        tag = word
                        break
            predict_topics.append(topic)
            predict_tags.append(tag)
    predict_json_df['predict_topic'] = predict_topics
    predict_json_df['predict_tag'] = predict_tags
    predict_doc_df = predict_json_df[predict_json_df[IID] > 0][[IID, 'predict_topic', 'predict_tag']]
    predict_url_df = predict_json_df[predict_json_df['url_id'] > 0][['url_id', 'predict_topic', 'predict_tag']]

    doc_df = pd.read_csv(DOC_TOPIC_TAG0624, sep='\t')
    doc_df = pd.merge(left=doc_df, right=predict_doc_df, on=IID, how='left').fillna('')
    doc_df['topic'] = doc_df.apply(lambda r: r['topic'] if r['topic'] != '' else r['predict_topic'], axis=1)
    doc_df['tag'] = doc_df.apply(lambda r: r['tag'] if r['tag'] != '' else r['predict_tag'], axis=1)

    url_df = pd.read_csv(URL_TOPIC_TAG0624, sep='\t')
    url_df = pd.merge(left=url_df, right=predict_url_df, on='url_id', how='left').fillna('')
    url_df['topic'] = url_df.apply(lambda r: r['topic'] if r['topic'] != '' else r['predict_topic'], axis=1)
    url_df['tag'] = url_df.apply(lambda r: r['tag'] if r['tag'] != '' else r['predict_tag'], axis=1)

    print(doc_df)
    print(url_df)

    doc_df.to_csv(DOC_TEXT_FILE, sep='\t', index=False)
    url_df.to_csv(URL_TEXT_FILE, sep='\t', index=False)
    return


def raw_topic_tag0624_2csv(in_file, out_file, url=False):
    df = []
    with open(in_file, 'r', encoding='gbk') as in_f:
        for line in tqdm(in_f, desc=in_file.split('/')[-1], leave=False, ncols=100, mininterval=1):
            raw_id, title, topic_tag = line.strip().split('\t')
            topic, tag = topic_tag[1:-1].split('}{')
            if topic == '' and tag == '':
                continue
            topic = ','.join([t.split(':')[0] for t in topic.split(',') if t != ''])
            tag = ','.join([t.split(':')[0] for t in tag.split(',') if t != ''])
            # df.append((raw_id, title, topic, tag))
            df.append((raw_id, title, topic.split(',')[0], tag.split(',')[0]))
    df = pd.DataFrame(df, columns=['raw_id', 'title', 'topic', 'tag'])

    if url:
        df = df_transfer_id(df, column='raw_id', dict_csv=URL_ID_DICT, key_c='url', value_c='url_id')
        print(df)
        topic_dict = dict(zip(df['raw_id'], df['topic']))
        tag_dict = dict(zip(df['raw_id'], df['tag']))
        text_df = pd.read_csv(URL_TEXT_ORIGIN_FILE, '\t')
        text_df['raw_topic'] = text_df['topic']
        text_df['topic'] = text_df.apply(lambda r: topic_dict[r['url_id']] if r['url_id'] in topic_dict else '', axis=1)
        text_df['raw_tag'] = text_df['tag']
        text_df['tag'] = text_df.apply(lambda r: tag_dict[r['url_id']] if r['url_id'] in tag_dict else '', axis=1)
    else:
        df = df_transfer_id(df, column='raw_id', dict_csv=ITEM_ID_DICT, key_c='doc_id', value_c=IID)
        topic_dict = dict(zip(df['raw_id'], df['topic']))
        tag_dict = dict(zip(df['raw_id'], df['tag']))
        text_df = pd.read_csv(DOC_TEXT_ORIGIN_FILE, '\t')
        text_df['raw_topic'] = text_df['topic']
        text_df['topic'] = text_df.apply(lambda r: topic_dict[r[IID]] if r[IID] in topic_dict else '', axis=1)
        text_df['raw_tag'] = text_df['tag']
        text_df['tag'] = text_df.apply(lambda r: tag_dict[r[IID]] if r[IID] in tag_dict else '', axis=1)
    text_df.to_csv(out_file, sep='\t', index=False)
    return


def main():
    # 预处理Log日志转为UTF-8
    # transfer2utf8(in_file=RAW_REC_FILE, out_file=REC_FILE, encoding='gbk', sep='\a', column_n=len(REC_COLUMNS))
    # transfer2utf8(in_file=RAW_DOC_FILE, out_file=DOC_FILE, encoding='gbk', sep='\t', column_n=len(DOC_COLUMNS))
    # transfer2utf8(in_file=RAW_SEARCH_FILE, out_file=SEARCH_FILE, encoding='utf8',
    #               sep='\t', column_n=len(SEARCH_COLUMNS))
    # transfer2utf8(in_file=RAW_URL_FILE, out_file=URL_FILE, encoding='utf8', errors='ignore')
    # transfer2utf8(in_file=RAW_URL_FILE, out_file=URL_FILE, encoding='utf8', errors='ignore',
    #               sep='\t', column_n=len(URL_COLUMNS))

    # 预处理数据集
    # build_item_id_dict()
    # build_user_id_dict()
    # build_url_id_dict()
    # build_query_id_dict()
    # build_rec_interactions_origin()
    # build_rec_interactions()
    # build_search_interactions()
    # group_user_interactions(
    #     in_csv=REC_INTERACTION_FILE, out_csv=REC_INTER_GROUP_FILE, group_c=UID, list_c=IID, pos_neg=1)
    # group_user_interactions(
    #     in_csv=SEARCH_INTERACTION_FILE, out_csv=SEARCH_INTER_GROUP_FILE, group_c=UID, list_c='url_id')
    # group_user_interactions(
    #     in_csv=SEARCH_INTERACTION_FILE, out_csv=QUERY_INTER_GROUP_FILE, group_c=UID, list_c='query_id')
    # count_user_interactions()
    # build_doc_text_df()
    # build_url_text_df()
    # build_query_text_df()

    # # # 修正topic tag
    # raw_topic_tag0624_2csv(in_file=RAW_DOC_TAG_FILE, out_file=DOC_TOPIC_TAG0624)
    # raw_topic_tag0624_2csv(in_file=RAW_URL_TAG_FILE, out_file=URL_TOPIC_TAG0624, url=True)

    # # # 腾讯工具包预测topic tag
    # build_topic_tag_taxonomy(DOC_TOPIC_TAG_TAXONOMY)
    # build_topic_tag_tvt(taxonomy_file=DOC_TOPIC_TAG_TAXONOMY, dataset_dir=TOPIC_TAG_DATASET_DIR, dataset_name='sgnews')
    # tencent_predict_topic_tag()

    # # 划分构建数据集
    # dataset_name = 'sogou'
    # start_time = datetime(2020, 3, 1).timestamp()
    # train_time = datetime(2020, 3, 9).timestamp()
    # vt_time = datetime(2020, 3, 10).timestamp()

    # split_dataset_by_time(dataset_name=dataset_name, drop_neg=True, max_click=30,
    #                       start_time=start_time, train_time=train_time, vt_time=vt_time)
    # dataset_word_dict(dataset_name=dataset_name)
    # dataset_read_word_dict(dataset_name=dataset_name, dict_size=DEFAULT_DICT_SIZE)
    # dataset_filter_words(dataset_name=dataset_name)
    # dataset_text(dataset_name=dataset_name)

    # 生成Word Graph
    # dataset_word2vec(dataset_name=dataset_name, top_n=100)  # Semantically-Similar
    # dataset_word_net(dataset_name=dataset_name, word_type=KEYWORD_T, url=False)# Co-Occurrence
    # dataset_word_net(dataset_name=dataset_name, word_type=TOPIC_T, url=False) # Co-Occurrence
    # dataset_word_cf_net(dataset_name=dataset_name, word_type=KEYWORD_T, url=False)  # Co-Click
    # dataset_word_cf_net(dataset_name=dataset_name, word_type=TOPIC_T, url=False)  # Co-Click
    # dataset_word_graph(dataset_name) # 合并3种关系的文件
    return


if __name__ == '__main__':
    main()
