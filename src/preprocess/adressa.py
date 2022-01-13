# coding=utf-8
import sys

sys.path.insert(0, '../')
sys.path.insert(0, './')

from utils.dataset import *
from utils.global_p import *
from shutil import copyfile
import math

import pickle
from tqdm import tqdm
from gensim.models.wrappers import FastText

np.random.seed(DEFAULT_SEED)
print(socket.gethostname())

RAW_DATA = os.path.join(DATA_DIR, 'Adressa/gnud_one_week/')
OUT_DIR = os.path.join(DATA_DIR, 'Adressa/one_week/')
WORD2VEC_DIR = '/work/shisy13/Library/word2vec/'
WORD2VEC_FILE = os.path.join(WORD2VEC_DIR, 'cc.no.300.bin')

RAW_NEWS_FILE = os.path.join(RAW_DATA, 'news.csv')
RAW_PRE_INTER_FILE = os.path.join(RAW_DATA, 'pre_inters.csv')
RAW_TRAIN_INTER_FILE = os.path.join(RAW_DATA, 'train_inters.csv')
RAW_TEST_INTER_FILE = os.path.join(RAW_DATA, 'test_inters.csv')

ITEM_ID_DICT = os.path.join(OUT_DIR, 'item_id_dict.csv')
USER_ID_DICT = os.path.join(OUT_DIR, 'user_id_dict.csv')

ENTITY_T, WORD_T, NAN_T = 1, 0, -1
DEFAULT_DICT_SIZE = 500000


def id2set(in_file, id_set, sep, index):
    in_f = pd.read_csv(in_file, sep=sep)
    for idx, new_id in enumerate(in_f[index].tolist()):
        id_set.add(new_id.strip())
    return id_set


def id_set2df(id_set, raw_name, id_name):
    df = pd.DataFrame({raw_name: list(id_set)})
    df = df.sort_values(raw_name).reset_index(drop=True)
    df[id_name] = df.index
    return df


def build_item_id_dict():
    raw_name, id_name = 'item_id', IID
    item_ids = {''}
    item_ids = id2set(RAW_NEWS_FILE, item_ids, sep='\t', index='item_id')
    df = id_set2df(item_ids, raw_name, id_name)
    print(df)
    df.to_csv(ITEM_ID_DICT, sep='\t', index=False)
    return


def build_user_id_dict():
    raw_name, id_name = 'user_id', UID
    user_ids = {''}
    user_ids = id2set(RAW_PRE_INTER_FILE, user_ids, sep='\t', index='user_id')
    user_ids = id2set(RAW_TRAIN_INTER_FILE, user_ids, sep='\t', index='user_id')
    user_ids = id2set(RAW_TEST_INTER_FILE, user_ids, sep='\t', index='user_id')
    df = id_set2df(user_ids, raw_name, id_name)
    print(df)
    df.to_csv(USER_ID_DICT, sep='\t', index=False)
    return


def df_transfer_id(df, column, dict_csv, key_c, value_c, nan_v=-1):
    id_df = pd.read_csv(dict_csv, sep='\t')
    id_dict = dict(zip(id_df[key_c], id_df[value_c]))
    df[column] = df[column].apply(lambda x: id_dict[x.strip()] if x in id_dict else nan_v)
    return df


def split_dataset(dataset_name):
    pre_df = pd.read_csv(RAW_PRE_INTER_FILE, sep='\t')
    train_df = pd.read_csv(RAW_TRAIN_INTER_FILE, sep='\t')
    test_df = pd.read_csv(RAW_TEST_INTER_FILE, sep='\t')

    test_uids = test_df['user_id'].unique()
    # valid_uids = np.random.choice(test_uids, size=int(0.2 * len(test_uids)), replace=False)
    valid_uids = np.random.choice(test_uids, size=int(0.1 * len(test_uids)), replace=False)
    valid_df = test_df[test_df['user_id'].isin(valid_uids)]
    test_df = test_df.drop(valid_df.index)
    train_df = pd.concat([pre_df, train_df], ignore_index=True)

    dataset_dir = os.path.join(DATASET_DIR, dataset_name)
    train_file = os.path.join(dataset_dir, dataset_name + TRAIN_SUFFIX)
    valid_file = os.path.join(dataset_dir, dataset_name + VALIDATION_SUFFIX)
    test_file = os.path.join(dataset_dir, dataset_name + TEST_SUFFIX)
    for df, out_file in [(train_df, train_file), (valid_df, valid_file), (test_df, test_file)]:
        df = df.sort_values(by=TIME, kind='mergesort').reset_index(drop=True)
        df = df_transfer_id(df, 'item_id', dict_csv=ITEM_ID_DICT, key_c='item_id', value_c=IID)
        df = df_transfer_id(df, 'user_id', dict_csv=USER_ID_DICT, key_c='user_id', value_c=UID)
        df[LABEL] = 1
        df = df.rename(columns={'user_id': UID, 'item_id': IID})
        df = df[[UID, IID, LABEL, TIME]]
        print(df)
        df.to_csv(out_file, sep='\t', index=False)
    return


def dataset_word_dict(dataset_name):
    dataset_dir = os.path.join(DATASET_DIR, dataset_name)
    out_file = os.path.join(dataset_dir, dataset_name + DICT_SUFFIX)
    in_df = pd.read_csv(RAW_NEWS_FILE, sep='\t')
    in_df.info()

    word_dict = {'': [0, 0, 0, 0, NAN_T]}  # wid, df, f0, f1, type

    for idx, row in tqdm(in_df.iterrows(),
                         total=len(in_df), leave=False, ncols=100, mininterval=1, desc=RAW_NEWS_FILE.split('/')[-1]):
        row_words = []
        row_words.extend([(t, ENTITY_T) for t in eval(row['entity']) if t != ''])
        row_words.extend([(t, WORD_T) for t in eval(row['title']) if t != ''])
        visited = set([])
        for word, w_type in row_words:
            if word not in word_dict:
                word_dict[word] = [len(word_dict), 0, 0, 0, w_type]
            word_dict[word][w_type + 2] += 1
            word_dict[word][1] += 1
            if w_type > word_dict[word][-1]:
                word_dict[word][-1] = w_type
            visited.add(word)
    out_df = pd.DataFrame.from_dict(word_dict, orient='index')
    columns = [C_WORD, C_WORD_ID, 'df', 'f0', 'f1', 'word_type']
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
    return word_dict_df


def dataset_text(dataset_name):
    dataset_dir = os.path.join(DATASET_DIR, dataset_name)
    word_dict = dataset_read_word_dict(dataset_name, dict_size=DEFAULT_DICT_SIZE)
    word2id = dict(zip(word_dict[C_WORD], word_dict[C_WORD_ID]))

    df = pd.read_csv(RAW_NEWS_FILE, sep='\t')
    df = df.fillna('')
    df = df.drop(columns=['type', 'time'])
    for c in df.columns:
        df[c] = df[c].apply(lambda x: x.strip())
    df = df_transfer_id(df, 'item_id', dict_csv=ITEM_ID_DICT, key_c='item_id', value_c=IID)
    df['title'] = df['title'].apply(lambda x: ','.join([str(word2id[w]) for w in eval(x)]))
    df['entity'] = df['entity'].apply(lambda x: ','.join([str(word2id[w]) for w in eval(x)]))
    df = df.rename(columns={'item_id': IID, 'title': 'title_cut'})
    df = df[[IID, 'title_cut', 'entity']]
    print(df)
    df.to_csv(os.path.join(dataset_dir, dataset_name + DOC_TEXT_SUFFIX), sep='\t', index=False)
    url_df = pd.DataFrame({'url_id': [0], 'title_cut': [''], 'entity': ['']})
    url_df.to_csv(os.path.join(dataset_dir, dataset_name + URL_TEXT_SUFFIX), sep='\t', index=False)
    print(url_df)
    return


def dataset_word2vec(dataset_name, top_n=100):
    out_file = os.path.join(DATASET_DIR, dataset_name, dataset_name + DICT_WORD2VEC_NET)
    model = FastText.load_fasttext_format(WORD2VEC_FILE)

    words_df = dataset_read_word_dict(dataset_name=dataset_name, dict_size=DEFAULT_DICT_SIZE)
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
    words_df = dataset_read_word_dict(dataset_name=dataset_name, dict_size=DEFAULT_DICT_SIZE)
    wid_list = words_df[C_WORD_ID].tolist()
    wid_set = set(wid_list)
    text_df = pd.read_csv(text_file, sep='\t', keep_default_na=False)
    tqdm.pandas(desc="title_cut", leave=False, ncols=100, mininterval=1)
    text_df['title_cut'] = text_df['title_cut'].astype(str).progress_apply(
        lambda x: [int(w) for w in x.split(',') if w != ''])
    text_df['title_cut'] = text_df['title_cut'].progress_apply(
        lambda x: [w for w in x if w in wid_set])
    tqdm.pandas(desc="entity", leave=False, ncols=100, mininterval=1)
    text_df['entity'] = text_df['entity'].astype(str).progress_apply(
        lambda x: [int(w) for w in x.split(',') if w != ''])
    text_df['entity'] = text_df['entity'].progress_apply(
        lambda x: [w for w in x if w in wid_set])
    words_dict = dict(zip(text_df[text_df.columns[0]], text_df['title_cut']))
    entity_dict = dict(zip(text_df[text_df.columns[0]], text_df['entity']))
    return words_dict, entity_dict


def dataset_word_net(dataset_name, word_type=ENTITY_T, min_tfidf=-1.0, max_n=100, ):
    dataset_dir = os.path.join(DATASET_DIR, dataset_name)
    out_file = os.path.join(dataset_dir, dataset_name + DICT_WORD_CO_NET)
    words_df = dataset_read_word_dict(dataset_name=dataset_name, dict_size=DEFAULT_DICT_SIZE)
    wid2type = dict(zip(words_df[C_WORD_ID].tolist(), words_df['word_type'].tolist()))
    wid_list = words_df[C_WORD_ID].tolist()

    words_co_net = {}
    for word in wid_list:
        words_co_net[word] = {}

    text_item_files = [os.path.join(dataset_dir, dataset_name + DOC_TEXT_SUFFIX)]

    for text_item_file in text_item_files:
        words_dict, entity_dict = dataset_read_words_dict(dataset_name, text_item_file)
        for item in tqdm(words_dict, total=len(words_dict), leave=False, ncols=100, mininterval=1,
                         desc=text_item_file.split('/')[-1]):
            left_words = list(set(words_dict[item] + entity_dict[item]))
            right_words = [w for w in left_words if wid2type[w] >= word_type]  # do not count regular words
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
    words_df = dataset_read_word_dict(dataset_name=dataset_name, dict_size=DEFAULT_DICT_SIZE)
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

        tfidfs = [t for t in tfidfs if t[0] != doc]
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


def dataset_word_cf_net(dataset_name, word_type=ENTITY_T, max_n=100):
    dataset_dir = os.path.join(DATASET_DIR, dataset_name)
    out_file = os.path.join(dataset_dir, dataset_name + DICT_WORD_CF_NET)

    words_df = dataset_read_word_dict(dataset_name=dataset_name, dict_size=DEFAULT_DICT_SIZE)
    wid2type = dict(zip(words_df[C_WORD_ID].tolist(), words_df['word_type'].tolist()))

    doc_words_dict, doc_entity_dict = dataset_read_words_dict(
        dataset_name, os.path.join(os.path.join(dataset_dir, dataset_name + DOC_TEXT_SUFFIX)))

    train_inter_df = pd.read_csv(os.path.join(dataset_dir, dataset_name + TRAIN_SUFFIX), sep='\t')
    user_his_dict = {}
    uids, iids, labels = train_inter_df[UID].tolist(), train_inter_df[IID].tolist(), train_inter_df[LABEL].tolist()
    for idx in tqdm(range(len(uids)), total=len(uids), leave=False, ncols=100, mininterval=1,
                    desc='build user his'):
        uid, iid, label = uids[idx], iids[idx], labels[idx]
        if uid not in user_his_dict:
            user_his_dict[uid] = []
        if label > 0:
            user_his_dict[uid].append(iid)

    word_cf_net = {}
    for user in tqdm(user_his_dict, total=len(user_his_dict), leave=False, ncols=100, mininterval=1,
                     desc=dataset_name + TRAIN_SUFFIX):
        history = list(set(user_his_dict[user]))
        for left in history:
            if left > 0:
                lws = list(set(doc_words_dict[left] + doc_entity_dict[left]))
            else:
                continue
            rws = set([])
            for right in history:
                if left == right: continue
                tmp_l = doc_words_dict[right] + doc_entity_dict[right]
                for w in tmp_l:
                    if wid2type[w] >= word_type:
                        rws.add(w)
            for lw in lws:
                for rw in rws:
                    if lw == rw: continue
                    if lw not in word_cf_net:
                        word_cf_net[lw] = {}
                    if rw not in word_cf_net[lw]:
                        word_cf_net[lw][rw] = 0
                    word_cf_net[lw][rw] += 1
    dataset_prune_word_net(dataset_name, word_net=word_cf_net, out_file=out_file, min_tfidf=-1, max_n=max_n)
    return


def dataset_word_graph(dataset_name):
    dataset_dir = os.path.join(DATASET_DIR, dataset_name)

    words_df = dataset_read_word_dict(dataset_name=dataset_name, dict_size=DEFAULT_DICT_SIZE)
    max_wid = words_df[C_WORD_ID].max()

    w2v_net = pickle.load(open(os.path.join(dataset_dir, dataset_name + DICT_WORD2VEC_NET), 'rb'))
    word_co_net = pickle.load(open(os.path.join(dataset_dir, dataset_name + DICT_WORD_CO_NET), 'rb'))
    word_cf_net = pickle.load(open(os.path.join(dataset_dir, dataset_name + DICT_WORD_CF_NET), 'rb'))

    graph_list = [(w2v_net, 100, C_W2V_SIM), (word_co_net, 100, C_WORD_CO),
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


def main():
    # build_item_id_dict()
    # build_user_id_dict()

    # dataset_name = 'adressa-1w'

    # 划分数据集，生成文本ID
    # split_dataset(dataset_name)
    # dataset_word_dict(dataset_name)
    # dataset_text(dataset_name)

    # 生成Word Graph
    # dataset_word2vec(dataset_name, top_n=100)  # Semantically-Similar
    # dataset_word_net(dataset_name=dataset_name, word_type=ENTITY_T)  # Co-Occurrence
    # dataset_word_cf_net(dataset_name=dataset_name, word_type=ENTITY_T)  # Co-Click
    # dataset_word_graph(dataset_name)  # 合并3种关系的文件
    return


if __name__ == '__main__':
    main()
