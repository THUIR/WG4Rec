# coding=utf-8
import logging
import numpy as np
import torch
from utils.global_p import *
import os
import inspect

LOWER_METRIC_LIST = ["rmse", 'mae']

LOG_NAME_EXCLUDE = [
    'gpu', 'verbose', 'log_file', 'result_file', 'random_seed', 'train', 'debugging', 'save_res', 'rank', 'regenerate',
    'path', 'dataset', 'sep', 'seq_sep', 'label', 'dict_size',
    'load', 'check_epoch', 'check_train', 'eval_batch_size', 'pre_gpu', 'num_workers', 'gc_batch', 'gc', 'pin_memory',
    'epoch', 'early_stop', 'es_worse', 'es_long', 'metrics',
    'model_path',
    'buffer_dp',
    'eval_train', 'task_name', 'unlabel_test'
]


def balance_data(data):
    """
    让正负样本数接近，正负样本数差距太大时使用
    :param data:
    :return:
    """
    pos_indexes = np.where(data['Y'] == 1)[0]
    copy_num = int((len(data['Y']) - len(pos_indexes)) / len(pos_indexes))
    if copy_num > 1:
        copy_indexes = np.tile(pos_indexes, copy_num)
        sample_index = np.concatenate([np.arange(0, len(data['Y'])), copy_indexes])
        for k in data:
            data[k] = data[k][sample_index]
    return data


def pad_array(a, max_len, v, dtype=np.int64):
    if len(a) == 0:
        return np.array([v] * max_len, dtype=dtype)
    if len(a) < max_len:
        a = np.concatenate([a, [v] * (max_len - len(a))]).astype(dtype)
    return np.array(a, dtype=dtype)


def pad_id_array(id_array, v_array, id_max=None, sup=0):
    if id_max is None:
        id_max = id_array.max()
    assert len(id_array) == len(v_array)
    id_dict = dict(zip(id_array, v_array))
    result = []
    for i in range(id_max + 1):
        if i not in id_dict:
            result.append(sup if sup is not None else i)
        else:
            result.append(id_dict[i])
    return np.array(result, dtype=v_array.dtype)


def pad2same_length(a, max_len=-1, v=0, dtype=np.int64):
    if max_len <= 0:
        max_len = max([len(l) for l in a])
    same_length = [pad_array(l, max_len, v, dtype=dtype) for l in a]
    return np.array(same_length, dtype=dtype)


def format_metric(metric):
    """
    把计算出的评价指标转化为str，float保留四位小数
    :param metric: 一些评价指标的元组或列表，或一个单独的数
    :return:
    """
    # print(metric, type(metric))
    if type(metric) is not tuple and type(metric) is not list:
        metric = [metric]
    format_str = []
    if type(metric) is tuple or type(metric) is list:
        for m in metric:
            # print(type(m))
            if type(m) is float or type(m) is np.float or type(m) is np.float32 or type(m) is np.float64:
                format_str.append('%.4f' % m)
            elif type(m) is int or type(m) is np.int or type(m) is np.int32 or type(m) is np.int64:
                format_str.append('%d' % m)
    return ','.join(format_str)


def format_arg_str(arg_dict, max_len=20):
    """
    格式化arg的输出样式
    :param arg_dict: 参数dict
    :param max_len: value的最大长度
    """
    linesep = os.linesep
    keys, values = arg_dict.keys(), arg_dict.values()
    key_title, value_title = 'Arguments', 'Values'
    key_max_len = max(map(lambda x: len(str(x)), keys))
    value_max_len = min(max(map(lambda x: len(str(x)), values)), max_len)
    key_max_len, value_max_len = max([len(key_title), key_max_len]), max([len(value_title), value_max_len])
    horizon_len = key_max_len + value_max_len + 5
    res_str = linesep + '=' * horizon_len + linesep
    res_str += ' ' + key_title + ' ' * (key_max_len - len(key_title)) + ' | ' \
               + value_title + ' ' * (value_max_len - len(value_title)) + ' ' + linesep + '=' * horizon_len + linesep
    for key in sorted(keys):
        value = arg_dict[key]
        if value is not None:
            key, value = str(key), str(value).replace('\t', '\\t')
            value = value[:max_len - 3] + '...' if len(value) > max_len else value
            res_str += ' ' + key + ' ' * (key_max_len - len(key)) + ' | ' \
                       + value + ' ' * (value_max_len - len(value)) + linesep
    res_str += '=' * horizon_len
    return res_str


def shuffle_in_unison_scary(data):
    """
    shuffle整个数据集dict的内容
    :param data:
    :return:
    """
    rng_state = np.random.get_state()
    for d in data:
        np.random.set_state(rng_state)
        np.random.shuffle(data[d])
    return data


def best_result(metric, results_list):
    """
    求一个结果list中最佳的结果
    :param metric:
    :param results_list:
    :return:
    """
    if type(metric) is list or type(metric) is tuple:
        metric = metric[0]
    if metric in LOWER_METRIC_LIST:
        return min(results_list)
    return max(results_list)


def strictly_increasing(l):
    """
    判断是否严格单调增
    :param l:
    :return:
    """
    return all(x < y for x, y in zip(l, l[1:]))


def strictly_decreasing(l):
    """
    判断是否严格单调减
    :param l:
    :return:
    """
    return all(x > y for x, y in zip(l, l[1:]))


def non_increasing(l):
    """
    判断是否单调非增
    :param l:
    :return:
    """
    return all(x >= y for x, y in zip(l, l[1:]))


def non_decreasing(l):
    """
    判断是否单调非减
    :param l:
    :return:
    """
    return all(x <= y for x, y in zip(l, l[1:]))


def monotonic(l):
    """
    判断是否单调
    :param l:
    :return:
    """
    return non_increasing(l) or non_decreasing(l)


def numpy_to_torch(d, gpu=True, requires_grad=False):
    """
    numpy array转化为pytorch tensor，有gpu则放到gpu
    :param d:
    :param gpu: whether put tensor to gpu
    :param requires_grad: whether the tensor requires grad
    :return:
    """
    t = torch.from_numpy(d)
    if d.dtype is np.float:
        t.requires_grad = requires_grad
    if gpu:
        t = tensor_to_gpu(t)
    return t


def tensor_to_gpu(t):
    if torch.cuda.device_count() > 0:
        t = t.cuda()
    return t


def get_init_paras_dict(class_name, paras_dict):
    """
    解析class_name对应类一系列继承关系所需要的所有参数，从paras_dict中取出对应的返回
    :param class_name:
    :param paras_dict:
    :return:
    """
    base_list = inspect.getmro(class_name)
    paras_list = []
    for base in base_list:
        paras = inspect.getfullargspec(base.__init__)
        paras_list.extend(paras.args)
    paras_list = sorted(list(set(paras_list)))
    out_dict = {}
    for para in paras_list:
        if para == 'self':
            continue
        if para not in paras_dict:
            continue
        out_dict[para] = paras_dict[para]
    return out_dict


def check_dir_and_mkdir(path):
    if os.path.basename(path).find('.') == -1 or path.endswith('/'):
        dirname = path
    else:
        dirname = os.path.dirname(path)
    if not os.path.exists(dirname):
        print('make dirs:', dirname)
        os.makedirs(dirname)
    return


def get_para_filename(paras_dict, prefix: list, exclude: list = None, max_l=240):
    if exclude is None:
        exclude = LOG_NAME_EXCLUDE
    paras = sorted(paras_dict.items(), key=lambda kv: kv[0])
    log_file_name = prefix + [p[0].replace('_', '')[:3] + str(p[1]) for p in paras if p[0] not in exclude]
    log_file_name = [l.replace(' ', '-').replace('_', '-') for l in log_file_name]
    log_file_name = '_'.join(log_file_name)[:max_l]
    return log_file_name
