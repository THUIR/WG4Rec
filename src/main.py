# coding=utf-8

import argparse
import logging
import sys
import numpy as np
import os
import torch
import datetime
import pickle
import copy
import time
import random

from utils import utils
from utils.global_p import *

# 动态打包__all__里面的所有类
from data_readers import *
from models import *
from data_processors import *
from runners import *


def parse_global_args(parser):
    """
    全局命令行参数
    :param parser:
    :return:
    """
    parser.add_argument('--gpu', type=str, default='0',
                        help='Set CUDA_VISIBLE_DEVICES.')
    parser.add_argument('--verbose', type=int, default=logging.INFO,
                        help='Logging Level, 0, 10, ..., 50.')
    parser.add_argument('--log_file', type=str, default='',
                        help='Logging file path.')
    parser.add_argument('--result_file', type=str, default='',
                        help='Evaluation result file path.')
    parser.add_argument('--random_seed', type=int, default=DEFAULT_SEED,
                        help='Random seed of numpy and pytorch.')
    parser.add_argument('--train', type=int, default=1,
                        help='To train the model or not.')
    parser.add_argument('--debugging', type=int, default=0,
                        help='When debugging, log and model will not override.')
    parser.add_argument('--save_res', type=int, default=1,
                        help='whether save prediction results to file')
    parser.add_argument('--rank', type=int, default=1,
                        help='1=ranking, 0=rating/click')
    parser.add_argument('--regenerate', type=int, default=1,
                        help='Whether to regenerate intermediate files.')
    parser.add_argument('--unlabel_test', type=int, default=0,
                        help='Whether test data has label.')
    return parser


def build_environment(args_dict, dr_name, dp_name, md_name, rn_name):
    if type(args_dict) is str:
        args_dict = eval(args_dict)
    model_name = eval('{0}.{0}'.format(md_name))
    data_reader_name = eval('{0}.{0}'.format(dr_name))
    data_processor_name = eval('{0}.{0}'.format(dp_name))
    runner_name = eval('{0}.{0}'.format(rn_name))
    logging.info('-' * 45 + ' BEGIN: ' + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + ' ' + '-' * 45)
    logging.info(utils.format_arg_str(args_dict))

    logging.info('DataReader: ' + dr_name)
    logging.info('Model: ' + md_name)
    logging.info('Runner: ' + rn_name)
    logging.info('DataProcessor: ' + dp_name)

    # random seed 这里设置使得数据集生成时的行为（采负例...）是可复现的
    torch.backends.cudnn.deterministic = True
    random_seed = args_dict['random_seed']
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)

    # cuda
    os.environ["CUDA_VISIBLE_DEVICES"] = args_dict['gpu']  # default '0'
    logging.info("# cuda devices: %d" % torch.cuda.device_count())

    # create data_reader
    dr_para_dict = utils.get_init_paras_dict(data_reader_name, args_dict)
    logging.info(os.linesep + dr_name + ': ' + utils.format_arg_str(dr_para_dict))
    corpus_path = os.path.join(DATASET_DIR, args_dict['dataset'], DATASET_BF_DIR, '{}.pk'.format(
        utils.get_para_filename(dr_para_dict, prefix=[str(args_dict['rank']) + str(args_dict['drop_neg']), dr_name])
    ))
    if args_dict['regenerate'] or not os.path.exists(corpus_path):
        data_reader = data_reader_name(**dr_para_dict)
        # 如果是top n推荐，只保留正例，负例是训练过程中采样得到，并且将label转换为01二值
        if args_dict['rank'] == 1:
            data_reader.label_01()
            if args_dict['drop_neg'] == 1:
                data_reader.drop_neg()
        logging.info('Saving corpus to {}'.format(corpus_path))
        utils.check_dir_and_mkdir(corpus_path)
        pickle.dump(data_reader, open(corpus_path, 'wb'), protocol=4)
    else:
        logging.info('Loading corpus from {}'.format(corpus_path))
        data_reader = pickle.load(open(corpus_path, 'rb'))

    # create data_processor
    dp_para_dict = utils.get_init_paras_dict(data_processor_name, args_dict)
    dp_para_dict['data_reader'], dp_para_dict['model_name'] = data_reader, model_name
    logging.info(os.linesep + dp_name + ': ' + utils.format_arg_str(dp_para_dict))

    # # prepare train,test,validation samples 需要写在模型产生和训练之前，保证对不同模型相同random seed产生一样的测试负例
    test_data = data_processor_name(
        df=data_reader.test_df, procedure=2, **dp_para_dict)
    validation_data = data_processor_name(
        df=data_reader.validation_df, procedure=1, **dp_para_dict)
    train_data = data_processor_name(
        df=data_reader.train_df, procedure=0, **dp_para_dict)
    if hasattr(validation_data, 'neg_dict'):
        validation_neg_iids = os.path.join(DATASET_DIR, args_dict['dataset'],
                                           '{}.validation_iids.csv'.format(args_dict['dataset']))
        test_neg_iids = os.path.join(DATASET_DIR, args_dict['dataset'],
                                     '{}.test_iids.csv'.format(args_dict['dataset']))
        if os.path.exists(validation_neg_iids):
            validation_data.load_neg_dict(validation_neg_iids)
        else:
            validation_data.save_neg_dict(validation_neg_iids)
        if os.path.exists(test_neg_iids):
            test_data.load_neg_dict(test_neg_iids)
        else:
            test_data.save_neg_dict(test_neg_iids)

    # create model
    # 根据模型需要生成：数据集的特征、特征总共one-hot/multi-hot维度、特征每个field最大值和最小值，
    features, feature_dims, feature_min, feature_max = data_reader.feature_info(model_name=model_name)
    args_dict['feature_num'], args_dict['feature_dims'] = len(features), feature_dims
    args_dict['user_feature_num'] = len([f for f in features if f.startswith('u_')])
    args_dict['item_feature_num'] = len([f for f in features if f.startswith('i_')])
    args_dict['context_feature_num'] = len([f for f in features if f.startswith('c_')])
    data_reader_vars = vars(data_reader)
    for key in data_reader_vars:
        if key not in args_dict:
            args_dict[key] = data_reader_vars[key]
    # print(args.__dict__.keys())
    model_para_dict = utils.get_init_paras_dict(model_name, args_dict)
    logging.info(os.linesep + md_name + ': ' + utils.format_arg_str(model_para_dict))
    model = model_name(**model_para_dict)
    logging.info(model)

    # init model paras
    model.apply(model.init_paras)

    # use gpu
    if torch.cuda.device_count() > 0:
        # model = model.to('cuda:0')
        model = model.cuda()

    # create runner
    runner_para_dict = utils.get_init_paras_dict(runner_name, args_dict)
    logging.info(os.linesep + rn_name + ': ' + utils.format_arg_str(runner_para_dict))
    runner = runner_name(**runner_para_dict)
    return data_reader, train_data, validation_data, test_data, model, runner


def main():
    # init args
    init_parser = argparse.ArgumentParser(description='Initial Args', add_help=False)
    init_parser.add_argument('--data_reader', type=str, default='', help='Choose data_reader')
    init_parser.add_argument('--model_name', type=str, default='BaseModel', help='Choose model to run.')
    init_parser.add_argument('--runner_name', type=str, default='', help='Choose runner')
    init_parser.add_argument('--data_processor', type=str, default='', help='Choose data_processor')
    init_args, init_extras = init_parser.parse_known_args()

    # choose model
    model_name = eval('{0}.{0}'.format(init_args.model_name))

    # choose data_reader
    if init_args.data_reader == '':
        init_args.data_reader = model_name.data_reader
    data_reader_name = eval('{0}.{0}'.format(init_args.data_reader))

    # choose data_processor
    if init_args.data_processor == '':
        init_args.data_processor = model_name.data_processor
    data_processor_name = eval('{0}.{0}'.format(init_args.data_processor))

    # choose runner
    if init_args.runner_name == '':
        init_args.runner_name = model_name.runner
    runner_name = eval('{0}.{0}'.format(init_args.runner_name))

    # cmd line paras
    parser = argparse.ArgumentParser(description='')
    parser = parse_global_args(parser)
    parser = data_reader_name.parse_data_args(parser)
    parser = model_name.parse_model_args(parser, model_name=init_args.model_name)
    parser = runner_name.parse_runner_args(parser)
    parser = data_processor_name.parse_dp_args(parser)

    origin_args, extras = parser.parse_known_args()

    # construct log,model,result filename
    log_file_name = utils.get_para_filename(paras_dict=vars(origin_args), prefix=[
        str(origin_args.rank) + str(origin_args.drop_neg), init_args.model_name,
        origin_args.dataset, str(origin_args.random_seed)
    ])

    # debugging模式不覆盖之前同样参数的log、result和model
    if origin_args.debugging:
        t = time.time()
        origin_args.log_file = '../log/log-{}-{}.txt'.format(init_args.model_name, t)
        origin_args.result_file = '../result/result-{}-{}.pk'.format(init_args.model_name, t)
        origin_args.model_path = '../model/model-{}-{}.pt'.format(init_args.model_name, t)

    if origin_args.log_file == '':
        origin_args.log_file = os.path.join(LOG_DIR, '%s/%s.txt' % (init_args.model_name, log_file_name))
    utils.check_dir_and_mkdir(origin_args.log_file)
    if origin_args.result_file == '':
        origin_args.result_file = os.path.join(RESULT_DIR, '%s/%s.pk' % (init_args.model_name, log_file_name))
    utils.check_dir_and_mkdir(origin_args.result_file)
    if origin_args.model_path == '':
        origin_args.model_path = os.path.join(MODEL_DIR, '%s/%s.pt' % (init_args.model_name, log_file_name))
    utils.check_dir_and_mkdir(origin_args.model_path)

    args = copy.deepcopy(origin_args)

    # logging
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(filename=args.log_file, level=args.verbose)
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

    data_reader, train_data, validation_data, test_data, model, runner = build_environment(
        vars(args), dr_name=init_args.data_reader, dp_name=init_args.data_processor,
        md_name=init_args.model_name, rn_name=init_args.runner_name)

    # training/testing
    # 如果load > 0，表示载入模型继续训练
    if args.load > 0:
        model.load_model()
    # 如果train > 0，表示需要训练，否则直接测试
    if args.train > 0:
        runner.train(model, train_data=train_data, validation_data=validation_data,
                     test_data=None if args.unlabel_test == 1 else test_data)

    train_result = runner.predict(model, train_data)
    validation_result = runner.predict(model, validation_data)
    test_result = runner.predict(model, test_data)

    # save test results
    if args.save_res:
        pickle.dump(train_result, open(args.result_file.replace('.pk', '.train.pk'), 'wb'), protocol=4)
        pickle.dump(validation_result, open(args.result_file.replace('.pk', '.valid.pk'), 'wb'), protocol=4)
        pickle.dump(test_result, open(args.result_file.replace('.pk', '.test.pk'), 'wb'), protocol=4)
        logging.info(os.linesep + 'Save Results to ' + args.result_file)

    # 训练结束后在指定的多个评价指标上进行评测
    all_metrics = ['rmse', 'mae', 'auc', 'f1', 'accuracy', 'precision', 'recall']
    if args.rank == 1:
        all_metrics = ['ndcg@1', 'ndcg@5', 'ndcg@10', 'ndcg@20', 'ndcg@50', 'ndcg@100'] \
                      + ['hit@1', 'hit@5', 'hit@10', 'hit@20', 'hit@50', 'hit@100'] \
                      + ['precision@1', 'precision@5', 'precision@10', 'precision@20', 'precision@50', 'precision@100'] \
                      + ['recall@1', 'recall@5', 'recall@10', 'recall@20', 'recall@50', 'recall@100']
    results = [train_result, validation_result, test_result]
    name_map = ['Train', 'Valid', 'Test']
    datasets = [train_data, validation_data]
    if args.unlabel_test != 1:
        datasets.append(test_data)
    for i, dataset in enumerate(datasets):
        metrics = model.evaluate_method(results[i], datasets[i], metrics=all_metrics, error_skip=True)
        log_info = 'Test After Training on %s: ' % name_map[i]
        log_metrics = ['%s=%s' % (metric, utils.format_metric(metrics[j])) for j, metric in enumerate(all_metrics)]
        log_info += ', '.join(log_metrics)
        logging.info(os.linesep + log_info + os.linesep)

    logging.info('# of params: %d' % model.total_parameters)
    logging.info(vars(origin_args))
    logging.info(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    return


if __name__ == '__main__':
    main()
