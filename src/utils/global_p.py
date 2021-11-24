# coding=utf-8

# default paras
DEFAULT_SEED = 2020
MAX_VT_USER = 100000  # leave out by time 时，最大取多少用户数
MAX_VT_USER_R = 0.1  # leave out by time 时，最大取多少用户比例

# Path
DATA_DIR = '../data/'  # 原始数据文件及预处理的数据文件目录
DATASET_DIR = '../dataset/'  # 划分好的数据集目录
DATASET_BF_DIR = 'buffer/'  # 数据集buffer目录，在对应的数据集目录下，可用来存储data_reader对象
MODEL_DIR = '../model/'  # 模型保存路径
LOG_DIR = '../log/'  # 日志输出路径
RESULT_DIR = '../result/'  # 数据集预测结果保存路径
COMMAND_DIR = '../command/'  # run.py所用command文件保存路径
LOG_CSV_DIR = '../log_csv/'  # run.py所用结果csv文件保存路径

LIBREC_DATA_DIR = '../librec/data/'  # librec原始数据文件及预处理的数据文件目录
LIBREC_DATASET_DIR = '../librec/dataset/'  # librec划分好的数据集目录
LIBREC_MODEL_DIR = '../librec/model/'  # librec模型保存路径
LIBREC_LOG_DIR = '../librec/log/'  # librec日志输出路径
LIBREC_RESULT_DIR = '../librec/result/'  # librec数据集预测结果保存路径
LIBREC_COMMAND_DIR = '../librec/command/'  # run_librec.py所用command文件保存路径
LIBREC_LOG_CSV_DIR = '../librec/log_csv/'  # run_librec.py所用结果csv文件保存路径

# Preprocess/DataReader
TRAIN_SUFFIX = '.train.csv'  # 训练集文件后缀
VALIDATION_SUFFIX = '.validation.csv'  # 验证集文件后缀
TEST_SUFFIX = '.test.csv'  # 测试集文件后缀
INFO_SUFFIX = '.info.json'  # 数据集统计信息文件后缀
USER_SUFFIX = '.user.csv'  # 数据集用户特征文件后缀
ITEM_SUFFIX = '.item.csv'  # 数据集物品特征文件后缀
TRAIN_POS_SUFFIX = '.train_pos.csv'  # 训练集用户正向交互按uid合并之后的文件后缀
VALIDATION_POS_SUFFIX = '.validation_pos.csv'  # 验证集用户正向交互按uid合并之后的文件后缀
TEST_POS_SUFFIX = '.test_pos.csv'  # 测试集用户正向交互按uid合并之后的文件后缀
TRAIN_NEG_SUFFIX = '.train_neg.csv'  # 训练集用户负向交互按uid合并之后的文件后缀
VALIDATION_NEG_SUFFIX = '.validation_neg.csv'  # 验证集用户负向交互按uid合并之后的文件后缀
TEST_NEG_SUFFIX = '.test_neg.csv'  # 测试集用户负向交互按uid合并之后的文件后缀

C_HISTORY = 'history'  # 历史记录column名称
C_HISTORY_LENGTH = 'history_length'  # 历史记录长度column名称
C_HISTORY_NEG = 'history_neg'  # 负反馈历史记录column名称
C_HISTORY_NEG_LENGTH = 'history_neg_length'  # 负反馈历史记录长度column名称
C_HISTORY_POS_TAG = 'history_pos_tag'  # 用于记录一个交互列表是正反馈1还是负反馈0

DICT_SUFFIX = '.dict.csv'
C_WORD = 'word'  # 词的column名称
C_WORD_ID = 'word_id'  # 词的column名称
C_TEXT = 'text'
C_TEXT_LENGTH = 'text_length'

# # BatchProcessor/feed_dict
X = 'x'
Y = 'y'
LABEL = 'label'
UID = 'uid'
IID = 'iid'
IIDS = 'iids'
TIME = 'time'  # 时间column名称
RANK = 'rank'
REAL_BATCH_SIZE = 'real_batch_size'
TOTAL_BATCH_SIZE = 'total_batch_size'
TRAIN = 'train'
DROPOUT = 'dropout'
SAMPLE_ID = 'sample_id'  # 在训练（验证、测试）集中，给每个样本编号。这是该column在data dict和feed dict中的名字。

# # out dict
PRE_VALUE = 'pre_value'
PREDICTION = 'prediction'  # 输出预测
CHECK = 'check'  # 检查中间结果
LOSS = 'loss'  # 输出损失
LOSS_L2 = 'loss_l2'  # 输出l2损失
EMBEDDING_L2 = 'embedding_l2'  # 当前batch涉及到的embedding的l2
L2_BATCH = 'l2_batch'  # 当前计算的embedding的l2的batch大小
TARGET_KEYS = 'target_keys'  # out_dict prediction要返回的，提供测试基础或存储的column list

# # sg_news
DICT_POS_SUFFIX = '.dict_pos.csv'
DOC_TEXT_SUFFIX = '.doc_text.csv'
URL_TEXT_SUFFIX = '.url_text.csv'
QUERY_TEXT_SUFFIX = '.query_text.csv'
DICT_WORD2VEC_NET = '.w2v_net.pk'
DICT_WORD_CO_NET = '.co_net.pk'
DICT_WORD_TOPIC_NET = '.topic_net.pk'
DICT_WORD_CF_NET = '.cf_net.pk'
DICT_TOPIC_CF_NET = '.topic_cf_net.pk'

# DOC_TITLE_SUFFIX = '.doc_title.pk'
# DOC_GCN_SUFFIX = '.doc_gcn.{}.pk'
# DOC_GRAPH_SUFFIX = '.doc_graph.{}.pk'
WORD_GRAPH_SUFFIX = '.word_graph.pk'

URL_ID = 'url_id'
C_SENT = 'sent'  # 句子、逻辑表达式column名称
C_SENT_LENGTH = 'sent_length'  # 句子长度的column名称
C_WORD_GRAPH = 'word_graph'  # 句子的词网络graph的column名称
C_GRAPH_LENGTH = 'word_graph_length'  # 句子的词网络的句子长度的column名称
C_GRAPH_SPLIT = 'graph_split'  # 句子的词网络graph的结构信息
C_GCN_GRAPH = 'gcn_graph'  # GCN网络graph的column名称
C_GCN_CONNECT = 'gcn_connect'  # GCN网络graph的连边的column名称

C_WORD_TOPIC = 'word_topic'  # 词相关topic的column的名称
C_TOPIC_CF = 'topic_cf'  # topic CF的column的名称
C_WORD_CF = 'word_cf'  # word CF的column的名称
C_W2V_SIM = 'w2v_sim'  # word2vec词相似词的column的名称
C_WORD_CO = 'word_co'  # 词共现关系的column的名称

NEG_IIDS = 'neg_iids'  # 与正例在同一列表里的负例
