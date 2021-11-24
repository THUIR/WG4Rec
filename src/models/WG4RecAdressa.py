# coding=utf-8

import torch
import torch.nn.functional as F
from models.RecModel import RecModel
from utils import utils, components
from utils.global_p import *


class WG4RecAdressa(RecModel):
    data_reader = 'WG4RecDR'  # 默认data_reader
    data_processor = 'AdressaDP'  # 默认data_processor

    @staticmethod
    def parse_model_args(parser, model_name='WG4RecAdressa'):
        parser.add_argument('--w2v_size', type=int, default=64,
                            help='Size of word vectors.')
        parser.add_argument('--att_size', type=int, default=32,
                            help='Size of query vectors.')
        parser.add_argument('--cs_ratio', type=float, default=0.1,
                            help='Cold-Sampling ratio of each batch.')
        parser.add_argument('--cf', type=int, default=1,
                            help='Whether add CF part or not.')
        parser.add_argument('--layer_norm', type=int, default=1,
                            help='Whether add layer norm to ui vectors.')
        parser.add_argument('--item_query', type=int, default=1,
                            help='Whether use target item as history query.')
        parser.add_argument('--his_rnn', type=int, default=1,
                            help='Whether use rnn to model user history.')
        parser.add_argument('--loss_f', type=str, default='cr',
                            help='type of loss function')
        return RecModel.parse_model_args(parser, model_name)

    def __init__(self, dictionary_size, w2v_size, att_size, cf, cs_ratio, layer_norm, item_query, his_rnn,
                 loss_f,
                 *args, **kwargs):
        self.dictionary_size = dictionary_size
        self.w2v_size = w2v_size
        self.att_size = att_size
        self.cf = cf
        self.cs_ratio = cs_ratio
        self.layer_norm = layer_norm
        self.item_query = item_query
        self.his_rnn = his_rnn
        self.loss_f = loss_f
        RecModel.__init__(self, *args, **kwargs)

    def _init_weights(self):
        assert self.u_vector_size == self.i_vector_size
        self.ui_vector_size = self.u_vector_size
        if self.cf == 1:
            self.uid_embeddings = torch.nn.Embedding(self.user_num, self.ui_vector_size)
            self.iid_embeddings = torch.nn.Embedding(self.item_num, self.ui_vector_size)
        self.word_embeddings = torch.nn.Embedding(self.dictionary_size, self.w2v_size)
        self.l2_embeddings = ['uid_embeddings', 'iid_embeddings', 'word_embeddings']

        self.graph_att_q = torch.nn.Linear(self.w2v_size, self.att_size)
        self.graph_att_k = torch.nn.Linear(self.w2v_size, self.att_size)

        self.word_g_trans = torch.nn.Linear(self.w2v_size * 2, self.w2v_size, bias=False)

        self.glayer_att_q = torch.nn.Linear(self.w2v_size, self.att_size)
        self.glayer_att_k = torch.nn.Linear(self.w2v_size, self.att_size)

        self.word_att_q = torch.nn.Linear(self.w2v_size, self.att_size)
        self.word_att_k = torch.nn.Linear(self.w2v_size, self.att_size)

        self.word2doc = torch.nn.Linear(self.w2v_size, self.ui_vector_size)

        self.doc_att_query = torch.nn.Linear(self.ui_vector_size, self.att_size)
        self.doc_att_key = torch.nn.Linear(self.ui_vector_size, self.att_size)

        if self.his_rnn == 1:
            self.rnn = torch.nn.GRU(
                input_size=self.ui_vector_size, hidden_size=self.ui_vector_size, batch_first=True,
                num_layers=1, dropout=0.5)

        self.cf_att_query = torch.nn.Linear(self.att_size, 1, bias=False)
        self.cf_att_key = torch.nn.Linear(self.ui_vector_size, self.att_size)

        self.act = torch.nn.LeakyReLU()
        # self.act_v = torch.nn.Tanh()
        # self.act_v = torch.nn.ReLU()
        self.act_v = torch.nn.LeakyReLU()
        self.ui_layer_norm = torch.nn.LayerNorm(self.ui_vector_size)

    def cf_cb_attention(self, cf_vectors, cb_vectors):
        vectors = torch.stack([cf_vectors, cb_vectors], dim=1)  # B * 2 * fn
        att_key = self.act_v(self.cf_att_key(vectors))  # B * 2 * q
        att_v = self.cf_att_query(att_key)  # B * 2 * 1
        att_exp = (att_v - att_v.max(dim=-2, keepdim=True)[0]).exp()  # B * 2 * 1
        att_w = att_exp / att_exp.sum(dim=-2, keepdim=True)  # B * 2 * 1
        vectors = (vectors * att_w).sum(dim=-2)  # B * fn
        return vectors

    def word_graph_agg(self, word_vectors, word_valid, split):
        sections = [1, sum(split[1:])]
        self_vectors, nb_vectors = word_vectors.split(sections, dim=-2)  # b * H+s * L * G?-1 * wv
        self_valid, nb_valid = word_valid.split(sections, dim=-2)  # b * H+s * L * G?-1 * 1

        graph_att_q = self.act_v(
            self.graph_att_q(nb_vectors).sum(dim=-2, keepdim=True))  # b * H+s * L * 1 * att
        graph_att_k = self.act_v(self.graph_att_k(nb_vectors))  # b * H+s * L * G?-1 * att
        nb_vectors = components.qk_attention(
            query=graph_att_q, key=graph_att_k, value=nb_vectors,
            valid=nb_valid.squeeze(dim=-1))  # b * H+s * L * wv

        self_vectors = self_vectors.squeeze(dim=-2)  # b * H+s * L * wv
        nb_vectors = self.act_v(self.word_g_trans(
            torch.cat([self_vectors, nb_vectors], dim=-1)))  # b * H+s * L * wv
        return self_vectors, nb_vectors

    def predict(self, feed_dict):
        check_list, embedding_l2 = [], []
        real_batch_size = feed_dict[REAL_BATCH_SIZE]
        total_batch_size = feed_dict[TOTAL_BATCH_SIZE]
        sample_n = int(total_batch_size / real_batch_size)
        u_ids = feed_dict[UID][:total_batch_size]  # B
        i_ids = feed_dict[IID][:total_batch_size]  # B
        doc = feed_dict[C_WORD_GRAPH]  # B * H+1 * L * G?
        # doc = feed_dict[C_SENT]  # B * H+1 * L
        graph_split = feed_dict[C_GRAPH_SPLIT]

        # # remove duplicate user calculation
        target_doc = doc[range(total_batch_size), -1]  # B * L * G?
        target_doc = torch.cat(target_doc.unsqueeze(dim=1).split(real_batch_size, dim=0), dim=1)  # b * s * L * G?
        doc = torch.cat([doc[:real_batch_size, :doc.size()[1] - 1], target_doc], dim=1)  # b * H+s * L * G?

        # # Word vectors
        word_valid = doc.gt(0).float().unsqueeze(dim=-1)  # b * H+s * L * G? * 1
        word_vectors = self.word_embeddings(doc) * word_valid  # b * H+s * L * G? * wv
        # word_vectors = F.normalize(self.word_embeddings(doc), p=2, dim=-1) * word_valid  # b * H+s * L * G? * wv
        word_vectors = torch.nn.Dropout(p=feed_dict[DROPOUT])(word_vectors)  # b * H+s * L * wv
        embedding_l2.extend([word_vectors])

        # # Word Graph
        layers = [word_vectors]  # b * H+s * L * G? * wv
        for split in graph_split:
            sections = [1, sum(split[1:])]
            self_layers = []
            for layer in layers[:-1]:
                self_layers.append(layer.split(sections, dim=-2)[0].squeeze(dim=-2))  # b * H+s * L * wv
            self_vectors, agg_vectors = self.word_graph_agg(layers[-1], word_valid=word_valid, split=split)
            self_vectors = torch.nn.Dropout(p=feed_dict[DROPOUT])(self_vectors)
            agg_vectors = torch.nn.Dropout(p=feed_dict[DROPOUT])(agg_vectors)
            word_valid = word_valid.sum(dim=-2).gt(0).float()  # b * H+s * L * 1
            self_layers.extend([self_vectors, agg_vectors * word_valid])
            layers = self_layers
        word_vectors = torch.cat([layer.unsqueeze(dim=-2) for layer in layers], dim=-2)  # b * H+s * L * gl * wv
        glayer_att_q = self.act_v(
            self.glayer_att_q(word_vectors).sum(dim=-2, keepdim=True))  # b * H+s * L * 1 * att
        glayer_att_k = self.act_v(self.glayer_att_k(word_vectors))  # b * H+s * L * gl * att
        word_vectors = components.qk_attention(
            query=glayer_att_q, key=glayer_att_k, value=word_vectors) * word_valid  # b * H+s * L * wv
        word_vectors = torch.nn.Dropout(p=feed_dict[DROPOUT])(word_vectors)  # b * H+s * L * wv

        # # User Item CF vectors
        cf_user_vectors, cf_item_vectors = None, None
        if self.cf == 1:
            cf_user_vectors = self.uid_embeddings(u_ids)  # B * uv
            cf_item_vectors = self.iid_embeddings(i_ids)  # B * iv

            # # Cold Sampling
            if feed_dict[TRAIN] and 1 > self.cs_ratio > 0:
                cf_user_vectors = components.cold_sampling(cf_user_vectors, cs_ratio=self.cs_ratio)
                cf_item_vectors = components.cold_sampling(cf_item_vectors, cs_ratio=self.cs_ratio)
            cf_user_vectors = self.ui_layer_norm(cf_user_vectors) if self.layer_norm == 1 else cf_user_vectors
            cf_item_vectors = self.ui_layer_norm(cf_item_vectors) if self.layer_norm == 1 else cf_item_vectors
            embedding_l2.extend([cf_user_vectors, cf_item_vectors])

        # # Item CB part
        word_att_q = self.act_v(self.word_att_q(word_vectors).sum(dim=-2, keepdim=True))  # b * H+s * 1 * att
        word_att_k = self.act_v(self.word_att_k(word_vectors))  # b * H+s * L * att
        doc_vectors = components.qk_attention(
            query=word_att_q, key=word_att_k, value=word_vectors, valid=word_valid.squeeze(dim=-1))  # b * H+s * L * wv
        doc_vectors = self.act_v(self.word2doc(doc_vectors))  # b * H+s * L * iv
        doc_vectors = torch.nn.Dropout(p=feed_dict[DROPOUT])(doc_vectors)  # b * H+s * L * iv

        cb_item_vectors = doc_vectors[:, -sample_n:, :]  # b * s * iv
        cb_item_vectors = torch.cat(cb_item_vectors.split(1, dim=1), dim=0).squeeze(dim=1)  # B * iv
        cb_item_vectors = self.ui_layer_norm(cb_item_vectors) if self.layer_norm == 1 else cb_item_vectors

        # # Item CF+CB
        item_vectors = cb_item_vectors
        if self.cf == 1:
            item_vectors = self.cf_cb_attention(cf_item_vectors, cb_item_vectors)  # B * iv
            # item_vectors = self.ui_layer_norm(item_vectors) if self.layer_norm == 1 else item_vectors

        # # User CB part
        doc_valid = word_valid.sum(dim=-2).gt(0).float()  # b * H+s * 1
        doc_vectors = doc_vectors[:, :-sample_n, :]  # b * H * iv
        doc_valid = doc_valid[:, :-sample_n, :]  # b * H * 1

        # # History RNN
        his_vectors = doc_vectors
        if self.his_rnn == 1:
            output, hidden = components.seq_rnn(seq_vectors=doc_vectors, valid=doc_valid.squeeze(dim=-1),
                                                rnn=self.rnn, lstm=False, init=None)
            his_vectors = output  # b * H * iv

        # # History Attention
        if self.item_query == 1:
            # # ->total batch size
            his_vectors = torch.cat([his_vectors] * sample_n, dim=0)  # B * H * iv
            doc_valid = torch.cat([doc_valid] * sample_n, dim=0)  # B * H * 1
            doc_query = self.act_v(self.doc_att_query(item_vectors)).unsqueeze(dim=1)  # B * 1 * att
            # doc_query = self.act_v(self.doc_att_query(cb_item_vectors)).unsqueeze(dim=1)  # B * 1 * att
        else:
            doc_query = self.act_v(self.doc_att_query(his_vectors)).sum(dim=-2, keepdim=True)  # b * 1 * att
        doc_key = self.act_v(self.doc_att_key(his_vectors))  # B/b * H * att
        cb_user_vectors = components.qk_attention(
            query=doc_query, key=doc_key, value=his_vectors, valid=doc_valid.squeeze(dim=-1))  # B/b * iv
        cb_user_vectors = self.ui_layer_norm(cb_user_vectors) if self.layer_norm == 1 else cb_user_vectors
        cb_user_vectors = torch.nn.Dropout(p=feed_dict[DROPOUT])(cb_user_vectors)  # B/b * iv

        # # ->total batch size
        if self.item_query != 1:
            cb_user_vectors = torch.cat([cb_user_vectors] * sample_n, dim=0)  # B * iv

        # # User CF+CB
        user_vectors = cb_user_vectors
        if self.cf == 1:
            user_vectors = self.cf_cb_attention(cf_user_vectors, cb_user_vectors)  # B * uv
            # user_vectors = self.ui_layer_norm(user_vectors) if self.layer_norm == 1 else user_vectors

        # Predict
        prediction = (item_vectors * user_vectors).sum(dim=-1).flatten()
        if feed_dict[RANK] == 0 or self.loss_f == 'cr':
            prediction = prediction.sigmoid()
        # check_list.append(('prediction', prediction))
        out_dict = {PREDICTION: prediction, CHECK: check_list, EMBEDDING_L2: embedding_l2,
                    REAL_BATCH_SIZE: feed_dict[REAL_BATCH_SIZE], TOTAL_BATCH_SIZE: feed_dict[TOTAL_BATCH_SIZE]}

        if self.cf == 1:
            cf_prediction = (cf_item_vectors * cf_user_vectors).sum(dim=-1).flatten()
            cb_prediction = (cb_item_vectors * cb_user_vectors).sum(dim=-1).flatten()
            if feed_dict[RANK] == 0 or self.loss_f == 'cr':
                cf_prediction = cf_prediction.sigmoid()
                cb_prediction = cb_prediction.sigmoid()
            # check_list.append(('cf_prediction', cf_prediction))
            # check_list.append(('cb_prediction', cb_prediction))
            out_dict['cf_prediction'] = cf_prediction
            out_dict['cb_prediction'] = cb_prediction
        return out_dict

    def forward(self, feed_dict):
        out_dict = self.predict(feed_dict)
        label = feed_dict[Y].float().flatten()
        if self.cf == 0:
            if feed_dict[RANK] == 1 and self.loss_f == 'bpr':
                # 计算topn推荐的loss，batch前一半是正例，后一半是负例
                loss = components.rank_loss(
                    prediction=out_dict[PREDICTION], label=label,
                    real_batch_size=feed_dict[REAL_BATCH_SIZE], loss_sum=self.loss_sum)
            elif feed_dict[RANK] == 1 and self.loss_f == 'cr':
                label = torch.cat([label, 1 - label])
                loss = torch.nn.BCELoss(reduction='sum' if self.loss_sum == 1 else 'mean')(out_dict[PREDICTION], label)
            else:
                if feed_dict[RANK] == 1:
                    label = torch.cat([label, 1 - label])
                # 计算rating/clicking预测的loss，默认使用mse
                loss = torch.nn.MSELoss(reduction='sum' if self.loss_sum == 1 else 'mean')(out_dict[PREDICTION], label)
            out_dict[LOSS] = loss
            out_dict[LOSS_L2] = self.l2(out_dict)
            return out_dict
        if feed_dict[RANK] == 1 and self.loss_f == 'bpr':
            loss = components.rank_loss(
                prediction=out_dict[PREDICTION], label=label,
                real_batch_size=feed_dict[REAL_BATCH_SIZE], loss_sum=self.loss_sum)
            cf_loss = components.rank_loss(
                prediction=out_dict['cf_prediction'], label=label,
                real_batch_size=feed_dict[REAL_BATCH_SIZE], loss_sum=self.loss_sum)
            cb_loss = components.rank_loss(
                prediction=out_dict['cb_prediction'], label=label,
                real_batch_size=feed_dict[REAL_BATCH_SIZE], loss_sum=self.loss_sum)
        elif feed_dict[RANK] == 1 and self.loss_f == 'cr':
            label = torch.cat([label, 1 - label])
            reduction = 'sum' if self.loss_sum == 1 else 'mean'
            loss = torch.nn.BCELoss(reduction=reduction)(out_dict[PREDICTION], label)
            cf_loss = torch.nn.BCELoss(reduction=reduction)(out_dict['cf_prediction'], label)
            cb_loss = torch.nn.BCELoss(reduction=reduction)(out_dict['cb_prediction'], label)
        else:
            if feed_dict[RANK] == 1:
                label = torch.cat([label, 1 - label])
            reduction = 'sum' if self.loss_sum == 1 else 'mean'
            loss = torch.nn.MSELoss(reduction=reduction)(out_dict[PREDICTION], label)
            cf_loss = torch.nn.MSELoss(reduction=reduction)(out_dict['cf_prediction'], label)
            cb_loss = torch.nn.MSELoss(reduction=reduction)(out_dict['cb_prediction'], label)
        out_dict[LOSS] = loss + cf_loss + cb_loss
        out_dict[LOSS_L2] = self.l2(out_dict)
        return out_dict
