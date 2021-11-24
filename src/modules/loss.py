# coding=utf-8
import torch
import torch.nn.functional as F


class SoftmaxRankLoss(torch.nn.Module):
    def forward(self, prediction, label, real_batch_size, loss_sum):
        prediction = prediction.view([-1, real_batch_size]).transpose(0, 1)  # b * (1+s)
        pre_softmax = (prediction - prediction.max(dim=1, keepdim=True)[0]).softmax(dim=1)  # b * (1+s)
        target_pre = pre_softmax[:, 0]  # b
        loss = -(target_pre * label + (1 - label) * (1 - target_pre)).log()  # b
        if loss_sum == 1:
            return loss.sum()
        return loss.mean()


class BPRRankLoss(torch.nn.Module):
    def forward(self, prediction, label, real_batch_size, loss_sum):
        '''
        计算rank loss，类似BPR-max，参考论文:
        @inproceedings{hidasi2018recurrent,
          title={Recurrent neural networks with top-k gains for session-based recommendations},
          author={Hidasi, Bal{\'a}zs and Karatzoglou, Alexandros},
          booktitle={Proceedings of the 27th ACM International Conference on Information and Knowledge Management},
          pages={843--852},
          year={2018},
          organization={ACM}
        }
        :param prediction: 预测值 [None]
        :param label: 标签 [None]
        :param real_batch_size: 观测值batch大小，不包括sample
        :param loss_sum: 1=sum, other= mean
        :return:
        '''
        pos_neg_tag = (label - 0.5) * 2
        observed, sample = prediction[:real_batch_size], prediction[real_batch_size:]
        # sample = sample.view([-1, real_batch_size]).mean(dim=0)
        sample = sample.view([-1, real_batch_size]) * pos_neg_tag.view([1, real_batch_size])  # ? * b
        sample_softmax = (sample - sample.max(dim=0)[0]).softmax(dim=0)  # ? * b
        sample = (sample * sample_softmax).sum(dim=0)  # b
        # loss = -(pos_neg_tag * (observed - sample)).sigmoid().log()
        loss = F.softplus(-pos_neg_tag * (observed - sample))  # b
        if loss_sum == 1:
            return loss.sum()
        return loss.mean()
