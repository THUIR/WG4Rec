# coding=utf-8

import torch


class ConvSeqVec(torch.nn.Module):
    def __init__(self, vector_size, filter_size, filter_num, in_channels=1):
        super(ConvSeqVec, self).__init__()
        self.vector_size = vector_size
        self.filter_size = [filter_size] if type(filter_size) is int else filter_size
        self.filter_num = [filter_num] if type(filter_num) is int else filter_num
        assert len(self.filter_size) == len(self.filter_num)
        self.in_channels = in_channels
        assert self.in_channels > 0
        self._init_weights()

    def _init_weights(self):
        layers = []
        for i, filter_size in enumerate(self.filter_size):
            filter_num = self.filter_num[i]
            filter_conv = torch.nn.Conv2d(
                in_channels=self.in_channels, out_channels=filter_num,
                kernel_size=(filter_size, self.vector_size),
                padding=(int(filter_size / 2), 0))
            layers.append(filter_conv)
        self.conv_layers = torch.nn.ModuleList(layers)

    def forward(self, x, pre_dims=1):
        '''
        :param x: ? * in_channels * seq_length * vector_size
        :param pre_dims: number of dims represented by '?'
        :return:
        '''
        input_size = x.size()
        seq_vectors = x.view([-1, input_size[-3], input_size[-2], input_size[-1]])  # n * c * l * v
        conv_result = []
        for conv_layer in self.conv_layers:
            layer_result = conv_layer(seq_vectors)  # n * kn * l * 1
            layer_result = layer_result.squeeze(dim=-1).transpose(-1, -2)  # n * l * kn
            conv_result.append(layer_result)
        conv_result = torch.cat(conv_result, dim=-1)  # n * l * sum(kn)
        result_size = conv_result.size()
        out_size = [input_size[i] for i in range(pre_dims)] + [result_size[-2], result_size[-1]]
        return conv_result.view(out_size)  # ? * seq_length * sum(filter_num)
