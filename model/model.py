# -*- coding: utf-8 -*-
"""
@author: ByteHong
@organization: SCUT
"""
import torch
import torch.nn as nn

from model.basenet import network_dict
from loss.loss import RCS_loss
from utils import globalvar as gl

class DNA(nn.Module):

    def __init__(self, basenet, n_class, bottleneck_dim, alpha_div=1.0, beta_div=0.0, lambda_div=0.01):
        super(DNA, self).__init__()
        self.basenet = network_dict[basenet]()
        self.basenet_type = basenet
        self._in_features = self.basenet.len_feature()
        
        if self.basenet_type.lower() not in ['alexnet']:
            self.bottleneck = nn.Sequential(
                nn.Linear(self._in_features, bottleneck_dim),
                nn.BatchNorm1d(bottleneck_dim),
                nn.ReLU(inplace=True)
            )
            self.bottleneck[0].weight.data.normal_(0, 0.005)
            self.bottleneck[0].bias.data.fill_(0.1)
            self.fc = nn.Linear(bottleneck_dim, n_class)
        else:
            print('use the shallower net, type:', basenet)
            self.fc = nn.Linear(self._in_features, n_class)

        self.fc.weight.data.normal_(0, 0.01)
        self.fc.bias.data.zero_()
        
        self.alpha=alpha_div
        self.beta=beta_div
        self.lambda_div = lambda_div

    def forward(self, source, target=None, source_label=None, target_label=None, test_upper_bound=False, label_filter=False):
        DEVICE = gl.get_value('DEVICE')
        source_features = self.basenet(source)
        if self.basenet_type.lower() not in ['alexnet']:
            source_features = self.bottleneck(source_features)
        source_output = self.fc(source_features)
        loss = 0
        if self.training == True and target is not None:
            # calculate the probability
            softmax_layer = nn.Softmax(dim=1).to(DEVICE)
            
            target_features = self.basenet(target)
            if self.basenet_type.lower() not in ['alexnet']:
                target_features = self.bottleneck(target_features)
            target_output = self.fc(target_features)
            target_softmax = softmax_layer(target_output)
            target_prob, target_l = torch.max(target_softmax, 1)
            loss = RCS_loss(source_features, source_label, target_features, target_l, DEVICE, lamda=self.lambda_div, alpha=self.alpha)
        # test the upper bound of this method


        return source_output, loss
    
    def get_bottleneck_features(self, inputs):
        features = self.basenet(inputs)
        return self.bottleneck(features)

    def get_fc_features(self, inputs):
        features = self.basenet(inputs)
        features = self.bottleneck(features)
        return self.fc(features)


