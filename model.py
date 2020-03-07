import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import librosa

import numpy as np
import math

from stft import Spectrogram, MelFilterBank

def loss_bce(output, target):
    return F.binary_cross_entropy(output['output'], target['target'])

def requires_grad_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def initialize_uniform(layer):
    nn.init.xavier_uniform_(layer.weight)
    if hasattr(layer, 'bias') and layer.bias is not None:
        layer.bias.data.fill_(0.)

def initialize_ones(layer):
    layer.weight.data.fill_(1.)
    layer.bias.data.fill_(0.)

class ConvBlock3x3(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock3x3, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.initialize()

    def initialize(self):
        initialize_uniform(self.conv1)
        initialize_uniform(self.conv2)
        initialize_ones(self.bn1)
        initialize_ones(self.bn2)

    def forward(self, input, pool_size=(2, 2), pool_filter='avg'):
        x = F.relu_(self.bn1(self.conv1(input)))
        x = F.relu_(self.bn2(self.conv2(x)))
        if pool_filter == 'max':
            return F.max_pool2d(x, kernel_size=pool_size)
        elif pool_filter == 'avg':
            return F.avg_pool2d(x, kernel_size=pool_size)
        else:
            raise Exception('Unknown filter for pooling')

class ConvBlock5x5(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock5x5, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.initialize()

    def initialize(self):
        initialize_uniform(self.conv1)
        initialize_ones(self.bn1)

    def forward(self, input, pool_size=(2, 2), pool_filter='avg'):
        x = F.relu_(self.bn1(self.conv1(input)))
        if pool_filter == 'max':
            return F.max_pool2d(x, kernel_size=pool_size)
        elif pool_filter == 'avg':
            return F.avg_pool2d(x, kernel_size=pool_size)
        else:
            raise Exception('Unknown filter for pooling')

class Cnn(nn.Module):
    def __init__(self, sample_rate, n_fft, hop_length, n_mels, fmin, fmax, num_class):
        super(Cnn, self).__init__()
        # default params
        window = 'hann'
        center = True
        pad_mode = 'reflect'
        ref = 1.0
        amin = 1e-10
        top_db = None

        self.spectrogram = Spectrogram(n_fft=n_fft, hop_length=hop_length, win_length=n_fft, window=window, center=center, pad_mode=pad_mode, requires_grad=False)
        self.melfilterbank = MelFilterBank(sr=sample_rate, n_fft=n_fft, n_mels=n_mels, fmin=fmin, fmax=fmax, ref=ref, amin=amin, top_db=top_db, requires_grad=False)

        self.bn = nn.BatchNorm2d(64) # TODO: 64 -> n_mels ?

        self.conv1 = ConvBlock5x5(in_channels=1, out_channels=64)
        self.conv2 = ConvBlock5x5(in_channels=64, out_channels=128)
        self.conv3 = ConvBlock5x5(in_channels=128, out_channels=256)
        self.conv4 = ConvBlock5x5(in_channels=256, out_channels=512)

        self.fc1 = nn.Linear(512, 512, bias=True)
        self.fc2 = nn.Linear(512, num_class, bias=True)

        self.initialize()

    def initialize(self):
        initialize_uniform(self.fc1)
        initialize_uniform(self.fc2)
        initialize_ones(self.bn)

    def forward(self, input):
        x = self.spectrogram(input)
        x = self.melfilterbank(x)
        x = x.transpose(1, 3)
        x = self.bn(x)
        x = x.transpose(1, 3)

        # TODO: augmentation

        dropout = 0.2

        x = self.conv1(x, pool_size=(2, 2), pool_filter='avg')
        x = F.dropout(x, p=dropout, training=self.training)
        x = self.conv2(x, pool_size=(2, 2), pool_filter='avg')
        x = F.dropout(x, p=dropout, training=self.training)
        x = self.conv3(x, pool_size=(2, 2), pool_filter='avg')
        x = F.dropout(x, p=dropout, training=self.training)
        x = self.conv4(x, pool_size=(2, 2), pool_filter='avg')
        x = F.dropout(x, p=dropout, training=self.training)

        x = torch.mean(x, dim=3)
        (x1, _) = torch.max(x, dim=2)
        x2 = torch.mean(x, dim=2)
        x = x1 + x2
        x = F.dropout(x, p=0.5, training=self.training)
        x= F.relu_(self.fc1(x))
        embedding = F.dropout(x, p=0.5, training=self.training)
        output = torch.sigmoid(self.fc2(x))
        return {
            'output': output,
            'embedding': embedding
        }

