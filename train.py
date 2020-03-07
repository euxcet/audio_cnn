import os
import sys
import math
import numpy as np
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as functional
import torch.optim as optim
import torch.utils.data
from dataset import Dataset, Sampler, Collator
from model import Cnn, requires_grad_parameters, loss_bce
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
torch.backends.cudnn.benchmark = True
torch.manual_seed(0)

def load_labels():
    fin = open('class_labels_indices.csv', 'r') 
    lines = fin.readlines()
    labels = []
    for line in lines[1:]:
        segs = line.strip().split(',')
        id = segs[0]
        label = segs[1]
        name = segs[2][1:-1]
        labels.append((label, name))
    return labels

def to_device(data, device):
    if 'float' in str(data.dtype):
        return torch.Tensor(data).to(device)
    elif 'int' in str(data.dtype):
        return torch.LongTensor(data).to(device)
    else:
        return data

def train():
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    sample_rate = 32000
    audio_length = sample_rate * 10
    n_fft = 1024
    hop_length = 320
    n_mels = 64
    fmin = 50
    fmax = 14000
    batch_size = 32
    num_workers = 8
    learning_rate = 1e-3
    labels = load_labels()
    num_class = len(labels)
    print("Number of classes:", num_class)

    model = Cnn(sample_rate=sample_rate, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels, fmin=fmin, fmax=fmax, num_class=num_class)

    train_hdf5_path = 'hdf5/meta/balanced_train.h5'
    train_wav_hdf5_path = 'hdf5/wav/balanced_train.h5'
    test_hdf5_path = 'hdf5/meta/eval.h5'
    test_wav_hdf5_path = 'hdf5/wav/eval.h5'

    num_param = requires_grad_parameters(model)
    print("Number of trainable parameters:", num_param)

    train_dataset = Dataset(train_hdf5_path, train_wav_hdf5_path, audio_length, num_class)
    train_sampler = Sampler(train_hdf5_path, batch_size)
    train_collator = Collator()

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_sampler=train_sampler, collate_fn=train_collator, num_workers=num_workers, pin_memory=True)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-8, weight_decay=0., amsgrad=True)
    
    print('Number of GPU:', torch.cuda.device_count())

    model.to(device)

    iteration = 0

    for batch in train_loader:
        for key in batch.keys():
            batch[key] = to_device(batch[key], device)
        model.train()
        output = model(batch['waveform'])
        target = {'target': batch['target']}
        loss = loss_bce(output, batch)
        loss.backward()
        print('Iteration:', iteration, 'Loss:', loss)
        iteration += 1
        optimizer.step()
        optimizer.zero_grad()
        if iteration == 10:
            break
