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
from dataset import *
from utils import *
from model import *
from sklearn import metrics
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
torch.backends.cudnn.benchmark = True
torch.manual_seed(0)

def evalute(model, loader, device):
    output_list = []
    target_list = []
    for n, batch in enumerate(loader):
        for key in batch.keys():
            batch[key] = to_device(batch[key], device)
        with torch.no_grad():
            model.eval()
            output = model(batch['waveform'])
        output_list.append(output['output'].data.cpu().numpy())
        target_list.append(batch['target'].data.cpu().numpy())
    
    output_list = np.concatenate(output_list, axis=0)
    target_list = np.concatenate(target_list, axis=0)
    print(output_list.shape, target_list.shape)
    precision = metrics.average_precision_score(target_list, output_list, average=None)

    # avoid error: only one class present in y_true
    for i in range(len(precision)):
        if math.isnan(precision[i]):
            target_list[0][i] = 1
    auc = metrics.roc_auc_score(target_list, output_list, average=None)

    precision_ = []
    auc_ = []
    for i in range(len(precision)):
        if not math.isnan(precision[i]):
            precision_.append(precision[i])
            auc_.append(auc[i])
    print('MAP:', np.average(precision_), 'AUC:', np.average(auc_))


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
    num_workers = 3
    learning_rate = 1e-3
    labels, _ = load_labels()
    num_class = len(labels)
    print("Number of classes:", num_class)

    model = Cnn_Large(sample_rate=sample_rate, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels, fmin=fmin, fmax=fmax, num_class=num_class)

    train_hdf5_path = 'hdf5/meta/balanced_train.h5'
    train_wav_hdf5_path = 'hdf5/wav/balanced_train.h5'
    test_hdf5_path = 'hdf5/meta/eval.h5'
    test_wav_hdf5_path = 'hdf5/wav/eval.h5'
    checkpoint_dir = './checkpoint/checkpoint_frame_batch_64/'

    num_param = requires_grad_parameters(model)
    print("Number of trainable parameters:", num_param)

    train_dataset = Dataset(train_hdf5_path, train_wav_hdf5_path, audio_length, num_class)
    train_sampler = Sampler(train_hdf5_path, batch_size)
    train_collator = Collator()
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_sampler=train_sampler, collate_fn=train_collator, num_workers=num_workers, pin_memory=True)

    test_dataset = Dataset(test_hdf5_path, test_wav_hdf5_path, audio_length, num_class)
    test_sampler = OrderedSampler(test_hdf5_path, batch_size)
    test_collator = Collator()
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_sampler=test_sampler, collate_fn=test_collator, num_workers=num_workers, pin_memory=True)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-8, weight_decay=0., amsgrad=True)
    
    print('Number of GPU:', torch.cuda.device_count())

    model.to(device)

    iteration = 0

    # load model
    '''
    checkpoint = torch.load('./checkpoint/iter_1.pth')
    iteration = checkpoint['iteration']
    model.load_state_dict(checkpoint['model'])
    train_sampler.load_state_dict(checkpoint['sampler'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    '''


    for batch in train_loader:
        # evalute
        evalute(model, test_loader, device)


        # train
        for key in batch.keys():
            batch[key] = to_device(batch[key], device)
        model.train()
        output = model(batch['waveform'])
        target = {'target': batch['target']}
        loss = loss_bce(output, target)
        loss.backward()
        print('Iteration:', iteration, 'Loss:', loss)
        iteration += 1
        optimizer.step()
        optimizer.zero_grad()
        if iteration % 1000 == 0:
            path = checkpoint_dir + 'iter_' + str(iteration) + '.pth'
            checkpoint = {
                'iteration': iteration,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'sampler': train_sampler.state_dict()
            }
            torch.save(checkpoint, path)
