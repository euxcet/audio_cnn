import os
import sys
import torch

def load_labels():
    fin = open('class_labels_indices.csv', 'r') 
    lines = fin.readlines()
    labels = []
    lnames = []
    for line in lines[1:]:
        segs = line.strip().split(',')
        id = segs[0]
        label = segs[1]
        name = segs[2][1:-1]
        labels.append(label)
        lnames.append(name)
    return labels, lnames

def to_device(data, device):
    if 'float' in str(data.dtype):
        return torch.Tensor(data).to(device)
    elif 'int' in str(data.dtype):
        return torch.LongTensor(data).to(device)
    else:
        return data
