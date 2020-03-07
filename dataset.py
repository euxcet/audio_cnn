import os
import sys
import numpy as np
import h5py

class Dataset(object):
    def __init__(self, meta_hdf5_path, wav_hdf5_path, audio_length, num_class):
        self.audio_length = audio_length
        self.num_class = num_class
        self.wav_hdf5_path = wav_hdf5_path
        print(meta_hdf5_path, wav_hdf5_path)
        with h5py.File(meta_hdf5_path, 'r') as f:
            self.audio_name = list(map(lambda x: x.decode(), f['audio_name']))
            self.hdf5_name = list(map(lambda x: x.decode(), f['hdf5_name']))
            self.index_in_hdf5 = list(f['index_in_hdf5'])
        self.len = len(self.audio_name)
    
    def __len__(self):
        return self.len

    def get_wav(self, index):
        with h5py.File(self.wav_hdf5_path, 'r') as f:
            return (f['waveform'][index] / 32767.).astype(np.float32), f['target'][index].astype(np.float32)

    def __getitem__(self, index):
        if not (index >= 0 and index < self.len) or self.audio_name[index] == '':
            return {
                'audio_name': None,
                'waveform': np.zeros((self.audio_length,), dtype=np.float32),
                'target': np.zeros((self.num_class,), dtype=np.float32)
            }
        else:
            wav, target = self.get_wav(self.index_in_hdf5[index])
            return {
                'audio_name': self.audio_name[index],
                'waveform': wav,
                'target': target
            }


class Sampler(object):
    def __init__(self, meta_hdf5_path, batch_size):
        self.meta_hdf5_path = meta_hdf5_path
        self.batch_size = batch_size
        with h5py.File(meta_hdf5_path, 'r') as f:
            self.target = f['target'][:].astype(np.float32)
        self.num_sample = self.target.shape[0]
        self.num_class = self.target.shape[1]
        self.index_in_class = [np.where(self.target[:, i] == 1)[0] for i in range(self.num_class)]
        self.current = [0 for i in range(self.num_class)]
        self.len_index_in_class = [len(self.index_in_class[i]) for i in range(self.num_class)]
        self.random_state = np.random.RandomState(0)

    def choose_class(self):
        class_id = np.arange(self.num_class)
        while True:
            self.random_state.shuffle(class_id)
            for id in class_id:
                yield id

    def __len__(self):
        return 0

    def __iter__(self):
        index = []
        for which_class in self.choose_class():
            if self.len_index_in_class[which_class] > 0:
                index.append(self.index_in_class[which_class][self.current[which_class]])
                self.current[which_class] += 1
                if self.current[which_class] == self.len_index_in_class[which_class]:
                    self.current[which_class] = 0
                if len(index) == self.batch_size:
                    yield index
                    index = []

class Collator(object):
    def __init__(self):
        self.random_state = np.random.RandomState(0)

    def __call__(self, data_list):
        return {
            'audio_name': np.array([data['audio_name'] for data in data_list]),
            'waveform': np.array([data['waveform'] for data in data_list]),
            'target': np.array([data['target'] for data in data_list])
        }