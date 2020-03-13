import os
import sys
import numpy as np
import librosa
import torch
import time
from utils import *
from model import *
import matplotlib.pyplot as plt
os.environ['CUDA_VISIBLE_DEVICES'] = '3'

class Inference:
    def __init__(self, checkpoint_path):
        self.sample_rate = 32000
        n_fft = 1024
        hop_length = 320
        n_mels = 64
        fmin = 50
        fmax = 14000
        _, self.labels = load_labels()
        num_class = len(self.labels)
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.frames_per_second = self.sample_rate // hop_length

#self.model = Cnn_Large(sample_rate=self.sample_rate, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels, fmin=fmin, fmax=fmax, num_class=num_class)
        self.model = Cnn(sample_rate=self.sample_rate, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels, fmin=fmin, fmax=fmax, num_class=num_class)
        self.model.load_state_dict(checkpoint['model'])
        self.model = torch.nn.DataParallel(self.model)
        self.model.to(self.device)
        self.model.eval()

    def inference(self, audio_path, data = None):
        wave = None
        if audio_path is not None:
            (wave, _) = librosa.core.load(audio_path, sr=self.sample_rate, mono=True)
            wave = to_device(wave[None, :], self.device)
        else:
            wave = to_device(data[None, :], self.device)

        st = time.time()
        result = self.model(wave)
        output = result['output'].data.cpu().numpy()[0]
        index = np.argsort(output)[::-1]
        for i in range(10):
            print(self.labels[index[i]], output[index[i]])
        print()

        '''
        frame = result['frame'].data.cpu().numpy()[0]
        index = np.argsort(np.max(frame, axis=0))[::-1]
        stft = librosa.core.stft(y=wave[0].data.cpu().numpy(), n_fft=n_fft, hop_length=hop_length, window='hann', center=True)
        num_frame = stft.shape[-1]
        fig, axs = plt.subplots(2, 1, sharex=True, figsize=(10, 4))
        axs[0].matshow(np.log(np.abs(stft)), origin='lower', aspect='auto', cmap='jet')
        axs[0].set_ylabel('Frequency bins')
        axs[0].set_title('Log spectrogram')
        axs[1].matshow(frame[:, index[0: 10]].T, origin='upper', aspect='auto', cmap='jet', vmin=0, vmax=1)
        axs[1].xaxis.set_ticks(np.arange(0, num_frame, frames_per_second))
        axs[1].xaxis.set_ticklabels(np.arange(0, num_frame / frames_per_second))
        axs[1].yaxis.set_ticks(np.arange(0, 10))
        axs[1].yaxis.set_ticklabels(np.array(labels)[index[0:10]])
        axs[1].yaxis.grid(color='k', linestyle='solid', linewidth=0.3, alpha=0.3)
        axs[1].set_xlabel('Seconds')
        axs[1].xaxis.set_ticks_position('bottom')
        plt.tight_layout()
        plt.savefig('1.png')
        '''


if __name__ == '__main__':
    infer = Inference('./checkpoint/checkpoint_frame_batch_64/iter_5000.pth')
    infer.inference('./examples/3.wav')

