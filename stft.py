import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import librosa

import numpy as np
import math

class DFTAbstract(nn.Module):
    def __init__(self):
        super(DFTAbstract, self).__init__()

    def dft_matrix(self, n):
        col, row = np.meshgrid(np.arange(n), np.arange(n))
        omega = np.exp(-2j * np.pi / n)
        return np.power(omega, col * row)
        
    def idft_matrix(self, n):
        col, row = np.meshgrid(np.arange(n), np.arange(n))
        omega = np.exp(2j * np.pi / n)
        return np.power(omega, col * row)

    def complex_mul(self, x_real, x_imag, y_real, y_imag):
        real = torch.matmul(x_real, y_real) - torch.matmul(x_imag, y_imag)
        imag = torch.matmul(x_real, y_imag) + torch.matmul(x_imag, y_real)
        return real, imag

    def seperate(self, x):
        return torch.Tensor(np.real(x)), torch.Tensor(np.imag(x))

class DFT(DFTAbstract):
    def __init__(self, n, normal):
        super(DFT, self).__init__()
        self.n = n
        self.normal = normal

        self.W = self.dft_matrix
        self.iW = self.idft_matrix

        self.W_real, self.W_imag = self.seperate(self.W)
        self.iW_real, self.iW_imag = self.seperate(self.iW)

    def dft(self, x_real, x_imag):
        real, imag = self.complex_mul(x_real, x_imag, self.W_real, self.W_imag)
        if self.normal:
            real /= math.sqrt(self.n)
            imag /= math.sqrt(self.n)
        return real, imag

    def idft(self, x_real, x_imag):
        real, imag = self.complex_mul(x_real, x_imag, self.iW_real, self.iW_imag)
        if self.normal:
            real /= math.sqrt(self.n)
            imag /= math.sqrt(self.n)
        else:
            real /= self.n

class STFT(DFTAbstract):
    def __init__(self, n_fft=2048, hop_length=None, win_length=None, window='hann', center=True, pad_mode='reflect', requires_grad=False):
        super(STFT, self).__init__()
        self.n_fft = n_fft
        self.center = center
        self.pad_mode = pad_mode
        self.W = self.dft_matrix(n_fft)
        rows = 1 + n_fft // 2
        win_length = n_fft if win_length is None else win_length
        hop_length = int(win_length // 4) if hop_length is None else hop_length
        window = librosa.filters.get_window(window, win_length, fftbins=True)

        self.conv_real = nn.Conv1d(in_channels=1, out_channels=rows, kernel_size=n_fft, stride=hop_length, padding=0, dilation=1, groups=1, bias=False)
        self.conv_imag = nn.Conv1d(in_channels=1, out_channels=rows, kernel_size=n_fft, stride=hop_length, padding=0, dilation=1, groups=1, bias=False)
        
        self.conv_real.weight.data = torch.Tensor(np.real(self.W[:, 0 : rows] * window[:, None]).T)[:, None, :]
        self.conv_imag.weight.data = torch.Tensor(np.imag(self.W[:, 0 : rows] * window[:, None]).T)[:, None, :]

        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, input):
        x = input[:, None, :]
        if self.center:
            x = F.pad(x, pad=(self.n_fft // 2, self.n_fft // 2), mode=self.pad_mode)
        # x (batch_size, channels, length)
        # real imag (batch_size, 1, steps, n_fft / 2 + 1)
        real = self.conv_real(x)[:, None, :, :].transpose(2, 3)
        imag = self.conv_imag(x)[:, None, :, :].transpose(2, 3)
        return real, imag
        
class ISTFT(DFTAbstract):
    def __init__(self, n_fft=2048, hop_length=None, win_length=None, window='hann', center=True, pad_mode='reflect', requires_grad=False):
        super(ISTFT, self).__init__()
        self.n_fft = n_fft
        self.center = center
        self.pad_mode = pad_mode
        self.W = self.idft_matrix(n_fft) / n_fft
        win_length = n_fft if win_length is None else win_length
        hop_length = int(win_length // 4) if hop_length is None else hop_length
        window = librosa.filters.get_window(window, win_length, fftbins=True)

        self.conv_real = nn.Conv1d(in_channels=n_fft, out_channels=n_fft, kernel_size=n_fft, stride=hop_length, padding=0, dilation=1, groups=1, bias=False)
        self.conv_imag = nn.Conv1d(in_channels=n_fft, out_channels=n_fft, kernel_size=n_fft, stride=hop_length, padding=0, dilation=1, groups=1, bias=False)
        
        self.conv_real.weight.data = torch.Tensor(np.real(self.W * window[None, :]).T)[:, :, None]
        self.conv_imag.weight.data = torch.Tensor(np.imag(self.W * window[None, :]).T)[:, :, None]

        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, real, imag, length):
        # todo
        pass

class Spectrogram(nn.Module):
    def __init__(self, n_fft=2048, hop_length=None, win_length=None, window='hann', center=True, pad_mode='reflect', power=2.0, requires_grad=False):
        super(Spectrogram, self).__init__()
        self.power = power
        self.stft = STFT(n_fft=n_fft, hop_length=hop_length, win_length=win_length, window=window, center=center, pad_mode=pad_mode, requires_grad=requires_grad)

    def forward(self, input):
        real, imag = self.stft.forward(input)
        # real, imag (batch_size, 1, steps, n_fft / 2 + 1)
        # spectrogram (batch_size, 1, steps, n_fft / 2 + 1)
        if self.power == 2.0:
            return real ** 2 + imag ** 2
        else:
            return (real ** 2 + imag ** 2) ** (self.power / 2.0)

class MelFilterBank(nn.Module):
    # TODO: why 32000
    def __init__(self, sr=32000, n_fft=2048, n_mels=64, fmin=50, fmax=14000, to_db=True, ref=1.0, amin=1e-10, top_db=80.0, requires_grad=False):
        super(MelFilterBank, self).__init__()
        self.to_db = to_db
        self.ref = ref
        self.amin = amin
        self.top_db = top_db
        self.mel = nn.Parameter(torch.Tensor(librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mels, fmin=fmin, fmax=fmax).T))
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def power_to_db(self, power):
        log_power = 10.0 * (torch.log10(torch.clamp(power, min=self.amin, max=np.inf)) - np.log10(np.maximum(self.amin, self.ref)))
        if self.top_db is None:
            return log_power
        else:
            return torch.clamp(log_power, min=log_power.max().item() - self.top_db, max=np.inf)

    def forward(self, input):
        # input (batch_size, 1, steps, n_fft / 2 + 1)
        # output (batch_size, 1, steps, n_mels)
        if self.to_db:
            return self.power_to_db(torch.matmul(input, self.mel))
        else:
            return torch.matmul(input, self.mel)




if __name__ == '__main__':
    y, sr = librosa.load(librosa.util.example_audio_file())
    stft_ans = librosa.stft(y)
    stft_ans_real = torch.Tensor(np.real(stft_ans))
    stft_ans_imag = torch.Tensor(np.imag(stft_ans))
    stft = STFT()
    real, imag = stft.forward(torch.Tensor(y)[None, :])
    stft_out_real = torch.squeeze(real).transpose(0, 1)
    stft_out_imag = torch.squeeze(imag).transpose(0, 1)
    assert torch.allclose(stft_ans_real, stft_out_real, rtol=1e-2, atol=1e-4)
    assert torch.allclose(stft_ans_imag, stft_out_imag, rtol=1e-2, atol=1e-4)
