import os
import numpy as np
import torch
import torch.nn as nn
import torch.functional as F
import cv2
import pyworld as pw
import config
import librosa
import soundfile as sf
import glob
import matplotlib.pyplot as plt
import librosa.display
from net.lstm_net import LSTM_net
from torch.optim import Adam
from jvs_dataset import JVS_dataset
from torch.utils.data.dataset import Subset
from torch.utils.data import DataLoader

dataset_path = 'dataset/jvs_ver1/'

voiced_wav_path = '/nonpara30w/wav24kHz16bit/'
voiced_npy_path = '/nonpara30w/npy/'



def dynamic_range_compression(x, C=1, clip_val=1e-5):
    return np.log(x.clip(clip_val,None))

def dynamic_range_decompression(x, C=1):
    return np.exp(x)

mel_freqs = librosa.mel_frequencies(n_mels=config.n_mels, fmin=config.fmin, fmax=config.fmax, htk=False).reshape(1,-1)
mel_filter = librosa.filters.mel(sr=config.sr, n_fft = config.n_fft, fmin= config.fmin, fmax= config.fmax, n_mels = config.n_mels, htk = False, norm='slaney').T

# デバイスの確認
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

# ネットワーク読み込み
net_f0 = LSTM_net(input_size=80,output_size=1).to(device)
net_f0.load_state_dict(torch.load('model/msp_2_f0_4_bestloss.pth'))
net_f0.eval()


net_msp = LSTM_net(input_size=81,output_size=80).to(device)
net_msp.load_state_dict(torch.load('model/mspmsp_f0_2_msp_1_bestloss.pth'))

net_msp.eval()
with torch.inference_mode():
    for person in os.listdir(dataset_path):
        if os.path.isdir(dataset_path+person):
            print(person+'>>>>>>>>>>>>>>>')
            voiced_out_path = 'dataset/jvs_ver1/' + person + voiced_npy_path

            for i,f in enumerate(os.listdir(dataset_path + person + voiced_wav_path)):
                    # voice
                    file_path = dataset_path + person + voiced_wav_path + f
                    # save
                    save_path = 'result/grifin/msp_f0_' + str(i) + '.wav'
                    
                    # 音声入力
                    wav,sr = librosa.load(file_path,sr=config.sr)
                    D = librosa.stft(y=wav, n_fft = config.n_fft, hop_length = config.hop_length, win_length = config.win_length, pad_mode='reflect').T
                    sp,phase = librosa.magphase(D)
                    
                    # メルスペクトログラム
                    msp = np.matmul(sp,mel_filter)
                    
                    # 対数に変換
                    lmsp = dynamic_range_compression(msp)
                    
                    # errorになるためデータ構造を変える
                    x = lmsp.reshape(1, -1, 80)
                    x = torch.from_numpy(x.astype(np.float32)).clone()
                    x = x.to(device)
                    rf0 = net_f0(x)
                    # 推論したf0mat
                    f0 = rf0.detach().to('cpu').numpy()[0]
                    print('--f0--', f0.shape)
                    # whisperのmspに推論したf0matをのせる
                    t = np.hstack((lmsp[:f0.shape[0]],f0[:msp.shape[0]]))
                    print(t.shape)
                    # mspを推論
                    t = t.reshape(1, -1, 81)
                    t = torch.from_numpy(t.astype(np.float32)).clone()
                    t = t.to(device)
                    rmsp = net_msp(t)
                    rmsp = rmsp.detach().to('cpu').numpy()[0]
                    msp = dynamic_range_decompression(rmsp)
                    print(msp.shape)

                    # メルスペクトルグラムの可視化
                    # fig, ax = plt.subplots(figsize=(8, 6))
                    # mesh = librosa.display.specshow(rmsp.T, sr=config.sr, hop_length = config.hop_length, x_axis="time", y_axis="hz", ax=ax)
                    # fig.colorbar(mesh, ax=ax, format="%+2.f dB")
                    # ax.set_title("mel-spectrogram")
                    # ax.set_xlabel("Time [sec]")
                    # ax.set_ylabel("Frequency [Hz]")
                    # plt.tight_layout()
                    # plt.show()
                    #音声出力
                    wav = librosa.feature.inverse.mel_to_audio(msp.T, sr=config.sr, n_fft=config.n_fft,n_iter=1000, hop_length=config.hop_length,power=1,pad_mode='reflect', win_length=config.win_length)
                    sf.write(save_path,wav,sr)