# %%
import os
import sys
import random
import glob
import pickle

from tqdm import tqdm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mne
from pymatreader import read_mat

from eeg_utils import *


# %%
def make_mne_data(data):
    # MNEのチャンネル情報の設定
    ch_names = [c.replace(' ', '') for c in data['ch_labels']]  # チャンネル名を取得
    ch_types = ['eeg'] * len(ch_names)  # チャンネルタイプ（全てEEGと仮定）

    # チャンネル情報を組み立てる
    info = mne.create_info(ch_names=ch_names, sfreq=500, ch_types=ch_types) # type: ignore

    # RawArrayオブジェクトの作成
    raw = mne.io.RawArray(data['data']*1e-6, info) # Vに変換
    raw.set_montage(mne.channels.make_standard_montage('standard_1020'))

    return raw

# %%
def make_mne_events(data, raw):
    event_type = data["event"]["type"].astype(int)
    event_index = raw.time_as_index(data["event"]["init_time"])
    pre_event = np.zeros(len(event_type),dtype=int)
    events = np.column_stack((event_index,pre_event,event_type))
    return events

# %%
def make_mne_epoch(data):
    event_dict = {
        'led/front': 11,
        'led/back': 12,
        'led/pump': 13,
        'laser/front': 21,
        'laser/back': 22,
        'laser/pump': 23
    }
    raw = make_mne_data(data)
    # raw.set_eeg_reference('average', projection=True)
    # raw.apply_proj()

    events = make_mne_events(data,raw)
    epochs = mne.Epochs(raw, events, tmin=0.1, tmax=0.8, event_id=event_dict, baseline=None, preload=True) #0.1 + 0.2-0.7 +0.1
    return epochs

# %%
sample = read_mat("train/subject0/train1.mat")
epochs = make_mne_epoch(sample)

data,_ = load_train(0,0,None,None)
std = data.std(axis=(0,1),keepdims=True)
std[std==0]=1
data /= std
data = data.clip(-4,4)

res=32
for epoch in data.transpose((0,2,1)):
    images=[]
    times = np.arange(epoch.shape[1])/epochs.info['sfreq']
    evoked = mne.EvokedArray(epoch, epochs.info, nave=1)
    fig = evoked.plot_topomap(times=times, show=False, contours=0, sensors=False, outlines=None, res=res)

    # 各サブプロットのAxesImageオブジェクトからデータを取得
    for ax in fig.axes:
        for im in ax.get_images():
            topo = im.get_array()
            images.append(topo.filled(0))
    plt.close(fig)  # 図を表示せずに保存
    images=np.array(images,dtype=np.float32)

    break

vmin,vmax=images.min(),images.max()
fig, axes = plt.subplots(5, 5, figsize=(12, 12))
for idx, image in enumerate(images[25:50]):
    ax = axes[idx // 5, idx % 5]  # 行と列を計算して配置
    ax.imshow(image, origin='lower',vmin=vmin,vmax=vmax)  # 画像を表示

    ax.axis('off')  # 軸を非表示にする
plt.tight_layout()
plt.show()


# %%
mne.set_log_level('ERROR')

train_data,_ = prepare_all()

res=32
all_data={}
for sub in range(5):
    subname=f"{sub:04d}"
    all_data[subname]=[]
    for i in range(3):
        print(subname,i)
        data = train_data[subname][i].copy()
        std = data.std(axis=(0,1))
        std[std==0]=1
        data /= std

        trial=[]
        
        for epoch in tqdm(data.transpose((0,2,1))):
            images=[]
            times = np.arange(epoch.shape[1])/epochs.info['sfreq']
            evoked = mne.EvokedArray(epoch, epochs.info, nave=1)
            fig = evoked.plot_topomap(times=times, show=False, contours=0, sensors=False, outlines=None, res=res)

            # 各サブプロットのAxesImageオブジェクトからデータを取得
            for ax in fig.axes:
                for im in ax.get_images():
                    topo = im.get_array()
                    images.append(topo.filled(0))
            plt.close(fig)  # 図を表示せずに保存

            trial.append(images)

        trial=np.array(trial,dtype=np.float32)
        all_data[subname].append(trial)

with open('topo32_2.pkl', 'wb') as f:
    pickle.dump(all_data, f)

test_data = prepare_testdata()
res=32
all_data={}
for sub in range(5):
    subname=f"{sub:04d}"
    all_data[subname]=[]
    print(subname)

    data = test_data[subname].copy()
    std = data.std(axis=(0,1))
    std[std==0]=1
    data /= std

    trial=[]
    
    for epoch in tqdm(data.transpose((0,2,1))):
        images=[]
        times = np.arange(epoch.shape[1])/epochs.info['sfreq']
        evoked = mne.EvokedArray(epoch, epochs.info, nave=1)
        fig = evoked.plot_topomap(times=times, show=False, contours=0, sensors=False, outlines=None, res=res)

        # 各サブプロットのAxesImageオブジェクトからデータを取得
        for ax in fig.axes:
            for im in ax.get_images():
                topo = im.get_array()
                images.append(topo.filled(0))
        plt.close(fig)  # 図を表示せずに保存

        trial.append(images)

    trial=np.array(trial,dtype=np.float32)
    all_data[subname].append(trial)

with open('topo32_2_test.pkl', 'wb') as f:
    pickle.dump(all_data, f)


