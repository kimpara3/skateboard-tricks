import os
import numpy as np
import scipy.signal as signal
from pymatreader import read_mat

event2str={
    11:'frontside_kickturn',
    12:'backside_kickturn',
    13:'pumping',
    21:'frontside_kickturn',
    22:'backside_kickturn',
    23:'pumping'
}
event2str3={
    1:'frontside_kickturn',
    2:'backside_kickturn',
    3:'pumping',
}

def prepare_all(lowcut=None, highcut=None):
    '''
    return (X,y)
    X["sub0-4"][0-2][N,T,C]
    y["sub0-4"][0-2]
    Xはトリガーの0.15秒から0.75秒
    '''
    X = {}
    y = {}
    for sub in range(5):
        signals=[]
        targets=[]
        for trial in range(0,3):
            signal, target = load_train(subject=sub,trial=trial,lowcut=lowcut,highcut=highcut)
            signals.append(signal)
            targets.append(target)       

        X[f"{sub:04d}"] = signals
        y[f"{sub:04d}"] = targets

    X["0002"][0][:,:,52] /= 100
    return X,y

def prepare_testdata(lowcut=None, highcut=None):
    '''
    return X["sub0-4][N,T,C]
    Xはトリガーの0.2秒から0.7秒
    '''
    X = {}
    for sub in range(5):
        X[f"{sub:04d}"] = load_test(subject=sub,lowcut=lowcut,highcut=highcut)

    X["0002"][:,:,52] /= 28

    return X

def load_train(subject:int,trial:int, lowcut, highcut):
    '''トリガーの0.15秒から0.75秒までを取得する'''
    filename = f"./train/subject{subject}/train{trial+1}.mat"
    data_dict = read_mat(filename)
    
    y=data_dict["event"]["type"].astype(int)
    ch_labels=data_dict["ch_labels"]
    init_time=(data_dict["event"]["init_time"]*1000).astype(int)  #msec
    X=[]
    for t in init_time:
        start_index = (t+150)//2
        end_index = (t+750)//2
        X.append(data_dict["data"][:,start_index:end_index]*1e-6)

    if lowcut or highcut:
        X = apply_filter(np.array(X), 500.0,lowcut=lowcut,highcut=highcut)

    X = np.array(X, dtype=np.float32)
    X = X-X.mean(axis=2,keepdims=True) #直流成分の除去
    X = np.transpose(X,(0,2,1))  # N,T,C
    return X,y

def load_test(subject:int, lowcut, highcut):
    '''トリガーの0.2秒から0.7秒までを取得する'''
    filename = f"./test/subject{subject}.mat"
    data_dict = read_mat(filename)
    X = data_dict["data"]*1e-6  # N,C,T

    if lowcut or highcut:
        X = apply_filter(X, 500.0 ,lowcut=lowcut,highcut=highcut)
        
    X = np.array(X, dtype=np.float32)
    X = X-X.mean(axis=2,keepdims=True) #直流成分の除去
    X = np.transpose(X,(0,2,1))  # N,T,C
    return X

def apply_filter(sig, fs, lowcut, highcut):
    '''
    sig: 信号(N,C,T)
    fs: サンプリング周波数
    lowcut: 下限カットオフ周波数 (Hz)
    highcut: 上限カットオフ周波数 (Hz)
    '''
    order = 1  # フィルタの次数    
    nyquist = 0.5*fs

    if lowcut and highcut:
        low = lowcut / nyquist
        high = highcut / nyquist
        b, a = signal.butter(order, [low, high], btype='bandpass')
    elif lowcut:
        low = lowcut / nyquist
        b, a = signal.butter(order, low, btype='highpass')
    elif highcut:
        high = highcut / nyquist
        b, a = signal.butter(order, high, btype='lowpass')
    else:
        return sig

    # フィルタの適用
    filtered_data = signal.filtfilt(b, a, sig)
    return filtered_data
