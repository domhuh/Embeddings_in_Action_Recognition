import torch
import random
import cv2
import os
import numpy as np
import pandas as pd
from tqdm.notebook import tqdm
from torch.utils.data import DataLoader, TensorDataset

def get_equal_classes(ref, n = 10):
    if n is None:
        reals = [list(ref)[idx] for idx, val in enumerate(ref.loc['label']) if val == "REAL"]
        fakes = [list(ref)[idx] for idx, val in enumerate(ref.loc['label']) if val == "FAKE"]
    else:
        reals = np.random.choice([list(ref)[idx] for idx, val in enumerate(ref.loc['label']) if val == "REAL"], n)
        fakes = np.random.choice([list(ref)[idx] for idx, val in enumerate(ref.loc['label']) if val == "FAKE"], n)
    ref = pd.concat([pd.DataFrame(data = np.zeros((1,len(reals))), columns = reals),
                     pd.DataFrame(data = np.ones((1,len(fakes))), columns = fakes)],axis=1)
    n = list(ref)
    random.shuffle(n)
    return ref[n]

def load_videos(ref, videoPath, useTorch=False, limit = float('inf'), verbose=True):    
    videos = []
    labels = []
    t = limit if limit < float('inf') else len(list(ref))
    pb = tqdm(zip(ref,ref.iloc[0]), total = t) if verbose else zip(ref,ref.iloc[0])
    for fn,label in pb:
        if label == 1.0: label = 1
        else: label = 0
        labels.append(label)
        vid = []
        cap = cv2.VideoCapture(os.path.join(videoPath, fn))
        while(cap.isOpened()):
            ret, frame = cap.read()
            if not ret: break
            frame = cv2.resize(frame, dsize = (64,64))/255.0
            if useTorch: frame = torch.tensor(frame).transpose(0,-1).numpy()
            vid.append(frame)
            f = frame
        d = 300-len(vid) #pad to 300 (for outliers)
        for i in range(d): vid.append(f)
        if useTorch: vid = torch.tensor(vid)
        else: vid = np.array(vid)
        videos.append(vid)
        if limit<len(labels): break
    return (np.array(videos),np.array(labels)) if not useTorch else (torch.stack(videos).float(), torch.tensor(labels).float())

def convert_dls(x,y,nf=10, dims=(3, 64, 64), split=None, limit=10):
    samples = np.random.permutation(np.arange(0,int(x.shape[1]/10)))[:limit]
    a = torch.stack(torch.split(x,nf, dim=1))[samples].reshape(-1,nf,*dims)
    b = torch.cat([y.expand(int(x.shape[1]/nf),y.shape[0])[:,i][samples] for i in range(y.shape[0])])
    if not split is None:
        idx = int(a.shape[0]*split)
        vdl = DataLoader(TensorDataset(a[idx:], b[idx:]), batch_size=256, shuffle=True)
        tdl = DataLoader(TensorDataset(a[:idx], b[:idx]), batch_size=256, shuffle=True)
        return tdl, vdl
    tdl = DataLoader(TensorDataset(a,b), batch_size=64, shuffle=True)
    return tdl

def get_data(ref, n, videoPath, nf= 10, useTorch=True,
             verbose=False, split = None, limit = 10):
    ref_ = get_equal_classes(ref, n)
    x, y = load_videos(ref_, videoPath, useTorch=useTorch, verbose=verbose)
    tdl, vdl = convert_dls(x,y,nf=nf,split=split, limit=limit)
    return tdl, vdl
