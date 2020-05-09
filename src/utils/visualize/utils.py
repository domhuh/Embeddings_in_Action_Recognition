import numpy as np
import cv2
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
import os
import scipy.stats as stats
import math

def loadData(videoPath, filenames, label, n=1):
    videos = []
    labels = []
    for fn in tqdm(filenames[:n]):
        labels.append(label)
        vid = []
        cap = cv2.VideoCapture(os.path.join(videoPath, fn))
        while(cap.isOpened()):
            ret, frame = cap.read()
            if not ret: break
            #cv2.cvtColor(frame, cv2.CV_32F) :: for visualization only
            frame = cv2.resize(frame, dsize = (128,128))/255.0
            #if useTorch: frame = torch.tensor(frame).transpose(0,-1).numpy()
            vid.append(frame)
            f = frame
        d = 300-len(vid)
        for i in range(d): vid.append(f)
        #print(np.array(vid).shape)
        videos.append(np.array(vid))
    return (np.array(videos),np.array(labels))

def getFMOTperClass(data):
    u, var = [], []
    for i in tqdm(range(data.shape[-1])):
        u.append(np.mean(data[:,:,:,:,i]))
        var.append(np.std(data[:,:,:,:,i])**2)
    return np.array(u),np.array(var)

def getFMOSperClass(data):
    u, var = [], []
    for x in tqdm(data):
        u.append(np.mean(x))
        var.append(np.std(x)**2)
    return np.array(u),np.array(var)

def bhattacharyya(h1, h2):
    def normalize(h):
        return h / np.sum(h)
    return 1 - np.sum(np.sqrt(np.multiply(normalize(h1), normalize(h2))))

def visualize(ru,rv,fu,fv,nf=5):
    bf = []
    xs = np.linspace(-3,3,1000)
    for fm, fvar, rm, rvar in zip(fu[:nf],fv[:nf],ru[:nf],rv[:nf]):
        f_std = fvar**0.5
        r_std = rvar**0.5
        plt.plot(xs, stats.norm.pdf(xs, fm, f_std), color = 'red')
        plt.plot(xs, stats.norm.pdf(xs, rm, r_std), color = 'blue')
        bf.append(bhattacharyya(stats.norm.pdf(xs, fm, f_std), stats.norm.pdf(xs, rm, r_std)))
    plt.show()
    return bf

