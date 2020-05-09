import numpy as np
import pandas as pd
import cv2, os
from tqdm.notebook import tqdm

def loadData(ref, videoPath, useTorch=False, limit = float('inf'), verbose=True):    
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
            #cv2.cvtColor(frame, cv2.CV_32F) :: for visualization only
            frame = cv2.resize(frame, dsize = (128,128))/255.0
            if useTorch: frame = torch.tensor(frame).transpose(0,-1).numpy()
            vid.append(frame)
            f = frame
        d = 300-len(vid)
        for i in range(d): vid.append(f)
        videos.append(np.array(vid))
        if limit<len(labels): break
    return (np.array(videos),np.array(labels)) if not useTorch else (torch.tensor(videos), torch.tensor(labels))

def iterRef(ref, n = 10):
    reals = np.random.choice([list(ref)[idx] for idx, val in enumerate(ref.loc['label']) if val == "REAL"], n)
    fakes = np.random.choice([list(ref)[idx] for idx, val in enumerate(ref.loc['label']) if val == "FAKE"], n)
    ref = pd.concat([pd.DataFrame(data = np.zeros((1,len(reals))), columns = reals),
               pd.DataFrame(data = np.ones((1,len(fakes))), columns = fakes)],axis=1)
    n = list(ref)
    random.shuffle(n)
    return ref[n]