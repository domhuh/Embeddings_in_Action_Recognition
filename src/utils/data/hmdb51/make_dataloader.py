import os
import numpy as np
import cv2
from tqdm import tqdm
import collections
import math

SAVE_PATH = "../HMDB51_NUMPY"
NUM_FRAME = 50
MAX_ITER = 1

if not os.path.exists(os.path.join(SAVE_PATH)):
    os.mkdir(os.path.join(SAVE_PATH))

folds = ["fold_1", "fold_2", "fold_3"]

labels = {key:value for value,key in enumerate(os.listdir(os.path.join(folds[0],"training")))}
print(labels)
for f in tqdm(folds):
    if not os.path.exists(os.path.join(SAVE_PATH, f)):
        os.mkdir(os.path.join(SAVE_PATH, f))

    split = os.listdir(f)
    for s in split:
        if not os.path.exists(os.path.join(SAVE_PATH, f, s)):
            os.mkdir(os.path.join(SAVE_PATH, f, s))
        c_path = os.path.join(f,s)
        classes = os.listdir(c_path)
        if True:#s == "training":
            for j in range(MAX_ITER):
                if not os.path.exists(os.path.join(SAVE_PATH, f, s, str(j))):
                    os.mkdir(os.path.join(SAVE_PATH, f, s,  str(j)))
                for c in tqdm(classes):
                    if not os.path.exists(os.path.join(SAVE_PATH, f, s, str(j), str(labels[c]))):
                        os.mkdir(os.path.join(SAVE_PATH, f, s, str(j), str(labels[c])))

                    v_path = os.path.join(c_path, c)
                    videos = os.listdir(v_path)
                    ref = collections.defaultdict(list)

                    for v in videos:
                        ref["_".join(v.split("_")[:-1])].append(v)

                    # if MAX_ITER < math.ceil(len(videos)/len(ref)):
                    #     MAX_ITER = math.ceil(len(videos)/len(ref))

                    for i, v in enumerate(ref.values()):
                        p = np.random.choice(v) #Right now, it is randomly selecting. (add biasing distribution if desired)
                        cap = cv2.VideoCapture(os.path.join(v_path,p))
                        vid = []
                        while (cap.isOpened()):
                            ret, frame = cap.read()
                            if not ret: break
                            vid.append(cv2.resize(frame / 255, (64,64)))
                        #subset = np.random.choice(len(vid), NUM_FRAME)
                        subset = np.arange(0,len(vid),math.floor(len(vid)/NUM_FRAME))
                        vid = np.array(vid)[subset]
                        np.save(os.path.join(SAVE_PATH, f, s, str(j), str(labels[c]), str(i)), vid)
        # else:
        #     for c in tqdm(classes):
        #         if not os.path.exists(os.path.join(SAVE_PATH, f, s, str(labels[c]))):
        #             os.mkdir(os.path.join(SAVE_PATH, f, s, str(labels[c])))
        #         if not os.path.exists(os.path.join(SAVE_PATH, f, s, '0', str(labels[c]))):
        #             os.mkdir(os.path.join(SAVE_PATH, f, s, '0', str(labels[c])))
        #
        #         v_path = os.path.join(c_path, c)
        #         videos = os.listdir(v_path)
        #         for i,v in enumerate(videos):
        #             l_path = os.path.join(v_path, v)
        #             cap = cv2.VideoCapture(l_path)
        #             vid = []
        #             while (cap.isOpened()):
        #                 ret, frame = cap.read()
        #                 if not ret: break
        #                 vid.append(cv2.resize(frame / 255, (64,64)))
        #             np.save(os.path.join(SAVE_PATH, f, s, '0', str(labels[c]), str(i)), vid)