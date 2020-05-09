import os
from shutil import copy

video_path = "ucf101"
annotations = "ucf101_annotations"
folds = ["fold_1", "fold_2", "fold_3"]

for i in folds:
    if not os.path.exists(i):
        os.mkdir(i)
        os.mkdir(os.path.join(i,"training"))
        os.mkdir(os.path.join(i,"validation"))

for path in os.listdir(annotations):
    try:
        fold = int([*str(path)][-5])-1
        split = path.split('_')[0]

        with open(os.path.join(annotations,path),'r') as f:
            file = f.read()
    except:
        continue
    for f in file.split("\n")[:-1]:
        key = f.split('/')[0]
        f = f.split(" ")[0]
        src = os.path.join(video_path, f)
        if split == 'train':
            dst = os.path.join(folds[fold],"training", key)
        if split == 'test':
            dst = os.path.join(folds[fold],"validation", key)
        if not os.path.exists(dst):
            os.mkdir(dst)
        copy(src, dst)
