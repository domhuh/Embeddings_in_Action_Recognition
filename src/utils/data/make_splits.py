import os
from shutil import copyfile

video_path = "hmdb51_org"
annotations = "hmdb51_annotations"
folds = ["fold_1", "fold_2", "fold_3"]

for i in folds:
    if not os.path.exists(i):
        os.mkdir(i)
        os.mkdir(os.path.join(i,"training"))
        os.mkdir(os.path.join(i,"validation"))


for path in os.listdir(annotations):
    fold = int([*str(path)][-5])-1
    with open(os.path.join(annotations,path),'r') as f:
        file = f.read()
    
    key = []
    for i in os.path.basename(path).split("_"):
        if i == "test": break
        key.append(i)
    key = "_".join(key)
    
    for f in file.split(" \n"):
        try:
            file_name, split = f.split(' ')
        except:
            continue
        src = os.path.join(video_path, key, key, file_name)
        if split == '0':
            continue
        if split == '1':
            dst = os.path.join(folds[fold],"training", key)
        if split == '2':
            dst = os.path.join(folds[fold],"validation", key)
        if not os.path.exists(dst):
            os.mkdir(dst)
        copyfile(src, os.path.join(dst, file_name))
