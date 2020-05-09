import pickle
from fastai.vision import Path
import matplotlib.pyplot as plt
import os
from tqdm.notebook import tqdm
from .models.trainers import HMDB51Trainer, UCF101Trainer
from .models import HMDB51_ConvRecurrent, UCF101_ConvRecurrent,DilatedRNN, AttentionRNN, ClockworkRNN, Transformer_Encoder, fusion, MLP

paths = [Path("../input/trained-hmdb51").ls(),Path("../input/pretrained-hmdb51").ls()]
paths[0].sort()
paths[1].sort()

for t,pt in zip(*paths):
    with open(str(t),'rb') as f:
        trained_model = pickle.load(f)
    s = os.path.basename(t)

    for FOLD in FOLDS:
        BASE_PATH = f"../input/hmdb51-fold{FOLD}/fold_{FOLD}"
        with open(os.path.join(BASE_PATH,"tdl_hmdb51.pkl"), 'rb') as f:
            tdl = pickle.load(f)
        with open(os.path.join(BASE_PATH,"vdl_hmdb51.pkl"), 'rb') as f:
            vdl = pickle.load(f)
        model = MLP(conv = trained_model.conv, nh = 2048, device = 'cuda').cuda()
        model.fit(tdl, vdl)

        FILE_NAME = f"hmdb51/es/{s}_{FOLD}.pkl" #downloads full model
        with open(FILE_NAME) as f:
            pickle.dump(model,f)

    with open(str(pt),'rb') as f:
        pretrained_model = pickle.load(f)
    s = os.path.basename(t)
    for FOLD in FOLDS:
        BASE_PATH = f"../input/hmdb51-fold{FOLD}/fold_{FOLD}"
        with open(os.path.join(BASE_PATH,"tdl_hmdb51.pkl"), 'rb') as f:
            tdl = pickle.load(f)
        with open(os.path.join(BASE_PATH,"vdl_hmdb51.pkl"), 'rb') as f:
            vdl = pickle.load(f)
        model = MLP(conv = trained_model.conv, nh = 2048, device = 'cuda').cuda()
        model.fit(tdl, vdl)

        FILE_NAME = f"hmdb51/es/{s}_{FOLD}.pkl" #downloads full model
        with open(FILE_NAME) as f:
            pickle.dump(model,f)

paths = [Path("../input/trained-ucf101").ls(),Path("../input/pretrained-ucf101").ls()]
paths[0].sort()
paths[1].sort()

for t,pt in zip(*paths):
    with open(str(t),'rb') as f:
        trained_model = pickle.load(f)
    s = os.path.basename(t)

    for FOLD in FOLDS:
        BASE_PATH = f"../input/ucf101-fold{FOLD}/fold_{FOLD}"
        with open(os.path.join(BASE_PATH,"tdl_ucf101.pkl"), 'rb') as f:
            tdl = pickle.load(f)
        with open(os.path.join(BASE_PATH,"vdl_ucf101.pkl"), 'rb') as f:
            vdl = pickle.load(f)
        model = MLP(conv = trained_model.conv, nh = 2048, device = 'cuda').cuda()
        model.fit(tdl, vdl)

        FILE_NAME = f"ucf101/es/{s}_{FOLD}.pkl" #downloads full model
        with open(FILE_NAME) as f:
            pickle.dump(model,f)

    with open(str(pt),'rb') as f:
        pretrained_model = pickle.load(f)
    s = os.path.basename(t)
    for FOLD in FOLDS:
        BASE_PATH = f"../input/ucf101-fold{FOLD}/fold_{FOLD}"
        with open(os.path.join(BASE_PATH,"tdl_ucf101.pkl"), 'rb') as f:
            tdl = pickle.load(f)
        with open(os.path.join(BASE_PATH,"vdl_ucf101.pkl"), 'rb') as f:
            vdl = pickle.load(f)
        model = MLP(conv = trained_model.conv, nh = 2048, device = 'cuda').cuda()
        model.fit(tdl, vdl)

        FILE_NAME = f"ucf101/es/{s}_{FOLD}.pkl" #downloads full model
        with open(FILE_NAME) as f:
            pickle.dump(model,f)

