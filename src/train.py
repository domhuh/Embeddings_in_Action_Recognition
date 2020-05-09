import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models as models

#from package.utils import *
from .models.trainers import HMDB51Trainer, UCF101Trainer
from .models import HMDB51_ConvRecurrent, UCF101_ConvRecurrent,DilatedRNN, AttentionRNN, ClockworkRNN, Transformer_Encoder, fusion
from .models.ConvRecurrent import init_weights
import pickle
import gc

tdl, vdl = None, None

conv_models = ["vanilla","EF","SF","LF"]
recurrent_models = ["RNN","LSTM","GRU","CRNN", "DRNN","Attention"]

FOLDS = [1,2,3]
pretrained = [True, False]

for FOLD in FOLDS:
    BASE_PATH = f"../input/hmdb51-fold{FOLD}/fold_{FOLD}"
    with open(os.path.join(BASE_PATH,"tdl_hmdb51.pkl"), 'rb') as f:
        tdl = pickle.load(f)
    with open(os.path.join(BASE_PATH,"vdl_hmdb51.pkl"), 'rb') as f:
        vdl = pickle.load(f)
    for conv_model in conv_models: 
        for recurrent_model in recurrent_models: 
            for pt in pretrained:
                model = HMDB51_ConvRecurrent(conv = conv_model, rnn=recurrent_model, pretrained=pt, device = 'cuda').cuda()
                _ = model.apply(init_weights)
                model.fit(tdl,vdl, epochs = 100, lr=1e-6)
                model.cpu()
                FILE_NAME = f"../hmdb51/{conv_model}_{recurrent_model}_pretrained_{pt}_{FOLD}.pkl" #downloads full model
                with open(FILE_NAME, 'wb') as f:
                    pickle.dump(model, f)
                model = None
                del model
                torch.cuda.empty_cache()
                gc.collect()
            if conv_model != "vanilla":
                break

for FOLD in FOLDS:
    BASE_PATH = f"../input/ucf101-fold{FOLD}/fold_{FOLD}"
    with open(os.path.join(BASE_PATH,"tdl_ucf101.pkl"), 'rb') as f:
        tdl = pickle.load(f)
    with open(os.path.join(BASE_PATH,"vdl_ucf101.pkl"), 'rb') as f:
        vdl = pickle.load(f)
    for conv_model in conv_models: 
        for recurrent_model in recurrent_models: 
            for pt in pretrained:
                model = UCF101_ConvRecurrent(conv = conv_model, rnn=recurrent_model, pretrained=pt, device = 'cuda').cuda()
                _ = model.apply(init_weights)
                model.fit(tdl,vdl, epochs = 100, lr=1e-6)
                model.cpu()
                FILE_NAME = f"../ucf101/{conv_model}_{recurrent_model}_pretrained_{pt}_{FOLD}.pkl" #downloads full model
                with open(FILE_NAME, 'wb') as f:
                    pickle.dump(model, f)
                model = None
                del model
                torch.cuda.empty_cache()
                gc.collect()
            if conv_model != "vanilla":
                break