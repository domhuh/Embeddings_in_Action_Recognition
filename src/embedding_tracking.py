import pickle
from fastai.vision import Path
import matplotlib.pyplot as plt
import os
from tqdm.notebook import tqdm
from .models import VAE
from .models.trainers import HMDB51Trainer, UCF101Trainer
from .models import HMDB51_ConvRecurrent, UCF101_ConvRecurrent,DilatedRNN, AttentionRNN, ClockworkRNN, Transformer_Encoder, fusion

paths = [Path("../input/trained-hmdb51").ls(),Path("../input/pretrained-hmdb51").ls()]
paths[0].sort()
paths[1].sort()

NSAMPLES = 50

def kl_divergence(mu1, mu2, sigma_1, sigma_2):
    sigma_diag_1 = np.eye(sigma_1.shape[0]) * sigma_1
    sigma_diag_2 = np.eye(sigma_2.shape[0]) * sigma_2

    sigma_diag_2_inv = np.linalg.inv(sigma_diag_2)

    kl = 0.5 * (np.log(np.linalg.det(sigma_diag_2) / np.linalg.det(sigma_diag_2))
                - mu1.shape[0] + np.trace(np.matmul(sigma_diag_2_inv, sigma_diag_1))
                + np.matmul(np.matmul(np.transpose(mu2 - mu1), sigma_diag_2_inv), (mu2 - mu1)))

    return kl

for t,pt in zip(*paths):
    with open(str(t),'rb') as f:
        trained_model = pickle.load(f)
    s = os.path.basename(t)
    dists = []
    x = torch.stack(trained_model.reduced_embeddings)
    model = VAE(conv = trained_model.conv, nh = 2048, device = 'cuda').cuda()
    model.fit_encoder(x, num_epochs=50, lr = 1e-3)
    for i in range(NSAMPLES):
        dists.append(model.get_dist(x))
    mu = torch.stack(dists).mean(0).cpu().detach().numpy()
    std = (torch.stack(dists).var(0)**0.5).cpu().detach().numpy()
    kl = []
    for i in range(100):
        ref = -1 #np.minimum(99,i+3)
        kl.append(kl_divergence(mu[i],mu[ref],std[i],std[ref]))
    kl = np.array(kl)
    FILE_NAME = f"/kls/{s}_kl.pkl" #downloads full model
    with open(FILE_NAME) as f:
        pickle.dump(kl,f)

    with open(str(pt),'rb') as f:
        pretrained_model = pickle.load(f)
    s = os.path.basename(t)[:-4]
    dists = []
    x = torch.stack(pretrained_model.reduced_embeddings)
    model = VAE(conv = pretrained_model.conv, nh = 2048, device = 'cuda').cuda()
    model.fit_encoder(x, num_epochs=50, lr = 1e-3)
    for i in range(NSAMPLES):
        dists.append(model.get_dist(x))
    mu = torch.stack(dists).mean(0).cpu().detach().numpy()
    std = (torch.stack(dists).var(0)**0.5).cpu().detach().numpy()
    kl = []
    for i in range(100):
        ref = -1 #np.minimum(99,i+3)
        kl.append(kl_divergence(mu[i],mu[ref],std[i],std[ref]))
    kl = np.array(kl)

    FILE_NAME = f"/kls/{s}_kl.pkl" #downloads full model
    with open(FILE_NAME) as f:
        pickle.dump(kl,f)

paths = [Path("../input/trained-ucf101").ls(),Path("../input/pretrained-ucf101").ls()]
paths[0].sort()
paths[1].sort()

for t,pt in zip(*paths):
    with open(str(t),'rb') as f:
        trained_model = pickle.load(f)
    s = os.path.basename(t)
    dists = []
    x = torch.stack(trained_model.reduced_embeddings)
    model = VAE(conv = trained_model.conv, nh = 2048, device = 'cuda').cuda()
    model.fit_encoder(x, num_epochs=50, lr = 1e-3)
    for i in range(NSAMPLES):
        dists.append(model.get_dist(x))
    mu = torch.stack(dists).mean(0).cpu().detach().numpy()
    std = (torch.stack(dists).var(0)**0.5).cpu().detach().numpy()
    kl = []
    for i in range(100):
        ref = -1 #np.minimum(99,i+3)
        kl.append(kl_divergence(mu[i],mu[ref],std[i],std[ref]))
    kl = np.array(kl)
    FILE_NAME = f"/kls/{s}_kl.pkl" #downloads full model
    with open(FILE_NAME) as f:
        pickle.dump(kl,f)

    with open(str(pt),'rb') as f:
        pretrained_model = pickle.load(f)
    s = os.path.basename(t)[:-4]
    dists = []
    x = torch.stack(pretrained_model.reduced_embeddings)
    model = VAE(conv = pretrained_model.conv, nh = 2048, device = 'cuda').cuda()
    model.fit_encoder(x, num_epochs=50, lr = 1e-3)
    for i in range(NSAMPLES):
        dists.append(model.get_dist(x))
    mu = torch.stack(dists).mean(0).cpu().detach().numpy()
    std = (torch.stack(dists).var(0)**0.5).cpu().detach().numpy()
    kl = []
    for i in range(100):
        ref = -1 #np.minimum(99,i+3)
        kl.append(kl_divergence(mu[i],mu[ref],std[i],std[ref]))
    kl = np.array(kl)

    FILE_NAME = f"/kls/{s}_kl.pkl" #downloads full model
    with open(FILE_NAME) as f:
        pickle.dump(kl,f)

#plt.plot(np.minimum(np.maximum(0,(kl[0]-kl)/kl[0]),1.0), label = name)
#plt.show()


# from torch.utils.data import DataLoader, TensorDataset
# import collections
# #split dl by class
# def splitbyclass(dl):
#     ref = collections.defaultdict(list)
#     for x,y in dl:
#         for x_, y_ in zip(x,y):
#             ref[y_.item()].append(x_)
#     return ref
# paths = [Path("../input/trained-hmdb51").ls(),Path("../input/pretrained-hmdb51").ls()]
# paths[0].sort()
# paths[1].sort()

# ref = splitbyclass(tdl)

# for c in ref.keys():
#     x = torch.stack(ref[c])
#     y = torch.tensor([c]).expand(x.shape[0])
#     dl = DataLoader(TensorDataset(x,y), shuffle = True, batch_size = 128)
#     #train a vae and save it (label by class)
#     for t,pt in zip(*paths):
#         with open(str(t),'rb') as f:
#             trained_model = pickle.load(f)
#         with open(str(pt),'rb') as f:
#             pretrained_model = pickle.load(f)
#         model = VAE(conv = m.conv, nh = 2048, device = 'cuda').cuda()
#         model.fit_encoder(tdl, num_epochs=50, lr = 1e-3)
#         with open(f"../working/{os.path.basename(t)}",'rb') as f:
#             pretrained_model = pickle.load(f)

# paths = [Path("../input/trained-ucf101").ls(),Path("../input/pretrained-ucf101").ls()]
# paths[0].sort()
# paths[1].sort()

# ref = splitbyclass(tdl)

# for c in ref.keys():
#     x = torch.stack(ref[c])
#     y = torch.tensor([c]).expand(x.shape[0])
#     dl = DataLoader(TensorDataset(x,y), shuffle = True, batch_size = 128)
#     #train a vae and save it (label by class)
#     for t,pt in zip(*paths):
#         with open(str(t),'rb') as f:
#             trained_model = pickle.load(f)
#         with open(str(pt),'rb') as f:
#             pretrained_model = pickle.load(f)
#         model = VAE(conv = m.conv, nh = 2048, device = 'cuda').cuda()
#         model.fit_encoder(tdl, num_epochs=50, lr = 1e-3)
#         with open(f"../working/{os.path.basename(t)}",'rb') as f:
#             pretrained_model = pickle.load(f)
#         break
#     break