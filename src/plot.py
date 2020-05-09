import pickle
from fastai.vision import Path
import matplotlib.pyplot as plt
import os

paths = [Path("../input/trained-hmdb51").ls(),Path("../input/pretrained-hmdb51").ls()]
paths[0].sort()
paths[1].sort()

colors = [(255/255,0.0,0.0),
          (255/255,127/255,0.0),
          (35/255,98/255,143/255),
          (79/255,143/255,35/255),
          (170/255,0.0,255/255),
          (255/255,0.0,170/255),
          (0.0,64/255,255/255),
          (200/255,100/255,255/255),
          (0.0,0.0,170/255),
          (90/255,64/255,52/255)]

fig = plt.figure(figsize=(20,10))
for n, x in enumerate(zip(*paths)):
    with open(str(x[0]),'rb') as f:
        trained_model = pickle.load(f)
    with open(str(x[1]),'rb') as f:
        pretrained_model = pickle.load(f)

    ax = plt.subplot(121)
    s = os.path.basename(x[0])[:-4].split("_")
    if s[0] == 'vanilla': name = f"{s[1]} ({s[-2]}={s[-1]})"
    else: name = f"{s[0]} ({s[-2]}={s[-1]})"
    ax.plot(np.array(trained_model.training_loss), color=(*colors[n],0.25), linestyle = "--")
    ax.plot(np.array(trained_model.validation_loss), color=(*colors[n],0.5), linestyle = "--", label=name)
    
    s = os.path.basename(x[1])[:-4].split("_")
    if s[0] == 'vanilla': name = f"{s[1]} ({s[-2]}={s[-1]})"
    else: name = f"{s[0]} ({s[-2]}={s[-1]})"
    ax.plot(pretrained_model.training_loss, color=(*colors[n],0.75), linestyle = "-")
    ax.plot(pretrained_model.validation_loss, color=(*colors[n],1), linestyle = "-", label=name)
    
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Loss")
    ax.set_title("HMDB51 Loss")
    
    ax = plt.subplot(122)
    s = os.path.basename(x[0])[:-4].split("_")
    if s[0] == 'vanilla': name = f"{s[1]} ({s[-2]}={s[-1]})"
    else: name = f"{s[0]} ({s[-2]}={s[-1]})"
    train = np.array(trained_model.training_accuracy)
    valid = np.array(trained_model.validation_accuracy)
    ax.plot(train, color=(*colors[n],0.75), linestyle = "-")
    ax.plot(valid, color=(*colors[n],1), linestyle = "-", label=name)

    s = os.path.basename(x[1])[:-4].split("_")
    if s[0] == 'vanilla': name = f"{s[1]} ({s[-2]}={s[-1]})"
    else: name = f"{s[0]} ({s[-2]}={s[-1]})"
    
    train = np.array(pretrained_model.training_accuracy)
    valid = np.array(pretrained_model.validation_accuracy)
    ax.plot(train, color=(*colors[n],0.75), linestyle = "-")
    ax.plot(valid, color=(*colors[n],1), linestyle = "-", label=name)

    ax.set_xlabel("Epochs")
    ax.set_ylabel("Accuracy")
    ax.set_title("HMDB51 Accuracy")
    
plt.subplot(122)
handles, labels = ax.get_legend_handles_labels()
lgd = plt.legend(handles, labels, bbox_to_anchor=(1.01, 0.5))#loc='best')
fig.savefig('../working/hmdb51_training.png', bbox_extra_artists=(lgd,), bbox_inches='tight',dpi=300)
plt.show()