import pickle
from ConvLSTM import *
from ..utils import *
import matplotlib.pyplot as plt
from fastai.vision import Path

if __name__ == "__main__":
	videoPath = '../input/deepfake-detection-challenge/train_sample_videos'
	ref = pd.read_json('../input/deepfake-detection-challenge/train_sample_videos/metadata.json')

	model = ConvLSTM(64, 1).cuda()

	for _ in range(50):
	    ref_ = iterRef(ref, 10)
	    x, y = loadData(ref_, videoPath, useTorch=True, verbose=False)
	    model.fit_(x.cuda().float(), y.cuda().float(), epochs = 5)

	plt.plot(model.accuracy)

	with open('../working/model.pkl', 'wb') as path:
	    pickle.dump(model, path)