from utils import *
import efficientnet.keras as efn
from fastai.vision import Path
import pandas as pd



if __name__ == "__main__":
	videoPath = '../input/deepfake-detection-challenge/train_sample_videos'
	#csv = pd.read_csv(str(src/"sample_submission.csv"))
	ref = pd.read_json('../input/deepfake-detection-challenge/train_sample_videos/metadata.json')

	reals = [list(ref)[idx] for idx, val in enumerate(ref.loc['label']) if val == "REAL"]
	fakes = [list(ref)[idx] for idx, val in enumerate(ref.loc['label']) if val == "FAKE"]

	x, y = loadData(videoPath,reals,0,n=25)
	x_f,y_f = loadData(videoPath,fakes,1,n=25)

	model = efn.EfficientNetB7(weights='imagenet', include_top=False)

	rout = [model.predict(x[i]) for i in tqdm(range(x.shape[0]))]
	fout = [model.predict(x_f[i]) for i in tqdm(range(x_f.shape[0]))]
	ru,rv = getFMOTperClass(np.array(rout))
	fu,fv = getFMOTperClass(np.array(fout))
	visualize(ru,rv,fu,fv)

	ru,rv = getFMOSperClass(np.array(rout))
	fu,fv = getFMOSperClass(np.array(fout))

	visualize(ru,rv,fu,fv)