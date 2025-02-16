import numpy as np
import pandas as pd
from tqdm import tqdm
import os
import warnings
warnings.filterwarnings("ignore")

from data_prep.feature_store.store import FeatureStore
from nilearn.connectome import ConnectivityMeasure

def preprocess(
    roi_dir: str = './downloads/functionals/cpac/filt_global/rois_cc200',
    store_path: str = './feature/rois_cc200/raw'
):
    # Basic preprocessing, for simple linear networks
    feature_store = FeatureStore(store_path)

    for f in tqdm(os.listdir(roi_dir)):
        sub_id = int(f.split('_')[-3])
        signal = pd.read_csv(os.path.join(roi_dir, f), sep='\t').to_numpy()
        corr = ConnectivityMeasure(kind='correlation').fit_transform([signal])[0]
        corr[np.isnan(corr)] = 0.0
        pcorr = ConnectivityMeasure(kind='partial correlation').fit_transform([signal])[0]
        pcorr[np.isnan(pcorr)] = 0.0

        feature_store.dump(k=sub_id, v={
            "signal": signal,
            "edge_mtx": corr
        })