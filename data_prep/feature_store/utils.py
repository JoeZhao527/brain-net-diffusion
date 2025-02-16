import numpy as np
import scipy.io
from typing import List
import numpy as np


def safe_corr(arr):
    """
    Compute pair wise pearson along the 2nd dimension
    To handle constant channel, replace nan with 0
    """
    corr = np.corrcoef(arr, rowvar=False)
    corr[np.isnan(corr)] = 0.0

    return corr

def zscore_norm(arr):
    # Z-score normalization
    mean_val = np.mean(arr)
    std_dev_val = np.std(arr)

    return (arr - mean_val) / std_dev_val


def roi_zscore_norm(roi):
    """
    z-score normalize for all regions in a sample
    """
    norm_res = []
    for region in roi.transpose():
        norm_res.append(zscore_norm(region))
    
    return np.stack(norm_res).transpose()


def roi_preprocessing(signal_path):
    """
    z-score normalize for all samples
    """
    data = scipy.io.loadmat(signal_path)

    sub_id = []
    roi_signals = []

    all_sample_roi = data['ROIsignals'].flatten()
    for i in range(data['ROIsignals'].shape[0]):
        _id = data['subID'].flatten()[i][0]

        # only keep run-1 fMRI
        if not "run-1" in _id: continue
        
        # z-score normalization
        roi_signals.append(all_sample_roi[i])

        # record corresponding sub id
        sub_id.append(int(_id.split('_')[0]))

    return roi_signals, sub_id


def acerta_roi_preprocessing(signal_path):
    # No Normalizing
    # And the data already has dimension (sample, time, ROI)
    data = scipy.io.loadmat(signal_path)

    return data['ROIsignals'][0], data['subID'][0]


def sliding_window(array: np.ndarray, window_size: int, step_size: int):
    """
    slide a window along the first dimension of an array
    array has dimension (time, region)
    """
    window_size = min(array.shape[0], window_size)
    windows = []
    for i in range(0, array.shape[0] - window_size + 1, step_size):
        windows.append(array[i:i+window_size, :])
        
    return windows

def build_corr_mtx(windows):
    """
    build pearson correlation matrix for each window
    """
    corr = []
    for w in windows:
        corr.append(safe_corr(w))
    
    return corr

def pair_dot_construction(array):
    """
    build pairwise dot matrix for each timestamp
    """
    frames = []
    for i in range(array.shape[0]):
        frames.append(array[i][:, None] * array[i][None, :])
        
    return frames

def stack_corrs(corrs, flatten: bool = True):
    """
    Stack the corr matrix

    If flatten is required, keep the lower triangular matrix and
    flatten it before stack
    """
    if flatten:
        feat = []
        for corr in corrs:
            feat.append(corr[np.tril_indices_from(corr, k=-1)])
    else:
        feat = corrs

    return np.stack(feat, axis=0)


def zero_one_normalize(array):
    """
    Perform zero-one normalization on a NumPy array.

    Args:
    array (np.ndarray): Input array to be normalized.

    Returns:
    np.ndarray: Normalized array.
    """
    min_val = np.min(array)
    max_val = np.max(array)
    normalized_array = (array - min_val) / (max_val - min_val)
    return normalized_array


def make_conditions(record: dict, conditions: List[str]):
    cond = []
    for k in conditions:
        if k in ["DX_GROUP", "SEX"]:
            cond.append(record[k])
        elif k in ["AGE_AT_SCAN"]:
            cond.append(record[k])
        elif k == "HANDEDNESS_CATEGORY":
            handedness_dict = {'L': 0, 'R': 1, 'Ambi': 2}
            if record[k] in ['L', 'R', 'Ambi']:
                hand = record[k]
            elif record[k] == 'Mixed':
                hand = 'Ambi'
            elif record[k] == 'L->R':
                hand = 'Ambi'
            else:
                hand = 'R'
            cond.append(handedness_dict[hand])
        elif k in ["FIQ", "PIQ", "VIQ"]:
            iq = float(record[k])
            if (record[k] == -9999) or np.isnan(record[k]):
                iq = 100
            cond.append(iq)

    return np.array(cond, dtype=np.float32)