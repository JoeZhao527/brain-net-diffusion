from typing import Any, List
import pandas as pd
import numpy as np
import random
from functools import partial

from torch.utils.data import Dataset
from data_prep.feature_store.store import FeatureStore
from tqdm import tqdm


def one_hot(cls_idx, n_classes: int = 2):
    oh = [0.0] * n_classes
    oh[int(cls_idx)] = 1.0
    return np.float32(oh)


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


class AbideDataset(Dataset):
    def __init__(
        self,
        store_path: str,
        data: pd.DataFrame,
        id_col: str,
        use_cols: List[str],
        conditions: List[str] = ["SEX", "AGE_AT_SCAN"],
        **kwargs
    ) -> None:
        super().__init__()

        self.id_col = id_col
        self.use_cols = use_cols

        self.conditions = conditions
        self.feature_store = FeatureStore(store_path)
        self.condition_func = partial(make_conditions, conditions=conditions)
        self.samples = self._init_data(data)
        
    def _init_data(self, data: pd.DataFrame):
        samples = []

        for rec in tqdm(data.to_dict("records"), desc="initializing data"):
            _id = rec['SUB_ID']
            label = rec['DX_GROUP'] - 1
            feat = self.feature_store.load(_id)

            samples.append({
                'id': _id,
                'edge_mtx': np.float32(feat['edge_mtx']),
                'label': one_hot(label),
                'condition': self.condition_func(record=rec),
                'condition_key': self.conditions,
                'split': rec['split']
            })

        return samples
    
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx) -> Any:
        return {
            k: v for k,v in self.samples[idx].items()
            if k in self.use_cols
        }
    

class AbideLatentDataset(AbideDataset):
    def _init_data(self, data: pd.DataFrame):
        samples = []

        for rec in data.to_dict("records"):
            _id = rec['SUB_ID']
            label = rec['DX_GROUP'] - 1
            feat = self.feature_store.load(_id)

            samples.append({
                'id': _id,
                'signal': np.float32(feat.get("reconstruct", np.array([]))),
                'edge': np.float32(feat["latent_repr"]),
                'edge_mtx': np.float32(feat.get("edge_mtx", np.array([]))),
                'label': one_hot(label),
                'sex': np.float32(rec['SEX']),
                'age': np.float32(rec['AGE_AT_SCAN']),
                'handiness': rec['HANDEDNESS_CATEGORY'],
                'condition': self.condition_func(record=rec),
                'split': rec['split']
            })

        return samples
    

class AbideAugmentDataset(AbideDataset):
    def __init__(
        self,
        original_feat_path: str,
        sampling: bool = False,
        sampling_prob: float = 0.5,
        graph_feat: bool = False,
        set_size: int = 5,
        set_num: int = 4,
        set_idx: str = "",
        fc_key: str = "edge",
        alter_labels: bool = False,
        **kwargs
    ) -> None:
        self.sampling = sampling
        self.sampling_prob = sampling_prob
        self.graph_feat = graph_feat
        self.fc_key = fc_key
        self.alter_labels = alter_labels
        
        print(f"Alter label: {self.alter_labels}")
        self.original_fs = FeatureStore(original_feat_path)

        set_idx = str(set_idx)
        self.sampling_idx = self._get_sample_idx(set_size, set_idx, set_num)
        print(f"sample idx: {self.sampling_idx}")
        super().__init__(**kwargs)
    
    def _get_sample_idx(self, set_size, set_idx, set_num):
        """
        Generated samples are in order like:
        [guide_0, guide_1, ..., guide_n, guide_0, guide_1, ..., guide_n, ...]

        Args:
            set_size: number of guided sample in each set
            set_num: number of sets
            set_idx: which guide to select in each set

        Returns:
            A list of sample's index to select
        """
        if len(set_idx) == 0:
            print(f"set_idx not given, sampling all samples")
            return []
        
        set_idx = [int(i) for i in set_idx.split("-")]
        sampling_idx = []
        for _set in range(set_num):
            offset = 1 + _set * set_size
            sampling_idx.extend([offset + idx for idx in set_idx])
        
        return sampling_idx
    
    def _init_data(self, data: pd.DataFrame):
        samples = []

        for rec in tqdm(data.to_dict("records"), desc="initializing data"):
            _id = rec['SUB_ID']
            label = one_hot(rec['DX_GROUP'] - 1)
            feat = self.feature_store.load(_id)
            
            samples.append({
                'id': _id,
                'edge_mtx': np.float32(feat[self.fc_key]),
                'label': np.float32(label),
                'condition': self.condition_func(record=rec),
                'split': rec['split'],
                'alter_labels': np.float32(feat.get('labels', []))
            })

        return samples
    
    def __getitem__(self, idx) -> Any:
        item = {
            k: v for k,v in self.samples[idx].items()
            if k in self.use_cols
        }

        # Sampling, item[0] is the real data, otherwise sythentic
        sampling = self.sampling and random.uniform(0, 1) <= self.sampling_prob

        if sampling and item["edge_mtx"].shape[0] > 1:
            if len(self.sampling_idx) > 0:
                # if the sampling indicies are given, sample from the given indicies
                view_idx = random.choice(self.sampling_idx)
            else:
                # Otherwise sample from the whole generated samples
                view_idx = random.randint(1, item["edge_mtx"].shape[0] - 1)
        else:
            view_idx = 0

        # print(view_idx, end='')
        # item["edge"] = item["edge"][view_idx]
        item["edge_mtx"] = item["edge_mtx"][view_idx]
        item["view_index"] = view_idx

        if self.alter_labels:
            item["label"] = item['alter_labels'][view_idx]
        
        
        return item