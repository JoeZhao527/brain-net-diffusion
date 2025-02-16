import os
from typing import Dict, Any
import torch


class FeatureStore:
    def __init__(self, store_path: str) -> None:
        self.store_path = store_path
        os.makedirs(store_path, exist_ok=True)

    def dump(self, k: str, v: Dict[str, Any]) -> None:
        dump_path = os.path.join(self.store_path, f"{k}.pt")
        if not os.path.exists(dump_path):
            with open(dump_path, "wb") as f:
                torch.save(v, dump_path)

    def load(self, k: str) -> Dict[str, Any]:
        load_path = os.path.join(self.store_path, f"{k}.pt")

        with open(load_path, "rb") as f:
            feat = torch.load(f)

        return feat