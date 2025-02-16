import os
import torch
from typing import List


def load_from_path(module: torch.nn.Module, ckpt_path: str, skip_prefix: List[str]):
    with open(ckpt_path, "rb") as f:
        state_dict = torch.load(f, map_location=torch.device('cpu'))
    
    state_dict = {
        ".".join([p for p in k.split(".") if p not in skip_prefix]): v
        for k, v in state_dict['state_dict'].items()
    }
    module.load_state_dict(state_dict)

    return module

def load_from_dir(module: torch.nn.Module, ckpt_dir: str, mode: str = "best", skip_prefix: List[str]= ['net']):
    ckpts = sorted(os.listdir(ckpt_dir))

    if len(ckpts) == 2:
        best, last = ckpts
    else:
        best = ckpts[0]
    
    return load_from_path(module, os.path.join(ckpt_dir, eval(mode)), skip_prefix)