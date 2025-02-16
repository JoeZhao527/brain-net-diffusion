import torch
import os

class TensorRecorder:
    def __init__(self, out_dir: str = "./debug") -> None:
        self.out_dir = out_dir
        self.counter = 0

        # os.makedirs(self.out_dir, exist_ok=True)

    def record(self, x, prefix: str = ""):
        path = os.path.join(self.out_dir, f"{prefix}{self.counter}.pt")

        # with open(path, "wb") as f:
        #     torch.save(x, f)

    def inc(self):
        self.counter += 1