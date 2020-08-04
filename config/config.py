import os
from types import SimpleNamespace


class modelParams:
    def __init__(self):
        self.ckpt_path = os.environ.get("CKPT_PATH")
        self.ckpt_file = os.environ.get("CKPT_FILE")

model_params = modelParams()