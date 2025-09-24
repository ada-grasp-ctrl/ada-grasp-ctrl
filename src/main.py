import os
import sys
import random
import logging
import traceback

import hydra
from omegaconf import DictConfig
import numpy as np


sys.path.append(os.path.dirname(__file__))
from task import *

seed = 12
np.random.seed(seed)
random.seed(seed)


@hydra.main(config_path="../config", config_name="base", version_base=None)
def main(cfg: DictConfig):
    eval(f"task_{cfg.task_name}")(cfg)


if __name__ == "__main__":
    main()
