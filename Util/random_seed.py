"""
Author: YidaChen
Time is: 2023/9/9
this Code: 随机数种子固定
"""
import torch
import numpy as np
import random

def random_seed(seed=2023):
    """
    设置随机数，保证代码的可重复性
    :param seed:
    :return:
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False