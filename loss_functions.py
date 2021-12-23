import torch.nn as nn
import torch.nn.functional as F


def get_loss_function(name: str):
    if name == 'cross_entropy':
        return nn.CrossEntropyLoss()
    else:
        raise ValueError(f'Loss function {name} not available')
