#%matplotlib inline
#import matplotlib.pylab as plt
import torchvision.models as models
import torch.nn as nn
import torch
import numpy as np
import pandas as pd
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.optim as optim

from pruner import Pruner

class SparsityPruner(Pruner):
    def prune(self, to_retain):
        all_weights = np.array([])
        for name, param in self.model.named_parameters():
            if self.matches(name, self.layers): # only prune supplied layers
                active_weights = torch.abs(param[param != 0])
                active_weights = active_weights.view(-1).cpu().data.numpy()
                all_weights = np.concatenate((all_weights, active_weights))

        threshold = np.quantile(all_weights, 1-to_retain)
        for name, param in self.model.named_parameters():
            if self.matches(name, self.layers): # only prune supplied layers
                mask = (torch.abs(param) >= threshold).float()
                self.masks[name] = mask

                required_grad = param.requires_grad
                if required_grad:
                    param.requires_grad_(requires_grad=False)
                param.mul_(mask)
                if required_grad:
                    param.requires_grad_(requires_grad=True)

    def apply_mask(self):
            for name, param in self.model.named_parameters():
                if name not in self.masks:
                    continue
                mask = self.masks[name]
                required_grad = param.requires_grad
                if required_grad:
                    param.requires_grad_(requires_grad=False)
                param.mul_(mask)
                if required_grad:
                    param.requires_grad_(requires_grad=True)