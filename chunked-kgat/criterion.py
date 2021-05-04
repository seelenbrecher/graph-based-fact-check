import torch
import torch.nn as nn
import torch.nn.functional as F

class Criterion:
    def __init__(self, args):
        self.criterion = args.criterion
        self.batch = args.gradient_accumulation_steps
    
    def calculate(self, probs, all_probs, targets):
        if self.criterion == 'standard':
            return F.nll_loss(probs, targets)
        elif self.criterion == 'label_smoothing':
            N = all_probs.shape[0]
            return -torch.sum(all_probs * targets) / self.batch