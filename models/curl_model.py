import torch

from torch import nn
from torch.nn import functional as F


class CURLModel(nn.Module):
    def __init__(self, device, x_dim, z_dim, temperature=1., bilinear=False, act=F.elu):
        super(CURLModel, self).__init__()
        self.x_dim = x_dim
        self.z_dim = z_dim
        self.device = device
        self.temperature = temperature
        self.bilinear = bilinear
        self.act = act

        self.W = nn.Parameter(torch.rand(z_dim, z_dim))
        self.qth_fc1 = nn.Linear(z_dim, z_dim)
        self.qth_fc2 = nn.Linear(z_dim, z_dim)
        self.kth_fc1 = nn.Linear(x_dim, x_dim)
        self.kth_fc2 = nn.Linear(x_dim, z_dim)

    def info_nce_loss(self, z_a, z_pos):
        """
        Uses logits trick for CURL:
        - compute (B,B) matrix z_a (W z_pos.T)
        - positives are all diagonal elements
        - negatives are all other elements
        - to compute loss use multiclass cross entropy with identity matrix for labels
        """
        if self.x_dim != self.z_dim:
            z_a = self.act(self.qth_fc1(z_a))
            z_a = self.act(self.qth_fc2(z_a))
            z_pos = self.act(self.kth_fc1(z_pos))
            z_pos = self.act(self.kth_fc2(z_pos))
        if self.bilinear:
            Wz = torch.matmul(self.W, z_pos.T)  # (z_dim,B)
            logits = torch.matmul(z_a, Wz)  # (B,B)
        else:
            logits = torch.matmul(z_a.T, z_pos)  # (B,B)
        logits = logits - torch.max(logits, 1)[0][:, None]
        labels = torch.arange(logits.shape[0]).long().to(self.device)
        logits = logits / self.temperature
        return logits, labels
