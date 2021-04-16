import torch
from torch import nn


class CURLModel(nn.Module):
    def __init__(self, device, z_dim, temperature=1., bilinear=False):
        super(CURLModel, self).__init__()
        self.device = device
        self.temperature = temperature
        self.bilinear = bilinear

        self.W = nn.Parameter(torch.rand(z_dim, z_dim))

    def info_nce_loss(self, z_a, z_pos):
        """
        Uses logits trick for CURL:
        - compute (B,B) matrix z_a (W z_pos.T)
        - positives are all diagonal elements
        - negatives are all other elements
        - to compute loss use multiclass cross entropy with identity matrix for labels
        """
        if self.bilinear:
            Wz = torch.matmul(self.W, z_pos.T)  # (z_dim,B)
            logits = torch.matmul(z_a, Wz)  # (B,B)
        else:
            logits = torch.matmul(z_a.T, z_pos)  # (B,B)
        logits = logits - torch.max(logits, 1)[0][:, None]
        labels = torch.arange(logits.shape[0]).long().to(self.device)
        logits = logits / self.temperature
        return logits, labels
