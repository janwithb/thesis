import torch

from torch import nn
from torch.nn import functional as F


class CURLModel(nn.Module):
    """
    Model for contrastive unsupervised representation learning (CURL).
    """
    def __init__(self, device, x_dim, z_dim, similarity='dot_product', temperature=1., act=F.elu):
        super(CURLModel, self).__init__()
        self.x_dim = x_dim
        self.z_dim = z_dim
        self.device = device
        self.similarity = similarity
        self.temperature = temperature
        self.act = act

        self.W = nn.Parameter(torch.rand(z_dim, z_dim))
        self.qth_fc1 = nn.Linear(z_dim, z_dim)
        self.qth_fc2 = nn.Linear(z_dim, z_dim)
        self.kth_fc1 = nn.Linear(x_dim, x_dim)
        self.kth_fc2 = nn.Linear(x_dim, z_dim)

    def info_nce_loss(self, z_a, z_pos):
        # transform heads
        if self.x_dim != self.z_dim:
            z_a = self.act(self.qth_fc1(z_a))
            z_a = self.act(self.qth_fc2(z_a))
            z_pos = self.act(self.kth_fc1(z_pos))
            z_pos = self.act(self.kth_fc2(z_pos))

        # measure similarities
        if self.similarity == 'dot_product':
            logits = torch.matmul(z_a.T, z_pos)
        elif self.similarity == 'bilinear_product':
            Wz = torch.matmul(self.W, z_pos.T)
            logits = torch.matmul(z_a, Wz)
        elif self.similarity == 'cosine':
            logits = torch.matmul(z_a.T, z_pos)
            logits = logits / (torch.norm(z_a) * torch.norm(z_pos))
            logits = logits / self.temperature

        logits = logits - torch.max(logits, 1)[0][:, None]
        labels = torch.arange(logits.shape[0]).long().to(self.device)
        return logits, labels
