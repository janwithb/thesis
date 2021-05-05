from torch import nn
from torch.nn import functional as F


class BYOLModel(nn.Module):
    def __init__(self, device, z_dim, act=F.elu):
        super(BYOLModel, self).__init__()
        self.z_dim = z_dim
        self.device = device
        self.act = act

        self.online_fc1 = nn.Linear(z_dim, z_dim)
        self.online_fc2 = nn.Linear(z_dim, z_dim)
        self.target_fc1 = nn.Linear(z_dim, z_dim)
        self.target_fc2 = nn.Linear(z_dim, z_dim)

        self.predictor_fc1 = nn.Linear(z_dim, z_dim)
        self.predictor_fc2 = nn.Linear(z_dim, z_dim)

    @staticmethod
    def _regression_loss(x, y):
        x = F.normalize(x, dim=1)
        y = F.normalize(y, dim=1)
        return 2 - 2 * (x * y).sum(dim=-1)

    def loss(self, z_one, z_two):
        z_one = self.act(self.online_fc1(z_one))
        z_one = self.act(self.online_fc2(z_one))
        target = self.act(self.target_fc1(z_two))
        target = self.act(self.target_fc2(target)).detach()
        pred = self.act(self.predictor_fc1(z_one))
        pred = self.act(self.predictor_fc2(pred))
        loss = self._regression_loss(pred, target)
        return loss.mean()
