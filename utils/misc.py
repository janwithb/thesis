import json
import os

import numpy as np
import torch
import torch.nn as nn

from typing import Iterable
from torch.nn import Module
from skimage.util.shape import view_as_windows


def make_dir(dir_path):
    try:
        os.mkdir(dir_path)
    except OSError:
        pass
    return dir_path


def save_config(config_dir, args):
    with open(os.path.join(config_dir, 'args.json'), 'w') as f:
        json.dump(vars(args), f, sort_keys=True, indent=4)


def weight_init(m):
    """Custom weight init for Conv2D and Linear layers."""
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.0)
    elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        assert m.weight.size(2) == m.weight.size(3)
        m.weight.data.fill_(0.0)
        m.bias.data.fill_(0.0)
        mid = m.weight.size(2) // 2
        gain = nn.init.calculate_gain('relu')
        nn.init.orthogonal_(m.weight.data[:, :, mid, mid], gain)


def get_parameters(modules: Iterable[Module]):
    """
    Given a list of torch modules, returns a list of their parameters.
    :param modules: iterable of modules
    :returns: a list of parameters
    """
    model_parameters = []
    for module in modules:
        model_parameters += list(module.parameters())
    return model_parameters


def conv_out(h_in, padding, kernel_size, stride):
    return int((h_in + 2. * padding - (kernel_size - 1.) - 1.) / stride + 1.)


def output_padding(h_in, conv_out, padding, kernel_size, stride):
    return h_in - (conv_out - 1) * stride + 2 * padding - (kernel_size - 1) - 1


def conv_out_shape(h_in, padding, kernel_size, stride):
    return tuple(conv_out(x, padding, kernel_size, stride) for x in h_in)


def output_padding_shape(h_in, conv_out, padding, kernel_size, stride):
    return tuple(output_padding(h_in[i], conv_out[i], padding, kernel_size, stride) for i in range(len(h_in)))


def infer_leading_dims(tensor, dim):
    """Looks for up to two leading dimensions in ``tensor``,infer_leading_dims before
    the data dimensions, of which there are assumed to be ``dim`` number.
    For use at beginning of model's ``forward()`` method, which should
    finish with ``restore_leading_dims()`` (see that function for help.)
    Returns:
    lead_dim: int --number of leading dims found.
    T: int --size of first leading dim, if two leading dims, o/w 1.
    B: int --size of first leading dim if one, second leading dim if two, o/w 1.
    shape: tensor shape after leading dims.
    """
    lead_dim = tensor.dim() - dim
    assert lead_dim in (0, 1, 2)
    if lead_dim == 2:
        B, T = tensor.shape[:2]
    else:
        T = 1
        B = 1 if lead_dim == 0 else tensor.shape[0]
    shape = tensor.shape[lead_dim:]
    return lead_dim, T, B, shape


class FreezeParameters:
    def __init__(self, modules: Iterable[Module]):
        """
        Context manager to locally freeze gradients.
        In some cases with can speed up computation because gradients aren't calculated for these listed modules.
        example:
        ```
        with FreezeParameters([module]):
            output_tensor = module(input_tensor)
        ```
        :param modules: iterable of modules. used to call .parameters() to freeze gradients.
        """
        self.modules = modules
        self.param_states = [p.requires_grad for p in get_parameters(self.modules)]

    def __enter__(self):
        for param in get_parameters(self.modules):
            param.requires_grad = False

    def __exit__(self, exc_type, exc_val, exc_tb):
        for i, param in enumerate(get_parameters(self.modules)):
            param.requires_grad = self.param_states[i]


def compute_return(reward: torch.Tensor,
                   value: torch.Tensor,
                   discount: torch.Tensor,
                   bootstrap: torch.Tensor,
                   lambda_: float):
    """
    Compute the discounted reward for a batch of data.
    reward, value, and discount are all shape [horizon - 1, batch, 1] (last element is cut off)
    Bootstrap is [batch, 1]
    """
    next_values = torch.cat([value[:, 1:], bootstrap[:, None]], 1)
    target = reward + discount * next_values * (1 - lambda_)
    timesteps = list(range(reward.shape[1] - 1, -1, -1))
    outputs = []
    accumulated_reward = bootstrap
    for t in timesteps:
        inp = target[:, t]
        discount_factor = discount[:, t]
        accumulated_reward = inp + discount_factor * lambda_ * accumulated_reward
        outputs.append(accumulated_reward)
    returns = torch.flip(torch.stack(outputs), [0])
    returns = torch.transpose(returns, 0, 1)
    return returns


def mlp(input_dim, hidden_dim, output_dim, hidden_depth, output_mod=None):
    if hidden_depth == 0:
        mods = [nn.Linear(input_dim, output_dim)]
    else:
        mods = [nn.Linear(input_dim, hidden_dim), nn.ReLU(inplace=True)]
        for i in range(hidden_depth - 1):
            mods += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU(inplace=True)]
        mods.append(nn.Linear(hidden_dim, output_dim))
    if output_mod is not None:
        mods.append(output_mod)
    trunk = nn.Sequential(*mods)
    return trunk


def soft_update_params(net, target_net, tau):
    for param, target_param in zip(net.parameters(), target_net.parameters()):
        target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)


def random_crop(imgs, output_size):
    n = imgs.shape[:2]
    batch_size = np.prod(n)
    imgs = np.reshape(imgs, (batch_size,) + imgs.shape[2:])
    img_size = imgs.shape[-1]
    crop_max = img_size - output_size
    w1 = np.random.randint(0, crop_max, batch_size)
    h1 = np.random.randint(0, crop_max, batch_size)

    # creates all sliding windows combinations of size (output_size)
    windows = view_as_windows(imgs, (1, 1, output_size, output_size))[..., 0, 0, :, :]

    # selects a random window for each batch element
    cropped_imgs = windows[np.arange(batch_size), :, w1, h1]
    cropped_imgs = np.reshape(cropped_imgs, n + cropped_imgs.shape[1:])
    return cropped_imgs


def compute_logits(z_a, z_pos, z_dim):
    n = z_a.shape[:2]
    batch_size = np.prod(n)
    z_a = torch.reshape(z_a, (batch_size,) + z_a.shape[2:])
    z_pos = torch.reshape(z_pos, (batch_size,) + z_pos.shape[2:])
    W = nn.Parameter(torch.rand(z_dim, z_dim))
    Wz = torch.matmul(W, z_pos.T)  # (z_dim,B)
    logits = torch.matmul(z_a, Wz)  # (B,B)
    logits = logits - torch.max(logits, 1)[0][:, None]
    return logits

def lambda_target(rewards, values, gamma, lambda_):
    """
    Compute lambda target of value function
    rewards and values should be 2D-tensor and same size,
    and first-dimension means time step
    gamma is discount factor and lambda_ is weight to compute lambda target
    """
    V_lambda = torch.zeros_like(rewards, device=rewards.device)

    H = rewards.shape[0] - 1
    V_n = torch.zeros_like(rewards, device=rewards.device)
    V_n[H] = values[H]
    for n in range(1, H+1):
        # compute n-step target
        # NOTE: If it hits the end, compromise with the largest possible n-step return
        V_n[:-n] = (gamma ** n) * values[n:]
        for k in range(1, n+1):
            if k == n:
                V_n[:-n] += (gamma ** (n-1)) * rewards[k:]
            else:
                V_n[:-n] += (gamma ** (k-1)) * rewards[k:-n+k]

        # add lambda_ weighted n-step target to compute lambda target
        if n == H:
            V_lambda += (lambda_ ** (H-1)) * V_n
        else:
            V_lambda += (1 - lambda_) * (lambda_ ** (n-1)) * V_n

    return V_lambda