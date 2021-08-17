import json
import os

import torch
import torch.nn as nn
import kornia


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


def center_crop_image(image, output_size, batch=False):
    if not batch:
        h, w = image.shape[1:]
    else:
        h, w = image.shape[3:]
    new_h, new_w = output_size, output_size

    top = (h - new_h)//2
    left = (w - new_w)//2

    image = image[..., top:top + new_h, left:left + new_w]
    return image


def augument_image(image):
    transforms = nn.Sequential(
        nn.ReplicationPad2d(4),
        kornia.augmentation.RandomCrop(size=(64, 64)),
        kornia.augmentation.ColorJitter(0.2, 0.3, 0.2, 0.3),
        #kornia.augmentation.RandomErasing(),
        #kornia.augmentation.RandomHorizontalFlip(),
        #kornia.augmentation.RandomVerticalFlip(),
        #kornia.augmentation.RandomRotation(degrees=5.0)
    )
    augumented_image = transforms(image + 0.5)
    return augumented_image - 0.5


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


def conv_out(h_in, padding, kernel_size, stride):
    return int((h_in + 2. * padding - (kernel_size - 1.) - 1.) / stride + 1.)


def conv_out_shape(h_in, padding, kernel_size, stride):
    return tuple(conv_out(x, padding, kernel_size, stride) for x in h_in)
