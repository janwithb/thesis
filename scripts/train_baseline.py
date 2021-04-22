import argparse
import os
import time
import dmc_remastered as dmcr
import torch

from stable_baselines3 import SAC
from dm_control import suite
from wrappers.action_repeat_wrapper import ActionRepeat
from wrappers.frame_stack_wrapper import FrameStack
from utils.misc import make_dir, save_config
from wrappers.gym_wrapper import GymWrapper
from wrappers.pixel_observation_wrapper import PixelObservation

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def parse_args():
    parser = argparse.ArgumentParser()

    # config
    parser.add_argument('--domain_name', default='cartpole', type=str)
    parser.add_argument('--task_name', default='swingup', type=str)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--randomize_env', default=False, action='store_true')
    parser.add_argument('--observation_type', default='pixel', type=str)
    parser.add_argument('--observation_size', default=64, type=int)
    parser.add_argument('--frame_stack', default=4, type=int)
    parser.add_argument('--action_repeat', default=4, type=int)
    parser.add_argument('--work_dir', default='../output', type=str)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    # create dm_control env
    if args.randomize_env:
        _, env = dmcr.benchmarks.visual_generalization(args.domain_name, args.task_name, num_levels=100, frame_stack=1)
    else:
        env = suite.load(args.domain_name, args.task_name, task_kwargs={'random': args.seed})
        env = GymWrapper(env)

    # augment observations by pixel values
    if args.observation_type == 'pixel':
        env = PixelObservation(env, args.observation_size, normalize=False)
    elif args.observation_type == 'state':
        assert args.randomize_env is False

    # stack several consecutive frames together
    env = FrameStack(env, args.frame_stack)

    # repeat actions
    env = ActionRepeat(env, args.action_repeat)

    # make work directory
    ts = time.gmtime()
    ts = time.strftime("%Y-%m-%d-%H-%M-%S", ts)
    exp_name = args.domain_name + '-' + args.task_name + '-' + ts
    args.work_dir = args.work_dir + '/' + exp_name
    make_dir(args.work_dir)

    # make other directories
    model_dir = make_dir(os.path.join(args.work_dir, 'model'))
    config_dir = make_dir(os.path.join(args.work_dir, 'config'))
    tensorboard_dir = make_dir(os.path.join(args.work_dir, 'tensorboard'))

    # save training configuration
    save_config(config_dir, args)

    # gpu settings
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if args.observation_type == 'pixel':
        model = SAC('CnnPolicy', env, verbose=1, tensorboard_log=tensorboard_dir, device=device, buffer_size=50000)
    elif args.observation_type == 'state':
        model = SAC('MlpPolicy', env, verbose=1, tensorboard_log=tensorboard_dir, device=device)
    model.learn(total_timesteps=10000000, log_interval=1)
    model.save(os.path.join(model_dir, 'model_final'))

    # close environment
    env.close()


if __name__ == '__main__':
    main()
