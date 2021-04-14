import argparse
import os
import time
import torch
import numpy as np
import dmc_remastered as dmcr

from dm_control import suite
from algos.dreamer_mpc import DreamerMPC
from wrappers.action_repeat_wrapper import ActionRepeat
from wrappers.frame_stack_wrapper import FrameStack
from utils.logger import Logger
from utils.misc import make_dir, save_config
from wrappers.gym_wrapper import GymWrapper
from wrappers.pixel_observation_wrapper import PixelObservation
from utils.sequence_replay_buffer import SequenceReplayBuffer

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def parse_args():
    parser = argparse.ArgumentParser()

    # environment
    parser.add_argument('--domain_name', default='cheetah', type=str)
    parser.add_argument('--task_name', default='run', type=str)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--randomize_env', default=True, action='store_true')
    parser.add_argument('--observation_size', default=84, type=int)
    parser.add_argument('--frame_stack', default=1, type=int)
    parser.add_argument('--action_repeat', default=4, type=int)

    # replay buffer
    parser.add_argument('--replay_buffer_capacity', default=1000000, type=int)
    parser.add_argument('--save_buffer', default=False, action='store_true')
    parser.add_argument('--load_buffer', default=False, action='store_true')
    parser.add_argument('--load_buffer_dir', default='', type=str)

    # train
    parser.add_argument('--init_episodes', default=1, type=int)
    parser.add_argument('--agent_episodes', default=1, type=int)
    parser.add_argument('--training_iterations', default=1000, type=int)
    parser.add_argument('--model_iterations', default=1, type=int)
    parser.add_argument('--render_training', default=True, action='store_true')
    parser.add_argument('--batch_size', default=50, type=int)
    parser.add_argument('--chunk_length', default=50, type=int)
    parser.add_argument('--grad_clip', default=100.0, type=float)

    # evaluation
    parser.add_argument('--eval_freq', default=10, type=int)
    parser.add_argument('--eval_episodes', default=1, type=int)

    # model
    parser.add_argument('--stochastic_size', default=30, type=int)
    parser.add_argument('--deterministic_size', default=200, type=int)
    parser.add_argument('--reward_hidden_dim', default=300, type=int)
    parser.add_argument('--model_lr', default=1e-3, type=float)
    parser.add_argument('--model_eps', default=1e-4, type=float)
    parser.add_argument('--free_nats', default=3, type=int)
    parser.add_argument('--kl_scale', default=1, type=int)
    parser.add_argument('--image_loss_type', default='contrastive', type=str)

    # agent
    parser.add_argument('--controller_type', default='random_shooting', type=str)
    parser.add_argument('--horizon', default=5, type=int)
    parser.add_argument('--num_control_samples', default=5, type=int)
    parser.add_argument('--max_iterations', default=10, type=int)
    parser.add_argument('--num_elites', default=100, type=int)
    parser.add_argument('--exploration_noise_var', type=float, default=0.3)

    # misc
    parser.add_argument('--work_dir', default='../output', type=str)
    parser.add_argument('--save_iter_model', default=False, action='store_true')
    parser.add_argument('--save_iter_model_freq', default=2, type=int)
    parser.add_argument('--load_model', default=False, action='store_true')
    parser.add_argument('--load_model_dir', default='', type=str)
    parser.add_argument('--full_tb_log', default=False, action='store_true')
    parser.add_argument('--model_log_freq', default=10, type=int)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    # create dm_control env
    if args.randomize_env:
        _, env = dmcr.benchmarks.visual_generalization(args.domain_name, args.task_name, num_levels=100)
    else:
        env = suite.load(args.domain_name, args.task_name, task_kwargs={'random': args.seed})
        env = GymWrapper(env)

    # augment observations by pixel values
    env = PixelObservation(env, args.observation_size)

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
    buffer_dir = make_dir(os.path.join(args.work_dir, 'buffer'))
    config_dir = make_dir(os.path.join(args.work_dir, 'config'))
    tensorboard_dir = make_dir(os.path.join(args.work_dir, 'tensorboard'))

    # save training configuration
    save_config(config_dir, args)

    # initialize logger
    logger = Logger(tensorboard_dir)

    # initialize and preload replay buffer
    args.observation_shape = env.observation_space.shape
    args.action_dim = env.action_space.shape[0]
    replay_buffer = SequenceReplayBuffer(args.replay_buffer_capacity,
                                         args.observation_shape,
                                         args.action_dim)
    if args.load_buffer:
        replay_buffer.load(args.load_buffer_dir)

    # gpu settings
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # set seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    # algorithm
    algorithm = DreamerMPC(env, logger, replay_buffer, device, args)

    # load model
    if args.load_model:
        algorithm.load_model(args.load_model_dir)

    # train model
    algorithm.train()

    # save training
    replay_buffer.save(buffer_dir, 'replay_buffer')
    algorithm.save_model(model_dir, 'model_final')

    # close environment
    env.close()


if __name__ == '__main__':
    main()
