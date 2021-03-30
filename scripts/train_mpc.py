import argparse
import os
import time
import torch

from dm_control import suite
from algos.dreamer_mpc import DreamerMPC
from wrappers.action_repeat_wrapper import ActionRepeat
from wrappers.frame_stack_wrapper import FrameStack
from utils.logger import Logger
from utils.misc import make_dir, save_config
from wrappers.gym_wrapper import GymWrapper
from wrappers.pixel_observation_wrapper import PixelObservation
from utils.sampler import Sampler
from utils.sequence_replay_buffer import SequenceReplayBuffer


def parse_args():
    parser = argparse.ArgumentParser()

    # environment
    parser.add_argument('--domain_name', default='cheetah', type=str)
    parser.add_argument('--task_name', default='run', type=str)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--frame_stack', default=1, type=int)
    parser.add_argument('--action_repeat', default=1, type=int)
    parser.add_argument('--crop_center_observation', default=False, action='store_true')
    parser.add_argument('--resize_observation', default=False, action='store_true')
    parser.add_argument('--observation_size', default=64, type=int)
    parser.add_argument('--grayscale_observation', default=False, action='store_true')
    parser.add_argument('--normalize_observation', default=True, action='store_true')

    # replay buffer
    parser.add_argument('--replay_buffer_capacity', default=100000, type=int)
    parser.add_argument('--save_buffer', default=True, action='store_true')
    parser.add_argument('--load_buffer', default=False, action='store_true')
    parser.add_argument('--load_buffer_dir', default='../output/Pendulum-v0-03-03/buffer/replay_buffer.pkl', type=str)

    # train
    parser.add_argument('--init_episodes', default=3, type=int)
    parser.add_argument('--init_episode_length', default=100, type=int)
    parser.add_argument('--policy_episodes', default=1, type=int)
    parser.add_argument('--policy_episode_length', default=100, type=int)
    parser.add_argument('--training_iterations', default=1000, type=int)
    parser.add_argument('--model_iterations', default=2, type=int)
    parser.add_argument('--render_training', default=True, action='store_true')
    parser.add_argument('--batch_size', default=10, type=int)
    parser.add_argument('--chunk_size', default=50, type=int)
    parser.add_argument('--discount', default=0.99, type=float)
    parser.add_argument('--discount_lambda', default=0.95, type=float)
    parser.add_argument('--train_noise', default=0.3, type=float)
    parser.add_argument('--grad_clip', default=100.0, type=float)

    # evaluation
    parser.add_argument('--eval_freq', default=1, type=int)
    parser.add_argument('--eval_episodes', default=1, type=int)
    parser.add_argument('--eval_episode_length', default=100, type=int)
    parser.add_argument('--save_eval_video', default=False, action='store_true')
    parser.add_argument('--render_eval', default=True, action='store_true')

    # transition model
    parser.add_argument('--stochastic_size', default=30, type=int)
    parser.add_argument('--deterministic_size', default=200, type=int)
    parser.add_argument('--model_lr', default=6e-4, type=float)
    parser.add_argument('--free_nats', default=3, type=int)
    parser.add_argument('--kl_scale', default=1, type=int)
    parser.add_argument('--representation_loss', default='reconstruction', type=str)
    parser.add_argument('--random_crop_size', default=64, type=int)

    # reward model
    parser.add_argument('--reward_layers', default=3, type=int)
    parser.add_argument('--reward_hidden', default=300, type=int)

    # controller
    parser.add_argument('--controller_type', default='cem', type=str)
    parser.add_argument('--horizon', default=15, type=int)
    parser.add_argument('--num_control_samples', default=30, type=int)
    parser.add_argument('--max_iterations', default=3, type=int)
    parser.add_argument('--num_elites', default=20, type=int)

    # misc
    parser.add_argument('--work_dir', default='../output', type=str)
    parser.add_argument('--save_iter_model', default=False, action='store_true')
    parser.add_argument('--save_iter_model_freq', default=2, type=int)
    parser.add_argument('--load_model', default=False, action='store_true')
    parser.add_argument('--load_model_dir', default='../output/Pendulum-v0-03-03/model/test', type=str)
    parser.add_argument('--tensorboard_log', default=True, action='store_true')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    # create dm_control env
    env = suite.load(args.domain_name, args.task_name, task_kwargs={'random': args.seed})

    # wrap env to gym env
    env = GymWrapper(env)

    # augment observations by pixel values
    env = PixelObservation(
        env,
        crop_center_observation=args.crop_center_observation,
        resize_observation=args.resize_observation,
        observation_size=args.observation_size,
        grayscale_observation=args.grayscale_observation,
        normalize_observation=args.normalize_observation
    )

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
    video_dir = make_dir(os.path.join(args.work_dir, 'video'))
    model_dir = make_dir(os.path.join(args.work_dir, 'model'))
    buffer_dir = make_dir(os.path.join(args.work_dir, 'buffer'))
    config_dir = make_dir(os.path.join(args.work_dir, 'config'))
    tensorboard_dir = make_dir(os.path.join(args.work_dir, 'tensorboard'))

    # save training configuration
    save_config(config_dir, args)

    # initialize logger
    logger = Logger(tensorboard_dir)

    # initialize sampler
    sampler = Sampler(env)

    # initialize and preload replay buffer
    replay_buffer = SequenceReplayBuffer(args.replay_buffer_capacity)
    if args.load_buffer:
        replay_buffer.load(args.load_buffer_dir)

    # gpu settings
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # algorithm
    dreamer = DreamerMPC(
        logger,
        sampler,
        replay_buffer,
        device=device,
        tensorboard_log_freq=args.model_iterations,
        image_shape=env.observation_space.shape,
        action_shape=env.action_space.shape,
        reward_shape=(1,),
        stochastic_size=args.stochastic_size,
        deterministic_size=args.deterministic_size,
        reward_layers=args.reward_layers,
        reward_hidden=args.reward_hidden,
        model_lr=args.model_lr,
        grad_clip=args.grad_clip,
        free_nats=args.free_nats,
        kl_scale=args.kl_scale,
        action_repeat=args.action_repeat,
        representation_loss=args.representation_loss,
        random_crop_size=args.random_crop_size,
        train_noise=args.train_noise,
        controller_type=args.controller_type,
        action_space=env.action_space,
        horizon=args.horizon,
        num_control_samples=args.num_control_samples,
        max_iterations=args.max_iterations,
        num_elites=args.num_elites
    )

    # load model
    if args.load_model:
        dreamer.load_model(args.load_model_dir)

    # train model
    dreamer.train(
        init_episodes=args.init_episodes,
        init_episode_length=args.init_episode_length,
        policy_episodes=args.policy_episodes,
        policy_episode_length=args.policy_episode_length,
        training_iterations=args.training_iterations,
        model_iterations=args.model_iterations,
        batch_size=args.batch_size,
        chunk_size=args.chunk_size,
        render_training=args.render_training,
        save_iter_model=args.save_iter_model,
        save_iter_model_freq=args.save_iter_model_freq,
        model_dir=model_dir,
        eval_freq=args.eval_freq,
        eval_episodes=args.eval_episodes,
        eval_episode_length=args.eval_episode_length,
        render_eval=args.render_eval,
        save_eval_video=args.save_eval_video,
        video_dir=video_dir
    )

    # save training
    replay_buffer.save(buffer_dir, 'replay_buffer')
    dreamer.save_model(model_dir, 'model_final')

    # close environment
    env.close()


if __name__ == '__main__':
    main()
