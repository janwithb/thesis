import argparse
import os
import time
import gym
import torch

from tqdm import tqdm
from algos.dreamer import Dreamer
from utils.frame_stack_wrapper import FrameStack
from utils.logger import Logger
from utils.misc import make_dir, save_config
from utils.pixel_observation_wrapper import PixelObservationWrapper
from utils.sampler import Sampler
from utils.sequence_replay_buffer import SequenceReplayBuffer


def parse_args():
    parser = argparse.ArgumentParser()

    # environment
    parser.add_argument('--env_name', default='Pendulum-v0', type=str)
    parser.add_argument('--frame_stack', default=1, type=int)
    parser.add_argument('--crop_center_observation', default=True, action='store_true')
    parser.add_argument('--resize_observation', default=True, action='store_true')
    parser.add_argument('--observation_size', default=64, type=int)
    parser.add_argument('--grayscale_observation', default=False, action='store_true')
    parser.add_argument('--normalize_observation', default=True, action='store_true')

    # replay buffer
    parser.add_argument('--replay_buffer_capacity', default=100000, type=int)
    parser.add_argument('--save_buffer', default=True, action='store_true')
    parser.add_argument('--load_buffer', default=False, action='store_true')
    parser.add_argument('--load_buffer_dir', default='../output/Pendulum-v0-03-03/buffer/replay_buffer.pkl', type=str)

    # train
    parser.add_argument('--init_episodes', default=5, type=int)
    parser.add_argument('--init_episode_length', default=100, type=int)
    parser.add_argument('--iter_episodes', default=10, type=int)
    parser.add_argument('--iter_episode_length', default=100, type=int)
    parser.add_argument('--training_iterations', default=20, type=int)
    parser.add_argument('--imagination_iterations', default=4, type=int)
    parser.add_argument('--render_training', default=False, action='store_true')
    parser.add_argument('--batch_size', default=2, type=int)
    parser.add_argument('--chunk_size', default=50, type=int)
    parser.add_argument('--imagine_horizon', default=15, type=int)
    parser.add_argument('--discount', default=0.99, type=float)
    parser.add_argument('--discount_lambda', default=0.95, type=float)
    parser.add_argument('--train_noise', default=0.3, type=float)
    parser.add_argument('--expl_method', default='additive_gaussian', type=str)
    parser.add_argument('--expl_amount', default=0.3, type=float)
    parser.add_argument('--expl_decay', default=0.0, type=float)
    parser.add_argument('--expl_min', default=0.0, type=float)
    parser.add_argument('--grad_clip', default=100.0, type=float)

    # evaluation
    parser.add_argument('--eval_freq', default=1, type=int)
    parser.add_argument('--eval_episodes', default=5, type=int)
    parser.add_argument('--eval_episode_length', default=100, type=int)
    parser.add_argument('--save_eval_video', default=False, action='store_true')
    parser.add_argument('--render_eval', default=False, action='store_true')
    parser.add_argument('--eval_noise', default=0.0, type=float)

    # transition model
    parser.add_argument('--stochastic_size', default=30, type=int)
    parser.add_argument('--deterministic_size', default=200, type=int)
    parser.add_argument('--model_lr', default=6e-4, type=float)
    parser.add_argument('--free_nats', default=3, type=int)
    parser.add_argument('--kl_scale', default=1, type=int)

    # reward model
    parser.add_argument('--reward_layers', default=3, type=int)
    parser.add_argument('--reward_hidden', default=300, type=int)

    # actor model
    parser.add_argument('--actor_lr', default=8e-5, type=float)
    parser.add_argument('--actor_layers', default=3, type=int)
    parser.add_argument('--actor_hidden', default=200, type=int)
    parser.add_argument('--actor_dist', default='tanh_normal', type=str)

    # value model
    parser.add_argument('--value_lr', default=8e-5, type=float)
    parser.add_argument('--value_layers', default=3, type=int)
    parser.add_argument('--value_hidden', default=200, type=int)

    # misc
    parser.add_argument('--work_dir', default='../output', type=str)
    parser.add_argument('--save_iter_video', default=False, action='store_true')
    parser.add_argument('--save_iter_video_freq', default=2, type=int)
    parser.add_argument('--save_iter_model', default=False, action='store_true')
    parser.add_argument('--save_iter_model_freq', default=2, type=int)
    parser.add_argument('--load_model', default=False, action='store_true')
    parser.add_argument('--load_model_dir', default='../output/Pendulum-v0-03-03/model/test', type=str)
    parser.add_argument('--tensorboard_log', default=True, action='store_true')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    # create gym environment
    env = gym.make(args.env_name)

    # augment observations by pixel values
    env = PixelObservationWrapper(
        env,
        crop_center_observation=args.crop_center_observation,
        resize_observation=args.resize_observation,
        observation_size=args.observation_size,
        grayscale_observation=args.grayscale_observation,
        normalize_observation=args.normalize_observation
    )

    # stack several consecutive frames together
    env = FrameStack(env, args.frame_stack)

    # make work directory
    ts = time.gmtime()
    ts = time.strftime("%Y-%m-%d-%H-%M-%S", ts)
    exp_name = args.env_name + '-' + ts
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

    # initialize and preload replay buffer
    replay_buffer = SequenceReplayBuffer(args.replay_buffer_capacity)
    if args.load_buffer:
        replay_buffer.load(args.load_buffer_dir)

    # initialize logger
    logger = Logger(tensorboard_dir, tensorboard_log=args.tensorboard_log)

    # gpu settings
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # algorithm
    dreamer = Dreamer(
        logger,
        device=device,
        tensorboard_log_freq=args.imagination_iterations,
        image_shape=env.observation_space.shape,
        action_shape=env.action_space.shape,
        reward_shape=(1,),
        stochastic_size=args.stochastic_size,
        deterministic_size=args.deterministic_size,
        reward_layers=args.reward_layers,
        reward_hidden=args.reward_hidden,
        model_lr=args.model_lr,
        actor_lr=args.actor_lr,
        value_lr=args.value_lr,
        grad_clip=args.grad_clip,
        free_nats=args.free_nats,
        kl_scale=args.kl_scale,
        value_shape=(1,),
        value_layers=args.value_layers,
        value_hidden=args.value_hidden,
        actor_layers=args.actor_layers,
        actor_hidden=args.actor_hidden,
        actor_dist=args.actor_dist,
        imagine_horizon=args.imagine_horizon,
        discount=args.discount,
        discount_lambda=args.discount_lambda,
        train_noise=args.train_noise,
        eval_noise=args.eval_noise,
        expl_method=args.expl_method,
        expl_amount=args.expl_amount,
        expl_decay=args.expl_decay,
        expl_min=args.expl_min
    )

    # initialize sampler
    sampler = Sampler(
        env,
        dreamer,
        replay_buffer=replay_buffer
    )

    # load model
    if args.load_model:
        dreamer.load_model(args.load_model_dir)

    # collect initial episodes
    sampler.collect_episodes(args.init_episodes, args.init_episode_length, random=True, render=args.render_training)

    # main training loop
    for it in tqdm(range(args.training_iterations), desc='Training progress'):
        itr_start_time = time.time()
        dreamer.training = True
        dreamer.eval = False

        # model training loop
        for img_it in tqdm(range(args.imagination_iterations), desc='Imagination progress'):
            # save video frequently
            if args.save_iter_video and img_it % args.save_iter_video_freq == 0:
                sampler.reset_video_recorder(video_dir, 'video_iter' + str(it))

            samples = replay_buffer.get_chunk_batch(args.batch_size, args.chunk_size)
            dreamer.optimize(samples)
            dreamer.img_itr += 1

        # collect new data
        sampler.collect_episodes(args.iter_episodes, args.iter_episode_length, render=args.render_training)

        # save model frequently
        if args.save_iter_model and it % args.save_iter_model_freq == 0:
            dreamer.save_model(model_dir, 'model_iter_' + str(it))

        itr_time = time.time() - itr_start_time
        logger.log('train/itr_time', itr_time, dreamer.step)
        dreamer.itr += 1

        # evaluate policy
        if it % args.eval_freq == 0:
            eval_start_time = time.time()
            dreamer.training = False
            dreamer.eval = True
            if args.save_eval_video:
                sampler.reset_video_recorder(video_dir, 'video_eval' + str(it))
            with torch.no_grad():
                episodes = sampler.collect_episodes(args.eval_episodes, args.eval_episode_length, save_episodes=False,
                                                    render=args.render_eval)
            dreamer.evaluate(episodes)
            eval_time = time.time() - eval_start_time
            logger.log('eval/eval_time', eval_time, dreamer.step)

    # save training
    replay_buffer.save(buffer_dir, 'replay_buffer')
    dreamer.save_model(model_dir, 'model_final')

    # close environment
    env.close()


if __name__ == '__main__':
    main()
