import argparse
import os
import time
import gym

from algos.dreamer import Dreamer
from utils.frame_stack_wrapper import FrameStack
from utils.pixel_observation_wrapper import PixelObservationWrapper
from utils.sequence_replay_buffer import Episode, SequenceReplayBuffer
from utils.video_recorder import VideoRecorder
from utils.utils import make_dir, save_config


def parse_args():
    parser = argparse.ArgumentParser()

    # environment
    parser.add_argument('--env_name', default='Pendulum-v0')
    parser.add_argument('--frame_stack', default=3, type=int)
    parser.add_argument('--crop_center_observation', default=True, action='store_true')
    parser.add_argument('--resize_observation', default=True, action='store_true')
    parser.add_argument('--observation_size', default=64, type=int)
    parser.add_argument('--grayscale_observation', default=True, action='store_true')
    parser.add_argument('--normalize_observation', default=True, action='store_true')

    # replay buffer
    parser.add_argument('--replay_buffer_capacity', default=100000, type=int)
    parser.add_argument('--save_buffer', default=False, action='store_true')
    parser.add_argument('--load_buffer', default=True, action='store_true')

    # train
    parser.add_argument('--init_episodes', default=0, type=int)
    parser.add_argument('--init_episode_length', default=100, type=int)
    parser.add_argument('--training_iterations', default=50, type=int)
    parser.add_argument('--render_training', default=False, action='store_true')
    parser.add_argument('--batch_size', default=5, type=int)
    parser.add_argument('--chunk_size', default=50, type=int)

    # encoder / decoder
    parser.add_argument('--encoder_feature_dim', default=50, type=int)
    parser.add_argument('--encoder_lr', default=1e-3, type=float)
    parser.add_argument('--encoder_tau', default=0.05, type=float)
    parser.add_argument('--decoder_lr', default=1e-3, type=float)
    parser.add_argument('--decoder_update_freq', default=1, type=int)
    parser.add_argument('--decoder_latent_lambda', default=1e-6, type=float)
    parser.add_argument('--decoder_weight_lambda', default=1e-7, type=float)
    parser.add_argument('--num_layers', default=4, type=int)
    parser.add_argument('--num_filters', default=32, type=int)

    # rssm
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

    # value model
    parser.add_argument('--value_lr', default=8e-5, type=float)
    parser.add_argument('--value_layers', default=3, type=int)
    parser.add_argument('--value_hidden', default=200, type=int)

    # misc
    parser.add_argument('--work_dir', default='../output', type=str)
    parser.add_argument('--save_video', default=False, action='store_true')
    parser.add_argument('--save_model', default=False, action='store_true')
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
    ts = time.strftime("%m-%d", ts)
    exp_name = args.env_name + '-' + ts
    args.work_dir = args.work_dir + '/' + exp_name
    make_dir(args.work_dir)

    # make other directories
    video_dir = make_dir(os.path.join(args.work_dir, 'video'))
    model_dir = make_dir(os.path.join(args.work_dir, 'model'))
    buffer_dir = make_dir(os.path.join(args.work_dir, 'buffer'))
    config_dir = make_dir(os.path.join(args.work_dir, 'config'))

    # initialize video recorder
    video_recorder = VideoRecorder(
        env,
        base_path=video_dir + '/video',
        enabled=args.save_video
    )

    # save training configuration
    save_config(config_dir, args)

    # replay buffer
    replay_buffer = SequenceReplayBuffer(args.replay_buffer_capacity)
    if args.load_buffer:
        replay_buffer.load(buffer_dir)

    # algorithm
    dreamer = Dreamer(
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
        free_nats=args.free_nats,
        kl_scale=args.kl_scale,
        value_shape=(1,),
        value_layers=args.value_layers,
        value_hidden=args.value_hidden
    )

    # collect initial episodes
    episodes = []
    for ep in range(args.init_episodes):
        observation, done = env.reset(), False
        starting_state = observation
        observations, actions, rewards = [], [], []
        for t in range(args.init_episode_length):
            action = env.action_space.sample()
            next_observation, reward, done, info = env.step(action)
            observations.append(next_observation)
            actions.append(action)
            rewards.append(reward)
            if args.render_training:
                env.unwrapped.render()
            video_recorder.capture_frame()
        episode = Episode(observations, actions, rewards, starting_state)
        episodes.append(episode)
    replay_buffer.add(episodes)

    for it in range(args.training_iterations):
        samples = replay_buffer.get_chunk_batch(args.batch_size, args.chunk_size)
        dreamer.optimize_model(samples)

    # save training
    replay_buffer.save(buffer_dir)

    # close components
    video_recorder.close()
    env.close()


if __name__ == '__main__':
    main()
