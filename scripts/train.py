import argparse
import os
import time
import gym

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
    parser.add_argument('--preprocessing', default=True, action='store_true')
    parser.add_argument('--observation_size', default=100, type=int)

    # replay buffer
    parser.add_argument('--replay_buffer_capacity', default=100000, type=int)
    parser.add_argument('--save_buffer', default=False, action='store_true')
    parser.add_argument('--load_buffer', default=False, action='store_true')

    # train
    parser.add_argument('--init_episodes', default=5, type=int)
    parser.add_argument('--render_training', default=False, action='store_true')

    # misc
    parser.add_argument('--work_dir', default='..', type=str)
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
        preprocessing=args.preprocessing,
        observation_size=args.observation_size
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

    # collect initial episodes
    episodes = []
    for ep in range(args.init_episodes):
        observation, done, t = env.reset(), False, 0
        starting_state = observation
        observations, actions, rewards = [], [], []
        while not done:
            action = env.action_space.sample()
            next_observation, reward, done, info = env.step(action)
            observations.append(next_observation)
            actions.append(action)
            rewards.append(reward)
            if args.render_training:
                env.unwrapped.render()
            video_recorder.capture_frame()
            t += 1
        episode = Episode(observations, actions, rewards, starting_state)
        episodes.append(episode)
    replay_buffer.add(episodes)

    # save training
    replay_buffer.save(buffer_dir)

    # close components
    video_recorder.close()
    env.close()


if __name__ == '__main__':
    main()
