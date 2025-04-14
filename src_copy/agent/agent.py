
import wandb
import gymnasium as gym
import argparse
import matplotlib.pyplot as plt
import torch.optim as optim
from utils.schedule import LinearSchedule
from algorithms.DQN import DQNTrainer, OptimizerSpec
from utils.ReplayBuffer import ReplayBuffer
import ale_py
from gymnasium.wrappers import RecordEpisodeStatistics, RecordVideo

gym.register_envs(ale_py)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run RL experiment with WandB logging.")

    # Environment & Training
    parser.add_argument("--num_timesteps", type=int, default=1000000, help="Total number of timesteps for training")
    parser.add_argument("--replay_buffer_size", type=int, default=100000, help="Size of the replay buffer")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--learning_starts", type=int, default=50000, help="Timesteps before training starts")
    parser.add_argument("--learning_freq", type=int, default=4, help="Frequency of model updates")
    parser.add_argument("--frame_history_len", type=int, default=4, help="Number of past frames used as input")
    parser.add_argument("--target_update_freq", type=int, default=10000, help="Frequency of target network updates")

    # Optimizer Parameters
    parser.add_argument("--learning_rate", type=float, default=0.00025, help="Learning rate for optimizer")
    parser.add_argument("--alpha", type=float, default=0.95, help="Alpha value for RMSprop optimizer")
    parser.add_argument("--eps", type=float, default=1e-2, help="Epsilon value for RMSprop optimizer")

    # DQN Hyperparameters
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor for future rewards")

    # Exploration Schedule
    parser.add_argument("--exploration_final_eps", type=float, default=0.1, help="Final epsilon for exploration")
    parser.add_argument("--exploration_schedule_steps", type=int, default=1000000, help="Steps for epsilon decay")

    args = parser.parse_args()


    wandb.login(key="272a6329f05b645580139131ec5b1eb14bb27769")
    wandb.init(
        project="RL_pacman",
        config=vars(args)
    )
    
    optimizer_spec = OptimizerSpec(
        constructor=optim.RMSprop,
        kwargs=dict(lr=args.learning_rate, alpha=args.alpha, eps=args.eps),
    )

    exploration_schedule = LinearSchedule(1000000, 0.1)

    # Create the environment with render_mode set to "rgb_array"
    env = gym.make('ALE/Breakout-v5', render_mode='rgb_array')

    env = RecordEpisodeStatistics(env)

    # Replay buffer
    replay_buffer = ReplayBuffer(
        args.replay_buffer_size,
        args.frame_history_len
    )

    dqn_trainer = DQNTrainer(
        env=env,
        gamma=args.gamma,
        epilson=args.eps,
        optimizer_spec=optimizer_spec,
        exploration_schedule= exploration_schedule,
        target_update_freq=args.target_update_freq,
        learning_freq=args.learning_freq,
        learning_starts=args.learning_starts,
        num_timesteps=args.num_timesteps,
        replay_buffer=replay_buffer,
        batch_size=args.batch_size,
    )

    dqn_trainer.training()
