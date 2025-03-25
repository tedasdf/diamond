from algorithms.DQN import DQNTrainer, OptimizerSpec
from algorithms.ReplayBuffer import ReplayBuffer
import wandb
import gymnasium as gym
import ale_py
import matplotlib.pyplot as plt
import argparse
import torch.optim as optim
from utils.schedule import LinearSchedule
import gymnasium as gym
from gymnasium.wrappers import Monitor

import gymnasium as gym
import matplotlib.pyplot as plt

# # Run the environment
# for _ in range(1000):
#     # Render the environment to show the game window
#     env.render()

#     # Default action: No-op (action 0)
#     action = 0

#     # Check for keyboard input to control the game
#     if keyboard.is_pressed('left'):
#         action = key_action_mapping['left']
#     elif keyboard.is_pressed('right'):
#         action = key_action_mapping['right']
#     elif keyboard.is_pressed('space'):
#         action = key_action_mapping['space']
#     elif keyboard.is_pressed('down'):
#         action = key_action_mapping['down']
    

#     # Step through the environment with the chosen action
#     obs, reward, terminated, truncated, info = env.step(action)



#     # Reset the environment if the game ends
#     if terminated or truncated:
#         obs, info = env.reset()

# # Close the environment after the loop
# env.close()


if __name__ == "__main__":
    env = gym.make('ALE/Breakout-v5', render_mode='rgb_array')

    # Wrap with the custom step counter
    env = Monitor(env,'./monitor_dir')
    # Now you can track the total steps with env.get_total_steps()
    print(f"Total steps: {env.get_total_steps()}")
    # parser = argparse.ArgumentParser(description="Run RL experiment with WandB logging.")

    # # Environment & Training
    # parser.add_argument("--num_timesteps", type=int, default=1000000, help="Total number of timesteps for training")
    # parser.add_argument("--replay_buffer_size", type=int, default=1000000, help="Size of the replay buffer")
    # parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    # parser.add_argument("--learning_starts", type=int, default=50000, help="Timesteps before training starts")
    # parser.add_argument("--learning_freq", type=int, default=4, help="Frequency of model updates")
    # parser.add_argument("--frame_history_len", type=int, default=4, help="Number of past frames used as input")
    # parser.add_argument("--target_update_freq", type=int, default=10000, help="Frequency of target network updates")

    # # Optimizer Parameters
    # parser.add_argument("--learning_rate", type=float, default=0.00025, help="Learning rate for optimizer")
    # parser.add_argument("--alpha", type=float, default=0.95, help="Alpha value for RMSprop optimizer")
    # parser.add_argument("--eps", type=float, default=1e-2, help="Epsilon value for RMSprop optimizer")

    # # DQN Hyperparameters
    # parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor for future rewards")

    # # Exploration Schedule
    # parser.add_argument("--exploration_final_eps", type=float, default=0.1, help="Final epsilon for exploration")
    # parser.add_argument("--exploration_schedule_steps", type=int, default=1000000, help="Steps for epsilon decay")

    # args = parser.parse_args()


    # wandb.login(key="272a6329f05b645580139131ec5b1eb14bb27769")
    # wandb.init(
    #     project="RL_pacman",
    #     config=vars(args)
    # )
    
    # optimizer_spec = OptimizerSpec(
    #     constructor=optim.RMSprop,
    #     kwargs=dict(lr=args.learning_rate, alpha=args.alpha, eps=args.eps),
    # )

    # exploration_schedule = LinearSchedule(1000000, 0.1)

    # # Create the environment with render_mode set to "rgb_array"
    # env = gym.make('ALE/Breakout-v5', render_mode='rgb_array')

    # # Replay buffer
    # replay_buffer = ReplayBuffer(
    #     args.replay_buffer_size,
    #     args.frame_history_len
    # )

    # dqn_trainer = DQNTrainer(
    #     env=env,
    #     gamma=args.gamma,
    #     epilson=args.eps,
    #     optimizer_spec=optimizer_spec,
    #     exploration_schedule= exploration_schedule,
    #     target_update_freq=args.target_update_freq,
    #     learning_freq=args.learning_freq,
    #     learning_starts=args.learning_starts,
    #     num_timesteps=args.num_timesteps,
    #     replay_buffer=replay_buffer,
    #     batch_size=args.batch_size,
    # )

    # dqn_trainer.training()
