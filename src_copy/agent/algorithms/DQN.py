

from collections import namedtuple
import random
from utils.gym import get_wrapper_by_name
import torch
import torch.nn as nn
import numpy as np
import wandb
from itertools import count

OptimizerSpec = namedtuple("OptimizerSpec", ["constructor", "kwargs"])

class DQN(nn.Module):
    def __init__(self, action_n , epilson, exploration):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(4, 16, (8 , 8), 4 ),
            nn.ReLU(),
            nn.Conv2d(16,32,(4,4) , 2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(9 * 9 * 32, 256),
            nn.ReLU(),
            nn.Linear(256 , action_n)
        )
        self.epilson = epilson

        self.exploration = exploration

    def forward(self, x):
        return self.layers(x)
    
    def select_epilson_greedy_action(self, x ,t , num_actions):
        sample = random.random()
        eps_threshold = self.exploration.value(t)
        if sample > eps_threshold:
            return self(x).max(dim=1)[1]
        else:
            return torch.tensor([[random.randrange(num_actions)]], dtype=torch.long)




class DQNTrainer():
    
    def __init__(self , 
                 env , 
                 gamma,
                 epilson, 
                 optimizer_spec ,
                 exploration_schedule,
                 target_update_freq, 
                 learning_freq,  
                 learning_starts, 
                 num_timesteps,
                 replay_buffer,
                 batch_size):
        # env 
        self.env = env
        self.action_n = env.action_space.n
        self.gamma = gamma
        # neural network
        self.Q = DQN(self.action_n ,epilson , exploration_schedule)
        self.target_Q = DQN(self.action_n ,epilson, exploration_schedule)

        self.optimizer = optimizer_spec.constructor(self.Q.parameters() , **optimizer_spec.kwargs)

        self.target_update_freq = target_update_freq
        
        self.learning_freq = learning_freq
        
        # epochs
        self.learning_starts = learning_starts
        self.num_timesteps = num_timesteps

        # dataset
        self.replay_buffer = replay_buffer
        self.batch_size = batch_size

        #wandb params:





    def training(self):
        current_step = 0
        num_param_updates = 0
        last_obs = self.env.reset()
        mean_episode_reward = -float('nan')
        best_mean_episode_reward = -float('inf')

        for epoch in range(self.num_timesteps):
            
            # storing the observation in replay memory
            last_idx = self.replay_buffer.store_frame(last_obs)
            
            recent_obs = self.replay_buffer.encode_recent_observation()

            if epoch > self.learning_starts:
                action = self.Q.select_epilson_greedy_action(recent_obs, epoch)[0, 0]
            else:
                action = random.randrange(self.action_n)
            
            obs, reward, done, _ = self.env.step(action)

            reward = max(-1.0, min(reward, 1.0))

            self.replay_buffer.store_effect(last_idx, action , reward, done)

            if done:
                obs = self.env.reset()
                
            last_obs = obs

            if (epoch > self.learning_starts and 
                epoch % self.learning_freq == 0 and 
                self.replay_buffer.can_sample(self.batch_size)):

                obs_batch, act_batch, rew_batch, next_obs_batch, done_mask = self.replay_buffer.sample(self.batch_size)

                dtype = torch.FloatTensor

                obs_batch = torch.tensor(obs_batch, dtype=dtype) / 255.0
                act_batch = torch.tensor(act_batch, dtype=torch.long)
                rew_batch = torch.tensor(rew_batch, dtype=dtype)
                next_obs_batch = torch.tensor(next_obs_batch, dtype=dtype) / 255.0
                not_done_mask = torch.tensor(1 - done_mask, dtype=dtype)
                
                # Compute current Q value, q_func takes only state and output value for every state-action pair
                # We choose Q based on action taken.
                current_Q_values = self.Q(obs_batch).gather(1, act_batch.unsqueeze(1))
                # Compute next Q value based on which action gives max Q values
                # Detach variable from the current graph since we don't want gradients for next Q to propagated
                next_max_q = self.target_Q(next_obs_batch).detach().max(1)[0]
                next_Q_values = not_done_mask * next_max_q
                # Compute the target of the current Q values
                target_Q_values = rew_batch + (self.gamma * next_Q_values)
                # Compute Bellman error
                bellman_error = target_Q_values - current_Q_values
                # clip the bellman error between [-1 , 1]
                clipped_bellman_error = bellman_error.clamp(-1, 1)
                # Note: clipped_bellman_delta * -1 will be right gradient
                d_error = clipped_bellman_error * -1.0
                # Clear previous gradients before backward pass
                self.optimizer.zero_grad()
                # run backward pass
                current_Q_values.backward(d_error.data.unsqueeze(1))
                
                self.optimizer.step()
                num_param_updates += 1

                if num_param_updates % self.target_update_freq == 0:
                    self.target_Q.load_state_dict(Q.state_dict())

            ### 4. Log progress and keep track of statistics
            episode_rewards = get_wrapper_by_name(self.env, "Monitor").get_episode_rewards()
            if len(episode_rewards) > 0:
                mean_episode_reward = np.mean(episode_rewards[-100:])
            if len(episode_rewards) > 100:
                best_mean_episode_reward = max(best_mean_episode_reward, mean_episode_reward)

            # Log to wandb
            wandb.log({
                "mean_episode_reward": mean_episode_reward,
                "best_mean_episode_reward": best_mean_episode_reward,
                "num_episodes": len(episode_rewards),
                "exploration": self.Q.exploration.value(epoch),
                "timestep": epoch,
                "observation": wandb.Image(obs, caption=f"Observation at timestep {epoch}")
            })
            
