

# https://github.com/haarnoja/sac/blob/master/sac/algos/base.py
# infinite-horizon Markov devision process
# Soft Actor-Critic (SAC) algorithm implementation
# defined by the tuple (S, A ,p ,r)
# S: state space
# A: action space
# p: transition probability
# r: reward function


# p : S x S x A -> [0, inf) representing the probablity density of the net state given current state and action
# r : S x A -> R representing the reward function


class Policy:
    def __init__(self )


class SoftActorCritic:
    def __init__(self, env, actor_critic, replay_buffer, config):
        self.env = env
        self.actor_critic = actor_critic
        self.replay_buffer = replay_buffer
        self.config = config

    def train(self):
        # Training loop for Soft Actor-Critic
        pass

    def update(self):
        # Update the actor and critic networks
        pass

    def save_model(self, filename):
        # Save the model parameters
        pass

    def load_model(self, filename):
        # Load the model parameters
        pass

    def reward_function(self, state, action):
        # Define the reward function
        pass