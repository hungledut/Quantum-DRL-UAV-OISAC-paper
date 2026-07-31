import gymnasium as gym
import math
import random
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count
from OISAC_MultiUAV_environment import ENV

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

env = ENV()

# Neural network model for approximating Q-values
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_dim)
        # self.fc4 = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)
    
# Hyperparameters
learning_rate = 0.001
gamma = 0.99
epsilon = 0.1
epsilon_min = 0.01
epsilon_decay = 0.995
batch_size = 64
target_update_freq = 1000
memory_size = 1000
episodes = 4000

policy_net = [DQN(32, 5) for _ in range(3)]
target_net = [DQN(32, 5) for _ in range(3)]


optimizer = [optim.Adam(policy_net[i].parameters(), lr=learning_rate) for i in range(3)]
memory = [deque(maxlen=memory_size) for _ in range(3)]

# Function to choose action using epsilon-greedy policy
def select_action(state, epsilon, policy_network):
    rand_value = random.random()
    if rand_value < epsilon:
        return env.action_space.sample()  # Explore
    else:
        state = torch.FloatTensor(state).unsqueeze(0)
        q_values = policy_network(state)
        return torch.argmax(q_values).item()  # Exploit

# Function to optimize the model using experience replay
def optimize_model(index):
    if len(memory[index]) < batch_size:
        return

    batch = random.sample(memory[index], batch_size)
    state_batch, action_batch, reward_batch, next_state_batch, done_batch = zip(*batch)

    state_batch = torch.FloatTensor(state_batch)
    action_batch = torch.LongTensor(action_batch).unsqueeze(1)
    reward_batch = torch.FloatTensor(reward_batch)
    next_state_batch = torch.FloatTensor(next_state_batch)
    done_batch = torch.FloatTensor(done_batch)

    # Compute Q-values for current states
    q_values = policy_net[index](state_batch).gather(1, action_batch).squeeze()

    # Compute target Q-values using the target network
    with torch.no_grad():
        max_next_q_values = target_net[index](next_state_batch).max(1)[0]
        target_q_values = reward_batch + gamma * max_next_q_values * (1 - done_batch)

    loss = nn.MSELoss()(q_values, target_q_values)

    optimizer[index].zero_grad()
    loss.backward()
    optimizer[index].step()

# Main training loop
rewards_per_episode = []
steps_done = 0

for episode in range(episodes):
    state = env.reset()
    episode_reward = 0
    done = [False, False, False]

    count = 0

    # One episode (One Trajectory)
    while not all(done):
        # if isinstance(state, tuple):
        #     state = state[0]

        if count > 299:
          break

        # Select action
        actions = []
        for i in range(3):
            action = select_action(state[i], epsilon, policy_net[i])
            actions.append(action)
        # action = select_action(state, epsilon)
        # print(actions)
        next_state, reward, done, _ , _ = env.step(actions)

        # Store transition in memory
        for i in range(3):
            memory[i].append((state[i], actions[i], reward[i], next_state[i], done[i]))


        # Update state
        state = next_state
        episode_reward += sum(reward)

        # Optimize model
        for i in range(3):
            optimize_model(i)

        # Update target network periodically
        if steps_done % target_update_freq == 0:
            for i in range(3):
                target_net[i].load_state_dict(policy_net[i].state_dict())

        steps_done += 1

        count += 1

    if episode % 10 == 0:
        env.plot()
    # Decay epsilon
    print('Episode ',episode,': Reward = ',episode_reward)

    epsilon = max(epsilon_min, epsilon_decay * epsilon)
    rewards_per_episode.append(episode_reward)

np.save('DQN_UAV_scores.npy', rewards_per_episode)