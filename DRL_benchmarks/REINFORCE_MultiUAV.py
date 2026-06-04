import numpy as np
import gymnasium as gym
import os
import tqdm
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from collections import deque
from IPython.display import Image
from matplotlib import animation
from OISAC_MultiUAV_environment import ENV

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

env = ENV()

# Policy Network
class Policy(nn.Module):
    def __init__(self , s_size=32 , a_size=5 , h_size=128):
        super (Policy , self ).__init__ ()
        self.fc1 = nn.Linear( s_size , h_size )
        self.fc2 = nn.Linear( h_size , h_size )
        self.fc3 = nn.Linear( h_size, a_size )
        # self.fc4 = nn.Linear( s_size , a_size )
    def forward(self , x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        #x = self.fc4(x)
        return F.softmax(x, dim =1)
    def act(self, state ):
        state = torch.from_numpy(state).float().unsqueeze(0)  #.to(device)
        probs = self.forward(state) # .cpu()
        m = Categorical(probs)
        # Random action
        action = m.sample()
        return action.item() , m.log_prob(action)
    

# Training Function
def reinforce(
        policy ,
        optimizer ,
        n_training_episodes ,
        max_steps ,
        gamma ,
        print_every
        ):
    # scores_deque = deque(maxlen =100)
    scores = []

    # Each Episode
    for i_episode in range(1, n_training_episodes + 1):
        saved_log_probs_UAV0 = []
        saved_log_probs_UAV1 = []
        saved_log_probs_UAV2 = []
        rewards_UAV0 = []
        rewards_UAV1 = []
        rewards_UAV2 = []
        state = env.reset()

        # t=1, 2, … , T (compute log(policy(a_t|s_t)))
        for t in range(max_steps):
            action_UAV0 , log_prob_UAV0 = policy[0].act(state[0])
            action_UAV1 , log_prob_UAV1 = policy[1].act(state[1])
            action_UAV2 , log_prob_UAV2 = policy[2].act(state[2])
            saved_log_probs_UAV0.append(log_prob_UAV0)
            saved_log_probs_UAV1.append(log_prob_UAV1)
            saved_log_probs_UAV2.append(log_prob_UAV2)
            state , reward , done , _ , _ = env.step([action_UAV0, action_UAV1, action_UAV2])
            rewards_UAV0.append(reward[0])
            rewards_UAV1.append(reward[1])
            rewards_UAV2.append(reward[2])
            if all(done):
                break
        # scores_deque.append(sum( rewards ))
        scores.append(sum(rewards_UAV0) + sum(rewards_UAV1) + sum(rewards_UAV2))

        returns_UAV0 = deque(maxlen = max_steps)
        returns_UAV1 = deque(maxlen = max_steps)
        returns_UAV2 = deque(maxlen = max_steps)
        n_steps = len(rewards_UAV0)

        # List of discounted Returns (compute gamma^t*G_t)
        for t in range(n_steps)[:: -1]:
            disc_return_t = returns_UAV0[0] if len(returns_UAV0) > 0 else 0
            returns_UAV0.appendleft(gamma*disc_return_t + rewards_UAV0[t])

        for t in range(n_steps)[:: -1]:
            disc_return_t = returns_UAV1[0] if len(returns_UAV1) > 0 else 0
            returns_UAV1.appendleft(gamma*disc_return_t + rewards_UAV1[t])

        for t in range(n_steps)[:: -1]:
            disc_return_t = returns_UAV2[0] if len(returns_UAV2) > 0 else 0
            returns_UAV2.appendleft(gamma*disc_return_t + rewards_UAV2[t])

        # Total loss (disc_return = gamma^t*G_t; log_prob = log(policy(a_t|s_t)))
        policy_loss_UAV0 = []
        for log_prob , disc_return in zip( saved_log_probs_UAV0 , returns_UAV0 ):
            policy_loss_UAV0.append(-log_prob * disc_return )
        policy_loss_UAV0 = torch.stack(policy_loss_UAV0).sum()

        policy_loss_UAV1 = []
        for log_prob , disc_return in zip( saved_log_probs_UAV1 , returns_UAV1 ):
            policy_loss_UAV1.append(-log_prob * disc_return )
        policy_loss_UAV1 = torch.stack(policy_loss_UAV1).sum()

        policy_loss_UAV2 = []
        for log_prob , disc_return in zip( saved_log_probs_UAV2 , returns_UAV2 ):
            policy_loss_UAV2.append(-log_prob * disc_return )
        policy_loss_UAV2 = torch.stack(policy_loss_UAV2).sum()

        optimizer[0].zero_grad()
        optimizer[1].zero_grad()
        optimizer[2].zero_grad()
        (policy_loss_UAV0 + policy_loss_UAV1 + policy_loss_UAV2).backward()
        optimizer[0].step()
        optimizer[1].step()
        optimizer[2].step()

        print(" Episode {}, Reward : {}".format( i_episode ,sum(rewards_UAV0) + sum(rewards_UAV1) + sum(rewards_UAV2)))

    return scores

# Hyperparameter
h_size = 128
lr = 0.001 # 0.0001
n_training_episodes = 4000
max_steps = 300
gamma = 0.99

policy = [Policy() for _ in range(3)] #.to(device)
optimizer = [optim.Adam(policy[i].parameters(), lr=lr) for i in range(3)]

scores = reinforce (
        policy ,
        optimizer ,
        n_training_episodes ,
        max_steps ,
        gamma ,
        print_every = 100)

np.save('REINFORCE_UAV_scores.npy', np.array(scores))