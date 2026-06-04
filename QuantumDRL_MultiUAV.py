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
# 
import torch.nn as nn
import torchquantum as tq
import torchquantum.functional as tqf
from OISAC_MultiUAV_environment import ENV

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

# Initialize the environment
env = ENV() #render_mode="human"

# state_space = env.observation_space.shape[0]
# print('State Space:', state_space)
# action_space = env.action_space.n
# print('Action Space:', action_space)


class Policy(tq.QuantumModule):
    class QLayer(tq.QuantumModule):
        def __init__(self):
            super().__init__()
            self.n_wires = 5
            # self.random_layer = tq.RandomLayer(n_ops=200,
            #                                    wires=list(range(self.n_wires)))
            # gates with trainable parameters
            self.rx0 = tq.RX(has_params=True, trainable=True)
            self.ry0 = tq.RY(has_params=True, trainable=True)
            self.rz0 = tq.RZ(has_params=True, trainable=True)
            self.rx1 = tq.RX(has_params=True, trainable=True)
            self.ry1 = tq.RY(has_params=True, trainable=True)
            self.rz1 = tq.RZ(has_params=True, trainable=True)
            self.rx2 = tq.RX(has_params=True, trainable=True)  
            self.ry2 = tq.RY(has_params=True, trainable=True)
            self.rz2 = tq.RZ(has_params=True, trainable=True)
            self.rx3 = tq.RX(has_params=True, trainable=True)  
            self.ry3 = tq.RY(has_params=True, trainable=True)
            self.rz3 = tq.RZ(has_params=True, trainable=True)
            self.rx4 = tq.RX(has_params=True, trainable=True)  
            self.ry4 = tq.RY(has_params=True, trainable=True)
            self.rz4 = tq.RZ(has_params=True, trainable=True)

        @tq.static_support
        def forward(self, q_device: tq.QuantumDevice):
            """
            1. To convert tq QuantumModule to qiskit or run in the static
            model, need to:
                (1) add @tq.static_support before the forward
                (2) make sure to add
                    static=self.static_mode and
                    parent_graph=self.graph
                    to all the tqf functions, such as tqf.hadamard below
            """
            self.q_device = q_device

            # self.random_layer(self.q_device)

            # some trainable gates (instantiated ahead of time)
            # self.rx0(self.q_device, wires=0)
            # tqf.cnot(self.q_device, wires=[1, 0])
            # self.ry0(self.q_device, wires=1)
            # tqf.cnot(self.q_device, wires=[2, 1])
            # self.rz0(self.q_device, wires=2)
            # tqf.cnot(self.q_device, wires=[3, 2])
            # self.rx1(self.q_device, wires=3)

            self.rx0(self.q_device, wires=0)
            self.rx1(self.q_device, wires=1)
            self.rx2(self.q_device, wires=2)
            self.rx3(self.q_device, wires=3)
            self.rx4(self.q_device, wires=4)
            self.ry0(self.q_device, wires=0)
            self.ry1(self.q_device, wires=1)
            self.ry2(self.q_device, wires=2)
            self.ry3(self.q_device, wires=3)
            self.ry4(self.q_device, wires=4)
            self.rz0(self.q_device, wires=0)
            self.rz1(self.q_device, wires=1)
            self.rz2(self.q_device, wires=2)
            self.rz3(self.q_device, wires=3)
            self.rz4(self.q_device, wires=4)
            tqf.cnot(self.q_device, wires=[1, 0])
            tqf.cnot(self.q_device, wires=[2, 1])
            tqf.cnot(self.q_device, wires=[3, 2])
            tqf.cnot(self.q_device, wires=[4, 3])
            # tqf.cnot(self.q_device, wires=[0, 3])
            # tqf.cnot(self.q_device, wires=[1, 3])
            # tqf.cnot(self.q_device, wires=[0, 3])

    def __init__(self):
        super().__init__()
        self.n_wires = 5
        self.q_device = tq.QuantumDevice(n_wires=self.n_wires)
        # self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict['5_ry'])


        self.q_layer = nn.ModuleList([self.QLayer() for _ in range(8)])

        self.measure = tq.MeasureAll(tq.PauliZ)

        self.quantum_state = []

    def forward(self, x, use_qiskit=False):
        # bsz = x.shape[0]
        # x = F.avg_pool2d(x, 6).view(bsz, 16)

        # output_ = self.fc2(F.relu(self.fc1(x)))
        
        x = torch.from_numpy(x).float().unsqueeze(0)
        # x = F.pad(x, (0, 2**self.n_wires - 31))
        if use_qiskit:
            x = self.qiskit_processor.process_parameterized(self.q_device, self.encoder, self.q_layer, self.measure, x)
        else:
            batch_size = x.shape[0]
            outputs = []
            for i in range(batch_size):
                q_device = tq.QuantumDevice(n_wires=self.n_wires)
                # self.encoder(q_device, x[i].unsqueeze(0)) #.unsqueeze(0)
                
                x = F.normalize(x, p=2, dim=1)
                q_device.set_states(x.to(torch.complex64))
                self.q_layer[0](q_device)
                self.q_layer[1](q_device)
                self.q_layer[2](q_device)
                self.q_layer[3](q_device)


                output = self.measure(q_device)
                # print(output)
                outputs.append(output.to(x.device))
            x = torch.cat(outputs, dim=0)


        # x = x.reshape(x.shape[0], 2, 2).sum(-1)#.squeeze()
        x = F.softmax(x)
        m = Categorical(x)
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
            action_UAV0 , log_prob_UAV0 = policy[0].forward(state[0])
            action_UAV1 , log_prob_UAV1 = policy[1].forward(state[1])
            action_UAV2 , log_prob_UAV2 = policy[2].forward(state[2])
            saved_log_probs_UAV0.append(log_prob_UAV0)
            saved_log_probs_UAV1.append(log_prob_UAV1)
            saved_log_probs_UAV2.append(log_prob_UAV2)
            state , reward , done , _ , _ = env.step([action_UAV0, action_UAV1, action_UAV2])
            rewards_UAV0.append(reward[0])
            rewards_UAV1.append(reward[1])
            rewards_UAV2.append(reward[2])
            if done :
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
        if i_episode % 10 == 0:
            env.plot()
            for i in range(3):
                torch.save(policy[i].state_dict(), "model_UAV{}.pth".format(i))
        

    return scores

# Hyperparameter
h_size = 128
lr = 0.001
n_training_episodes = 4000
max_steps = 300
gamma = 0.99

policy = [Policy().to(device) for _ in range(3)]
optimizer = [optim.Adam(policy[i].parameters(), lr=lr) for i in range(3)]

scores = reinforce (
        policy ,
        optimizer ,
        n_training_episodes ,
        max_steps ,
        gamma ,
        print_every = 100)
np.save('QDRL_UAV_scores.npy', np.array(scores))
# import matplotlib.pyplot as plt
# plt.plot(scores)
# plt.xlabel('Episode')
# plt.ylabel('Reward')
# plt.title('REINFORCE on UAV')
# plt.show()
