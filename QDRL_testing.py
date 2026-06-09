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
    


if __name__ == '__main__':
    link_switching = False
    env = ENV(link_switching=link_switching, test = True, seed=350) #render_mode="human"
    policy = [Policy().to(device) for _ in range(3)]
    for i in range(3):
        policy[i].load_state_dict(torch.load('QDRL_weights/model_UAV'+str(i)+'.pth',map_location="cpu"), strict=False)

    number_of_sp_users_UAV0 = []
    number_of_sp_users_UAV1 = []
    number_of_sp_users_UAV2 = []
    Visibility = []
    state = env.reset()
    for t in range(300):
        action_UAV0 , log_prob_UAV0 = policy[0].forward(state[0])
        action_UAV1 , log_prob_UAV1 = policy[1].forward(state[1])
        action_UAV2 , log_prob_UAV2 = policy[2].forward(state[2])
        state , _ , _ , sp_users , visibility = env.step([action_UAV0, action_UAV1, action_UAV2])
        number_of_sp_users_UAV0.append(sp_users[0])
        number_of_sp_users_UAV1.append(sp_users[1])
        number_of_sp_users_UAV2.append(sp_users[2])
        Visibility.append(visibility)
        if t%50==0:
            env.plot()
        print(f"Step: {t}, SP Users: {sp_users}")
    if link_switching:
        np.save('number_of_sp_users_UAV0.npy', number_of_sp_users_UAV0)
        np.save('number_of_sp_users_UAV1.npy', number_of_sp_users_UAV1)
        np.save('number_of_sp_users_UAV2.npy', number_of_sp_users_UAV2)
        np.save('Visibility.npy', Visibility)
    else:
        np.save('number_of_sp_users_UAV0_no_switching.npy', number_of_sp_users_UAV0)
        np.save('number_of_sp_users_UAV1_no_switching.npy', number_of_sp_users_UAV1)
        np.save('number_of_sp_users_UAV2_no_switching.npy', number_of_sp_users_UAV2)
        # np.save('Visibility_no_switching.npy', Visibility)
