import time
import numpy as np
import math
import gymnasium as gym
from scipy.special import erf
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import random
from matplotlib import cm
from matplotlib.colors import ListedColormap
import torch.nn.functional as F
import torch

class ENV(gym.Env):
    def __init__(self,
        ###################### parameters ###########################

        lambda_ = [1550e-9,1555e-9,1560e-9], # (m)
        # d1 = 4000, d2 = 4000, # (m)
        h_UAV = 350, # (m)
        h_BS = 5, # (m) height of the base station
        ###################### Geo Loss & Atmospheric Loss ######################
        aperture_diameter = 2e-1, # (m) diameter of aperture
        divergence_angle = 1e-4, # (rad) divergence angle of the beam
        photon_responsitivity = 0.9, # (A/W) photon responsitivity of the photodetector
 
        ###################### UAV Environment #########################
        users = 350,
        uavs = 3,
        size = 2000,
        varphi_ = np.pi/4,
        v0 = 20, # UAV's velocity (m/s)
        tau = 1,
        ##### FSO Backhaul ######
        noise_power_FSO_backhaul = 1e-4, # (W)
        P_FSO_backhaul = 0.4, # (W)
        B_FSO_backhaul = 1e9, # (Hz)
        ##### RF Backhaul ######
        noise_power_RF_backhaul = 1e-6, # (W)
        P_RF_backhaul = 100, # (W)
        B_RF_backhaul = 50e6, # (Hz)
        frequency_RF_backhaul = [2e9, 2.1e9, 2.2e9], # (Hz)
        ##### RF Access ######
        noise_power = 1e-14, # (W)
        P_UAV = 50, # (W) -> 
        B_RF = 20e6, # (Hz) 
        Total_bandwidth_RF = 3e9, # (GHz)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    
        r_th = 100e6, # (bps) ~ 20Mbps
        max_step = 300,
        grid_num = 5,
        ####### Train or Test ? #######
        test = False,
        seed = 42,
        ###### Gauss-Markov Mobility Model for User Moving #########
        user_moving_percent = 0.2, # percentage of user moving
        link_switching = True
        ):

        self.lambda_ = np.array(lambda_)
        self.h_UAV = h_UAV
        self.h_BS = h_BS
        ###################### Geometric Loss #########################
        self.aperture_diameter = aperture_diameter 
        self.divergence_angle = divergence_angle
        self.photon_responsitivity = photon_responsitivity
        self.Visibility = np.random.uniform(40, 100) # Randomly fluctuate visibility to simulate the real environment
        ############## UAV Environment ##################################
        self.users = users
        self.uavs = uavs
        self.size = size
        self.varphi_ = varphi_
        self.UAV_coverage = self.h_UAV*math.tan(self.varphi_)
        self.v_0 = v0
        self.tau = tau
        ############## FSO Backhaul ############################
        self.P_FSO_backhaul = P_FSO_backhaul
        self.B_FSO_backhaul = B_FSO_backhaul
        self.C_FSO = np.zeros(self.uavs)
        self.noise_power_FSO_backhaul = noise_power_FSO_backhaul
        ############## RF Backhaul ############################
        self.noise_power_RF_backhaul = noise_power_RF_backhaul
        self.P_RF_backhaul = P_RF_backhaul
        self.B_RF_backhaul = B_RF_backhaul
        self.frequency_RF_backhaul = frequency_RF_backhaul
        self.C_RF_backhaul = np.zeros(self.uavs)
        self.C_FSO_or_RF = np.zeros(self.uavs) # FSO or RF backhaul capacity, depending on the environment conditions
        ############## RF Access ############################
        self.noise_power = noise_power
        self.P_UAV = np.zeros(self.uavs) + 50
        self.B_RF = B_RF 
        self.r_th = r_th
        #################### Access Links #############
        self.K = 50
        self.psi_L = 1
        self.psi_M_real = np.random.normal(0, 1/np.sqrt(2))
        self.psi_M_imag = np.random.normal(0, 1/np.sqrt(2))
        self.d = 1
        self.lambda_c = 3e8/np.array([5e9,6e9,7e9])
        self.alpha = 2.7
        ####################
        self.max_step = max_step
        self.uavs_location = np.zeros((2, self.uavs))
        # self.hap_location = np.array([[300],[400]])
        self.users_location = np.clip(np.random.normal(loc=1000, scale=300, size=(2, self.users)),0,1999)
        self.satisfied_users = np.zeros(self.users)
        self.rate_UAV = np.zeros(self.users) - 1000
        self.UAV0_behavior = np.zeros((2, self.max_step))
        self.UAV1_behavior = np.zeros((2, self.max_step))
        self.UAV2_behavior = np.zeros((2, self.max_step))
        self.step_ = 0
        self.grid_num = grid_num
        self.grid_size = self.size/self.grid_num
        self.heatmap_users = np.zeros((self.grid_num,self.grid_num))
        # Dictionary maps the abstract actions to the directions
        self.action_space = gym.spaces.Discrete(5)
        self._action_to_direction = {
            0: np.array([0, 0]),  # remain stationary
            1: np.array([0,self.v_0*self.tau]),  # up
            2: np.array([-self.v_0*self.tau, 0]),  # left
            3: np.array([0, -self.v_0*self.tau]),  # down
            4: np.array([self.v_0*self.tau, 0]),  # right
        }
        ###### Cloud attenuation ##########
        # self.Nc = Nc 
        # self.Hcl = Hcl
        # self.heatmap_cloud = 0 #CLWC map
        # self.CLWC = 0
        # self.cloud_moving_step = cloud_moving_step # if =0.1 -> 1 step = 10 cloud step 
        # self.grid_num_cloud = 50
        # self.grid_size_cloud = self.size/self.grid_num_cloud
        # self.cloud_range = int(1/self.cloud_moving_step * self.max_step)
        ##### Train or Test ??? ###########
        self.test = test
        self.seed = seed
        ###### Gauss-Markov Mobility for User Moving ###############
        self.s_markov = np.zeros(self.users)
        self.d_markov = np.zeros(self.users)
        self.s_mean = 0.67
        self.d_mean = 90* (180/np.pi) # convert to radian
        self.al = 0.5 
        self.random_user_list = random.sample(list(range(self.users)), 20)
        self.user_moving_percent = user_moving_percent
        ######### the number of users within coverage ########
        self.N_UAV0 = 0
        self.N_UAV1 = 0
        self.N_UAV2 = 0
        self.N_UAV = np.array([self.N_UAV0,self.N_UAV1,self.N_UAV2])
        ######### the number of supported users of each UAV ####
        self.S_UAV0 = 0
        self.S_UAV1 = 0
        self.S_UAV2 = 0
        self.S_UAV = np.array([self.S_UAV0,self.S_UAV1,self.S_UAV2])
        ######## Heatmap of UAV ##############
        self.heatmap_UAV0 = np.zeros((self.grid_num,self.grid_num))
        self.heatmap_UAV1 = np.zeros((self.grid_num,self.grid_num))
        self.heatmap_UAV2 = np.zeros((self.grid_num,self.grid_num))
        self.heatmap_users_unsatisfied = np.zeros((self.grid_num,self.grid_num))

        ##### discrete action space for UAV movement ######
        self._action_to_direction = {
            0: np.array([0, 0]),  # remain stationary
            1: np.array([0,self.v_0*self.tau]),  # up
            2: np.array([-self.v_0*self.tau, 0]),  # left
            3: np.array([0, -self.v_0*self.tau]),  # down
            4: np.array([self.v_0*self.tau, 0]),  # right
        }
        self.FSO_RF_switching_mode = [0,0,0] # 0: FSO backhaul, 1: RF backhaul, 2: Link selection between FSO and RF backhaul based on the environment conditions

        self.link_switching = link_switching

    def users_markov(self,random_user_list):
        if self.test:
            np.random.seed(self.seed) # Set the seed for reproducibility in testing
            s_random = np.random.normal(0, 1)
            d_random = np.random.normal(0, 45)
        else:
            s_random = np.random.normal(0, 1)
            d_random = np.random.normal(0, 45)
        self.s_markov = self.al*self.s_markov + (1-self.al)*self.s_mean + math.sqrt(1-self.al**2)*s_random
        self.d_markov = self.al*self.d_markov + (1-self.al)*self.d_mean + math.sqrt(1-self.al**2)*d_random
        for i in random_user_list:
            self.users_location[0,i] += self.s_markov[i]*math.cos(self.d_markov[i])
            self.users_location[1,i] += self.s_markov[i]*math.sin(self.d_markov[i])
        self.users_location = np.clip(self.users_location,0,self.size)

    
    def users_inside_UAV_coverage(self,d_UAV):
        # UAV Coverage
        connect_tem = np.zeros((self.uavs,self.users))
        d_argmin = np.argmin(d_UAV, axis=0) # Calcute the closest pair of UAV and User
        for n in range(self.users):
            argmin = d_argmin[n] # UAV closest to User
            if d_UAV[argmin,n] <= self.UAV_coverage :   # if < 400 (m) -> connect , may be connected to multiple UAVs
                connect_tem[argmin,n] = 1
        return connect_tem
    
    def path_loss_UAV(self,d_UAV):
        gain_UAV = np.zeros((self.uavs,self.users))
        for n in range(self.users):
            for k in range(self.uavs):
                psi_UAV = math.sqrt((math.sqrt(self.K/(1+self.K))*self.psi_L + math.sqrt(1/(1+self.K))*self.psi_M_real)**2 + math.sqrt(1/(1+self.K))*self.psi_M_imag**2)
                theta = -20*math.log10(4*3.14*self.d/self.lambda_c[k]) # (dB)
                theta = 10**(theta/10)
                g_UAV = (abs(psi_UAV)**2)*theta*(math.sqrt((d_UAV[k,n])**2 + (self.h_UAV)**2)/self.d)**(-self.alpha)
                gain_UAV[k,n] = g_UAV
        return gain_UAV
    
    def distance_UAVs_users(self):
        d_UAV = np.zeros((self.uavs,self.users))
        for k in range(self.uavs):
            for n in range(self.users):
                d_UAV[k,n] = np.linalg.norm(self.users_location[:,n] - self.uavs_location[:,k])
        return d_UAV
    def power_consumption(self,V):
        delta = 0.012 # Profile drag coefficient
        p = 1.225 # Air density (kg/m^3)
        s = 0.05 # Rotor solidity 
        A = 0.503 # Rotor disc area (m^2)
        Omega = 300 # Blade angular velocity (rad/s)
        R = 0.4 # Rotor radius in meters (m)
        P_0 = delta/8 * p  * s * A * (Omega**3) * (R**3)

        k = 0.1 # Incremental correction factor for induced power 
        W = 20 # Aircraft weight (N)
        P_i = (1 + k) * (W**(3/2)) / math.sqrt(2 * p * A)

        U_tip = Omega * R
        v_0 = math.sqrt(W / (2 * p * A))
        d_0 = 0.6 # Fuselage drag ratio
        P = P_0 * (1 + 3 * (V**2) / (U_tip)**2) + P_i * (math.sqrt(math.sqrt(1 + V**4 / (4 * (v_0)**4)) - V**2 / (2 * (v_0)**2))) + 0.5 * d_0 * p * s * A * (V**3)
        return P
    
    # Backhaul RF Capacity
    def FSO_gain(self,UAV_index,V): # RF-based backhaul
        distance_BS_UAV = math.sqrt((self.uavs_location[0,UAV_index])**2 + (self.uavs_location[1,UAV_index])**2 + (self.h_UAV-self.h_BS)**2)
        geo_loss = math.erf(math.sqrt(np.pi)*self.aperture_diameter/(2*math.sqrt(2)*self.divergence_angle*distance_BS_UAV))**2
        # print(geo_loss)
        epsilon = 0
        if V > 50:
            epsilon = 1.6
        elif 6 < V and V <= 50:
            epsilon = 1.3
        elif 1 < V and V <= 6:
            epsilon = 0.16*V + 0.34
        elif 0.5 < V and V <= 1:
            epsilon = V - 0.5
        elif V <= 0.5:
            epsilon = 0

        attenuation_coefficient = 3.91/V * (self.lambda_[UAV_index]/550e-9)**(-epsilon) # (dB/km)
        noise = np.random.normal(0, 0.05)
        print("Attenuation Coefficient: ", attenuation_coefficient, " dB/km")
        atmospheric_loss = math.exp(-attenuation_coefficient*distance_BS_UAV/1000) 
        print(geo_loss,atmospheric_loss)
        return geo_loss*atmospheric_loss
    def RF_gain(self,UAV_index):
        distance_BS_UAV = math.sqrt((self.uavs_location[0,UAV_index])**2 + (self.uavs_location[1,UAV_index])**2 + (self.h_UAV-self.h_BS)**2)
        path_loss_RF_backhaul = (4*math.pi*distance_BS_UAV*self.frequency_RF_backhaul[UAV_index]/3e8)**(-2) # Free-space path loss
        Gain_t = 1 # Transmitter antenna gain (linear scale)
        Gain_r = 1 # Receiver antenna gain (linear scale)
        RF_gain = Gain_t * Gain_r/path_loss_RF_backhaul
        return RF_gain

    def step(self,actions):
        ########################### Reset each step ##################################################

        self.satisfied_users = np.zeros(self.users)
        self.heatmap_users = np.zeros((self.grid_num,self.grid_num))
        self.users_markov(self.random_user_list) # Gauss-Markov mobility model for User moving
        self.P_UAV = np.zeros(self.uavs) + 50
        self.heatmap_UAV0 = np.zeros((self.grid_num,self.grid_num))
        self.heatmap_UAV1 = np.zeros((self.grid_num,self.grid_num))
        self.heatmap_UAV2 = np.zeros((self.grid_num,self.grid_num))
        self.heatmap_users_unsatisfied = np.zeros((self.grid_num,self.grid_num))

        ############################## psi_M is complex number ###############################
        self.psi_M_real = np.random.normal(0, 1/np.sqrt(2))
        self.psi_M_imag = np.random.normal(0, 1/np.sqrt(2))
        ##############################################################################################
        # print(actions[0])
        ################################# UAVs take the actions ################################################
        # for i,action in enumerate(actions[0:1]): #actions[0:self.uavs]
        #     self.uavs_location[0,i] += action[0]*self.v_0
        #     self.uavs_location[1,i] += action[1]*self.v_0
        # self.uavs_location = np.clip(self.uavs_location, 0,self.size) # Restrict UAVs' coordinates

        # actions[3] = np.clip(actions[3], 0, 1)  # Restrict IRS area percentage
        # self.irs = actions[3] * (self.Lx*self.Ly)  # Update IRS area of each UAV  
        # print(actions[3])
        for (action,k) in zip(actions,range(self.uavs)):
            self.uavs_location[:,k] += self._action_to_direction[action]
        self.uavs_location = np.clip(self.uavs_location, 0,self.size) # Restrict UAVs' coordinates
        
        ####################################### UAV behavior ##########################################################
        self.UAV0_behavior[:,self.step_] = self.uavs_location[:,0]
        self.UAV1_behavior[:,self.step_] = self.uavs_location[:,1]
        self.UAV2_behavior[:,self.step_] = self.uavs_location[:,2]
        self.step_ += 1
        ################## Visibility varies over time due to Fog #########
        if self.test:
            np.random.seed(self.seed) # Set the seed for reproducibility in testing
            step = np.random.uniform(-0.5, 0.5) # Randomly fluctuate visibility to simulate the real environment
        else: 
            step = np.random.uniform(-0.5, 0.5) 
        self.Visibility += step
        # self.Visibility = 6
        self.Visibility = np.clip(self.Visibility, 0.5, 10)  # Restrict visibility to a reasonable range
        ################################# Geometric Loss & Atmospheric Loss ##############################
        FSO_backhaul_gain = []
        for UAV_i in range(self.uavs):
            FSO_backhaul_gain.append(self.FSO_gain(UAV_i,self.Visibility))
            print("FSO Backhaul Gain UAV ", UAV_i, ": ", FSO_backhaul_gain[UAV_i])
        # print("Geometric Loss: ", geo_loss)
        RF_backhaul_gain = []
        for UAV_i in range(self.uavs):
            RF_backhaul_gain.append(self.RF_gain(UAV_i))

        ################################## FSO Capacity ###########################################

        SNR_FSO_backhaul = []
        for UAV_i in range(self.uavs):
            SNR = math.e * self.P_FSO_backhaul**2 * FSO_backhaul_gain[UAV_i]**2 * self.photon_responsitivity**2/ (2*np.pi*self.noise_power_FSO_backhaul)
            #print(SNR)
            self.C_FSO[UAV_i] = 1/2 * self.B_FSO_backhaul*math.log2(1+SNR)
            # print("FSO Backhaul Capacity UAV ", UAV_i, ": ", self.C_FSO[UAV_i]/1e9, " Gbps")
            SNR_FSO_backhaul.append(SNR)

        SNR_RF_backhaul = []
        for UAV_i in range(self.uavs):
            SNR = self.P_RF_backhaul*RF_backhaul_gain[UAV_i]/self.noise_power_RF_backhaul
            self.C_RF_backhaul[UAV_i] = self.B_RF_backhaul*math.log2(1+SNR)
            # print("RF Backhaul Capacity UAV ", UAV_i, ": ", self.C_RF_backhaul[UAV_i]/1e9, " Gbps")
            SNR_RF_backhaul.append(SNR)

        ########## Link Selection between FSO and RF backhaul #############
        for UAV_i in range(self.uavs):
            print("UAV ", UAV_i, ": FSO Backhaul Capacity = ", self.C_FSO[UAV_i]/1e9, " Gbps, RF Backhaul Capacity = ", self.C_RF_backhaul[UAV_i]/1e9, " Gbps")
            
            if self.link_switching:
                if self.C_FSO[UAV_i] >= self.C_RF_backhaul[UAV_i]:
                    self.C_FSO_or_RF[UAV_i] = self.C_FSO[UAV_i]
                    self.FSO_RF_switching_mode[UAV_i] = 0
                else:
                    self.C_FSO_or_RF[UAV_i] = self.C_RF_backhaul[UAV_i]
                    self.FSO_RF_switching_mode[UAV_i] = 1
            else:
                self.C_FSO_or_RF[UAV_i] = self.C_FSO[UAV_i]
                self.FSO_RF_switching_mode[UAV_i] = 0
        ###########################################################################################

        ## Distance
        d_UAV = self.distance_UAVs_users()

        # UAV Coverage
        connect_tem = self.users_inside_UAV_coverage(d_UAV)
        ## Signal-to-noise ratio(SNR)
        gain_UAV = self.path_loss_UAV(d_UAV)

        gain_UAV = gain_UAV*connect_tem
        ## Calculate needed power
        # start = time.time()
        P_total_needed_index, P_total_needed = self.power_allocation(gain_UAV)
        # end = time.time()
        # print("Power Allocation Time: ", end - start)
        # Temp
        # JUST ONE UAV , IF MULTIPLE UAVS, please change this one
        remaining_backhaul = []
        for UAV_i in range(self.uavs):
            C_FSO_temp = self.C_FSO_or_RF[UAV_i]
            for i in range(P_total_needed_index.shape[1]):
                if self.P_UAV[UAV_i] < P_total_needed[UAV_i,P_total_needed_index[UAV_i,i]]:
                    break
                if C_FSO_temp < self.r_th:
                    break
                # Allocate Data Rate and Power
                C_FSO_temp -= self.r_th
                self.P_UAV[UAV_i] -= P_total_needed[UAV_i,P_total_needed_index[UAV_i,i]]
                self.satisfied_users[P_total_needed_index[UAV_i,i]] = 1
            remaining_backhaul.append(C_FSO_temp)
        # remaining_backhaul = np.mean(remaining_backhaul)
        ##############################

        ##################### Heat map of Users ###########################
        users_list = []
        for i in range(self.users):
            users_list.append([self.users_location[0,i],self.users_location[1,i]])

        if len(users_list) > 0:
            for i in range(len(users_list)):
                x = int(abs(users_list[i][0]-1)//self.grid_size) # avoid user in border
                y = int(abs(users_list[i][1]-1)//self.grid_size)
                self.heatmap_users[x,y] += 1
        
        users_list_0 = []
        for i in range(self.users):
            if connect_tem[0,i] == 1:
                users_list_0.append([self.users_location[0,i],self.users_location[1,i]])
        if len(users_list_0) > 0:
            for i in range(len(users_list_0)):
                x = int(abs(users_list_0[i][0]-1)//self.grid_size) # avoid user in border
                y = int(abs(users_list_0[i][1]-1)//self.grid_size)
                self.heatmap_UAV0[x,y] += 1

        users_list_1 = []
        for i in range(self.users):
            if connect_tem[1,i] == 1:
                users_list_1.append([self.users_location[0,i],self.users_location[1,i]])
        if len(users_list_1) > 0:
            for i in range(len(users_list_1)):
                x = int(abs(users_list_1[i][0]-1)//self.grid_size) # avoid user in border
                y = int(abs(users_list_1[i][1]-1)//self.grid_size)
                self.heatmap_UAV1[x,y] += 1
        
        users_list_2 = []
        for i in range(self.users):
            if connect_tem[2,i] == 1:
                users_list_2.append([self.users_location[0,i],self.users_location[1,i]])

        if len(users_list_2) > 0:
            for i in range(len(users_list_2)):
                x = int(abs(users_list_2[i][0]-1)//self.grid_size) # avoid user in border
                y = int(abs(users_list_2[i][1]-1)//self.grid_size)
                self.heatmap_UAV2[x,y] += 1

        users_list_satisfied = []
        for i in range(self.users):
            if self.satisfied_users[i] == 0:
                users_list_satisfied.append([self.users_location[0,i],self.users_location[1,i]])

        if len(users_list_satisfied) > 0:
            for i in range(len(users_list_satisfied)):
                x = int(abs(users_list_satisfied[i][0]-1)//self.grid_size) # avoid user in border
                y = int(abs(users_list_satisfied[i][1]-1)//self.grid_size)
                self.heatmap_users_unsatisfied[x,y] += 1
        ##############################################################################
        # print(self.FSO_RF_switching_mode)


        O_UAV0 = np.concatenate((self.uavs_location[:,0]/100,self.uavs_location[:,1]/100,self.uavs_location[:,2]/100,np.reshape(self.heatmap_users+self.heatmap_UAV0,self.grid_num**2), np.array([self.C_FSO_or_RF[0] / 1e9])))
        O_UAV1 = np.concatenate((self.uavs_location[:,0]/100,self.uavs_location[:,1]/100,self.uavs_location[:,2]/100,np.reshape(self.heatmap_users+self.heatmap_UAV1,self.grid_num**2), np.array([self.C_FSO_or_RF[1] / 1e9])))
        O_UAV2 = np.concatenate((self.uavs_location[:,0]/100,self.uavs_location[:,1]/100,self.uavs_location[:,2]/100,np.reshape(self.heatmap_users+self.heatmap_UAV2,self.grid_num**2), np.array([self.C_FSO_or_RF[2] / 1e9])))

        # arr_supported_users = np.array([self.S_UAV0,self.S_UAV1,self.S_UAV2,self.N_UAV0,self.N_UAV1,self.N_UAV2])
        # O_IRS = np.concatenate((self.uavs_location[:,0],self.uavs_location[:,1],self.uavs_location[:,2],np.reshape(self.heatmap_users_unsatisfied,self.grid_num**2),np.reshape(self.heatmap_UAV0,self.grid_num**2),np.reshape(self.heatmap_UAV1,self.grid_num**2),np.reshape(self.heatmap_UAV2,self.grid_num**2),np.reshape(self.heatmap_CLWC(),10**2),self.C_FSO/(1e9),arr_supported_users,self.irs))
        
        S = np.sum(self.satisfied_users)
        self.N_UAV0 = np.sum(connect_tem[0,:])
        self.N_UAV1 = np.sum(connect_tem[1,:])
        self.N_UAV2 = np.sum(connect_tem[2,:])

        self.S_UAV0 = np.sum(connect_tem[0,:]*self.satisfied_users)
        self.S_UAV1 = np.sum(connect_tem[1,:]*self.satisfied_users)
        self.S_UAV2 = np.sum(connect_tem[2,:]*self.satisfied_users)

        std_remain_users = np.std(np.array([abs(self.N_UAV0 - self.S_UAV0),abs(self.N_UAV1 - self.S_UAV1),abs(self.N_UAV2 - self.S_UAV2)]))

        return [O_UAV0, O_UAV1, O_UAV2], [self.S_UAV0/10, self.S_UAV1/10, self.S_UAV2/10], False, [self.S_UAV0, self.S_UAV1, self.S_UAV2], self.Visibility   # [O_UAV0, O_UAV1, O_UAV2],sum([self.N_UAV0 , self.N_UAV1 , self.N_UAV2])/10
    
    def power_allocation(self,gain_UAV):
        P_total_needed = np.zeros((self.uavs,self.users))
        for UAV_i in range(self.uavs):
             # the number of users supported by UAV_i = Backhaul Capacity/ r_th
            for i in range(gain_UAV.shape[1]):
                if gain_UAV[UAV_i,i] == 0: # means no connection to UAV
                    P_total_needed[UAV_i,i] = 10000 
                else:
                    P_total_needed[UAV_i,i] = 2**(self.r_th/(self.B_RF) - 1) * (self.noise_power/gain_UAV[UAV_i,i])
        P_total_needed_index = np.argsort(P_total_needed, axis=1)
        return P_total_needed_index, P_total_needed
    def plot(self):
        ######################################### UAVs' colour ################################
        color_uav = ['purple','green','blue']
        plt.rcParams.update({'font.size': 15})
        ######################################### For Visualization ################################

        ## Distance
        d_UAV = self.distance_UAVs_users()

        ######################################### Visibility ################################

        import matplotlib.colors as mcolors
        import matplotlib.cm as cm
        cmap = ListedColormap(plt.cm.bone(np.linspace(0.4, 1, 256)))
        norm = mcolors.Normalize(vmin=0.5, vmax=10)
        # Convert visibility to color
        bg_color = cmap(norm(self.Visibility))
        fig, ax = plt.subplots()
        # Set background color
        ax.set_facecolor(bg_color)
        # Create colorbar
        sm = cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])

        cbar = fig.colorbar(sm, ax=ax)
        cbar.set_label('Visibility (km)')
        # ################################## UAVs' Coverage #############################################

        # Lấy axes hiện tại
        ax = plt.gca()
        # Vẽ hình tròn quanh điểm
        for UAV_i in range(self.uavs):
            ax.add_patch(Circle((self.uavs_location[0,UAV_i],self.uavs_location[1,UAV_i]), self.UAV_coverage, fill=False, color = 'black', linewidth=1)) #color_uav[UAV_i]
            plt.gca().set_aspect('equal')
        
        # Cài đặt giới hạn hiển thị
        plt.xlim(-50, self.size + 50)
        plt.ylim(-50, self.size + 50)

        ######### BS  #############
        # plt.scatter(0,0, label = 'Starting Position of UAV', marker = ',', color = 'yellow',linewidths=1,edgecolors='black',s=100)
        ######### HAP #############
        # plt.scatter(self.hap_location[0,0],self.hap_location[1,0], label = 'HAP with IRS', marker = '*', color = 'y',linewidths=1,edgecolors='black',s=200)

        # plt.scatter(self.users_location[0,:],self.users_location[1,:], marker = 'o', color = 'c')
        # plt.scatter(self.uavs_location[0,0],self.uavs_location[1,0], label = 'UAV0', marker = ',', color = 'r',linewidths=1,edgecolors='black')
        ######### UAVs ############
        from matplotlib.offsetbox import OffsetImage, AnnotationBbox
        import imageio.v3 as iio  # Used to read images, or use plt.imread
        # fig, ax2 = plt.subplots()
        def add_icon(ax, image_path, x, y, zoom=0.1):
            """
            Places an image at a specific (x, y) coordinate on the plot.
            """
            try:
                # Load the image
                img = iio.imread(image_path)
                
                # Create the OffsetImage object to handle scaling
                imagebox = OffsetImage(img, zoom=zoom)
                
                # Position the image box at the data coordinates (x, y)
                ab = AnnotationBbox(imagebox, (x, y), frameon=False)
                
                # Add it to the axes
                ax.add_artist(ab)
            except Exception as e:
                print(f"Could not load image: {e}")
        for UAV_i in range(self.uavs): # self.uavs
            #plt.scatter(self.uavs_location[0,UAV_i],self.uavs_location[1,UAV_i], label = 'UAV' + str(UAV_i+1), marker = ',', color = color_uav[UAV_i] ,linewidths=1,edgecolors='black',s=50)
            add_icon(ax, 'drone.png', x=self.uavs_location[0,UAV_i], y=self.uavs_location[1,UAV_i], zoom=0.06)
            ax.text(self.uavs_location[0,UAV_i] + 20, self.uavs_location[1,UAV_i] + 120, f"UAV {UAV_i+1}", fontsize=8, fontweight='bold', bbox=dict(facecolor='white', alpha=0.9, edgecolor='none', pad=2))
            #### draw FSO backhaul link
            if self.FSO_RF_switching_mode[UAV_i] == 0: # FSO backhaul
                plt.plot([0, self.uavs_location[0,UAV_i]], [0, self.uavs_location[1,UAV_i]],color='red', label='FSO Backhaul Link' if UAV_i == 0 else None, linewidth=2) # FSO backhaul link
        ######## Users ############
        connect_tem = self.users_inside_UAV_coverage(d_UAV)

        ########## Base station ############
        add_icon(ax, 'BS.png', x=0, y=80, zoom=0.07) # Legend for user
        ax.text(-30, 230, "Base Station", fontsize=9, fontweight='bold', bbox=dict(facecolor='white', alpha=0.9, edgecolor='none', pad=2))
        
        ########## Users ############
        for user_i in range(self.users):
            if connect_tem[0,user_i] == 1 and self.satisfied_users[user_i] == 0:
                plt.scatter(self.users_location[0,user_i],self.users_location[1,user_i], marker = 'o', color = 'gray',s=8)
            elif connect_tem[0,user_i] == 1 and self.satisfied_users[user_i] == 1:
                #add_icon(ax, 'MU_served.png', x=self.users_location[0,user_i], y=self.users_location[1,user_i], zoom=0.015)
                plt.scatter(self.users_location[0,user_i],self.users_location[1,user_i], marker = 'o', color = color_uav[0],s=15,linewidths=1,edgecolors='black')
            elif connect_tem[1,user_i] == 1 and self.satisfied_users[user_i] == 0:
                plt.scatter(self.users_location[0,user_i],self.users_location[1,user_i], marker = 'o', color = 'gray',s=8)
            elif connect_tem[1,user_i] == 1 and self.satisfied_users[user_i] == 1:
                #add_icon(ax, 'MU_served.png', x=self.users_location[0,user_i], y=self.users_location[1,user_i], zoom=0.015)
                plt.scatter(self.users_location[0,user_i],self.users_location[1,user_i], marker = 'o', color = color_uav[1],s=15,linewidths=1,edgecolors='black')
            elif connect_tem[2,user_i] == 1 and self.satisfied_users[user_i] == 0:
                plt.scatter(self.users_location[0,user_i],self.users_location[1,user_i], marker = 'o', color = 'gray',s=8)
            elif connect_tem[2,user_i] == 1 and self.satisfied_users[user_i] == 1:
                #add_icon(ax, 'MU_served.png', x=self.users_location[0,user_i], y=self.users_location[1,user_i], zoom=0.015)
                plt.scatter(self.users_location[0,user_i],self.users_location[1,user_i], marker = 'o', color = color_uav[2],s=15,linewidths=1,edgecolors='black')
                #plt.scatter(self.users_location[0,user_i],self.users_location[1,user_i], marker = 'o', color = color_uav[2],s=25,linewidths=1,edgecolors='black')
            else:
                plt.scatter(self.users_location[0,user_i],self.users_location[1,user_i], marker = 'o', color = 'gray',s=8)

        plt.scatter(self.UAV0_behavior[0,:],self.UAV0_behavior[1,:], color = 'black', s=0.5, alpha=0.4) # 'r'
        plt.scatter(self.UAV1_behavior[0,:],self.UAV1_behavior[1,:], color = 'black', s=0.5, alpha=0.4) # 'g'
        plt.scatter(self.UAV2_behavior[0,:],self.UAV2_behavior[1,:], color = 'black', s=0.5, alpha=0.4) # 'b'
            
        plt.xlabel('x(m)')
        plt.ylabel('y(m)')
        # plt.title('IRS-assisted FSO Communication in UAV Environment \n' + 'Step '+ str(self.step_))
        plt.legend()
        plt.savefig("UAV_moving.png", bbox_inches='tight')
        plt.savefig("UAV_moving.pdf", bbox_inches='tight')
        # plt.savefig("UAV_moving_step"+ str(self.step_) +".pdf", bbox_inches='tight')
        # plt.show()
        plt.close()

        ######################################### For Table ################################
        #Dữ liệu cho bảng
        table_data = [
            ['Backhaul Capacity', str(round(self.C_FSO[0] / 1e9, 2)) + ' (Gbps)',str(round(self.C_FSO[1] / 1e9, 2)) + ' (Gbps)',str(round(self.C_FSO[2] / 1e9, 2)) + ' (Gbps)'],
            ['Threshold Rate', str(self.r_th // 1e6) + ' (Mbps)', str(self.r_th // 1e6) + ' (Mbps)', str(self.r_th // 1e6) + ' (Mbps)'],
            ['The number Supported Users', str(int(np.sum(self.S_UAV0))), str(int(np.sum(self.S_UAV1))), str(int(np.sum(self.S_UAV2)))],
            ['Users within coverage', str(int(np.sum(self.N_UAV0))), str(int(np.sum(self.N_UAV1))), str(int(np.sum(self.N_UAV2)))],
            ['UAV max power', '10 (W)','10 (W)','10 (W)'],
            ['UAV remaining power', str(round(self.P_UAV[0], 2)) + ' (W)', str(round(self.P_UAV[1], 2)) + ' (W)', str(round(self.P_UAV[2], 2)) + ' (W)']
        ]

        # Tạo figure và axis
        fig, ax = plt.subplots()
        ax.axis('off')  # Tắt trục

        # Tạo bảng với tiêu đề cột
        column_labels = ['Parameter', 'UAV0','UAV1','UAV2']
        table = ax.table(
            cellText=table_data,
            colLabels=column_labels,
            cellLoc='center',
            loc='center'
        )

        # Tùy chỉnh font size
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1.2, 1.2)  # Tăng kích thước bảng

        # Tiêu đề
        plt.title('Step ' + str(self.step_), fontsize=14, pad=20)
        plt.savefig("parameters_overtime.png")
        plt.close()
        return 

        
    def reset(self):
        self.uavs_location = np.zeros((2, self.uavs))  # np.concatenate([np.random.uniform(low=0, high=2000, size=(2, 1)), np.zeros((2, self.uavs - 1))], axis=1) 

        # Set seed for reproducible user distribution only
        # rng = np.random.default_rng(42)
        if self.test:
            np.random.seed(42)  # Set seed for reproducibility in test mode
            self.users_location = np.clip(np.random.normal(loc=1000, scale=300, size=(2, self.users)),0, 1999)
        else: 
            self.users_location = np.clip(np.random.normal(loc=1000, scale=300, size=(2, self.users)),0, 1999)

        self.satisfied_users = np.zeros(self.users)
        self.UAV0_behavior = np.zeros((2, self.max_step))
        self.UAV1_behavior = np.zeros((2, self.max_step))
        self.UAV2_behavior = np.zeros((2, self.max_step))
        self.step_ = 0
        self.heatmap_users = np.zeros((self.grid_num,self.grid_num))
        self.P_UAV = np.zeros(self.uavs) + 50
        self.Visibility = np.random.uniform(50, 100) # Randomly initialize visibility to simulate the real environment
        ############ Heatmap #############################
        self.heatmap_UAV0 = np.zeros((self.grid_num,self.grid_num))
        self.heatmap_UAV1 = np.zeros((self.grid_num,self.grid_num))
        self.heatmap_UAV2 = np.zeros((self.grid_num,self.grid_num))
        self.heatmap_users_unsatisfied = np.zeros((self.grid_num,self.grid_num))
        ######### the number of users within coverage ########
        self.N_UAV0 = 0
        self.N_UAV1 = 0
        self.N_UAV2 = 0
        self.N_UAV = np.array([self.N_UAV0,self.N_UAV1,self.N_UAV2])
        ######### the number of supported users of each UAV ####
        self.S_UAV0 = 0
        self.S_UAV1 = 0
        self.S_UAV2 = 0
        self.S_UAV = np.array([self.S_UAV0,self.S_UAV1,self.S_UAV2])

        ############################### Reset User Mobility ############################
        self.s_markov = np.zeros(self.users)
        self.d_markov = np.zeros(self.users)
        self.random_user_list = random.sample(list(range(self.users)), int(self.user_moving_percent*self.users))


        ####### Observations ######

        O_UAV0 = np.concatenate((self.uavs_location[:,0]/100,self.uavs_location[:,1]/100,self.uavs_location[:,2]/100,np.reshape(self.heatmap_users+self.heatmap_UAV0,self.grid_num**2), np.array([self.C_FSO_or_RF[0] / 1e9])))
        O_UAV1 = np.concatenate((self.uavs_location[:,0]/100,self.uavs_location[:,1]/100,self.uavs_location[:,2]/100,np.reshape(self.heatmap_users+self.heatmap_UAV1,self.grid_num**2), np.array([self.C_FSO_or_RF[1] / 1e9])))
        O_UAV2 = np.concatenate((self.uavs_location[:,0]/100,self.uavs_location[:,1]/100,self.uavs_location[:,2]/100,np.reshape(self.heatmap_users+self.heatmap_UAV2,self.grid_num**2), np.array([self.C_FSO_or_RF[2] / 1e9])))

        arr_supported_users = np.array([self.S_UAV0,self.S_UAV1,self.S_UAV2,self.N_UAV0,self.N_UAV1,self.N_UAV2])

        return [O_UAV0, O_UAV1, O_UAV2]
    






        


