# -*- coding: utf-8 -*-
"""
Created on Wed Mar 15 02:28:54 2023

@author: Rohan Chhabra
"""


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


num_arms = int(input("Enter the number of arms: "))
num_steps = 1000
num_bandits= 2000

# We will be randomly taking mean and variance for each arm.

# exp_var = np.random.normal(0,1,num_arms)

num_pulls= np.zeros((num_bandits, num_arms))
reward_values = np.zeros((num_bandits, num_steps))
q_est = np.zeros((num_bandits, num_arms))

reward_values_e0= np.zeros(num_bandits)
reward_values_e001= np.zeros(num_bandits)
reward_values_e01= np.zeros(num_bandits)


def generate_true_arms(num_arms):
    true_arms_value = np.random.normal(0,1,num_arms)
    return true_arms_value

def get_reward(chosen_arm):
    return np.random.normal(true_arms_value[chosen_arm], 1)

def eps_greedy_bandit(iter, num_arms, epsilon):
    
    for i in range(num_steps):
        if np.random.random()<epsilon:
            # We will explore the arms provided
            selected_arm = np.random.randint(num_arms)
        
        else:
            # Exploiting the arm with current maximum reward value
            selected_arm = np.argmax(q_est[iter])
    
        reward = true_arms_value[selected_arm] + np.random.normal(0, 1)
    
        # Incrementing the counter for selected arm
        num_pulls[iter, selected_arm]+=1
        
        # Updating the reward score for the selected arm
        q_est[iter, selected_arm] += (reward - q_est[iter, selected_arm]) / num_pulls[iter, selected_arm]
        # reward_values[iter, selected_arm]+= (reward - reward_values[iter, selected_arm])/ num_pulls[iter, selected_arm]
        
        reward_values[iter, i]= reward
        
    average_rewards= np.mean(reward_values, axis=0)
    return average_rewards


for iter in range(num_bandits):
    epsilon=0.0
    true_arms_value= generate_true_arms(num_arms)
    reward_values_e0 =eps_greedy_bandit(iter, num_arms, epsilon)
    
for iter in range(num_bandits):
    epsilon=0.01
    true_arms_value= generate_true_arms(num_arms)
    reward_values_e001 =eps_greedy_bandit(iter, num_arms, epsilon)
    

for iter in range(num_bandits):
    epsilon=0.1
    true_arms_value= generate_true_arms(num_arms)
    reward_values_e01 =eps_greedy_bandit(iter, num_arms, epsilon)

plt.figure(figsize=(10, 5))

plt.plot(reward_values_e0, label = "e = 0")
plt.plot(reward_values_e001, label = "e = 0.01")
plt.plot(average_reward_e01, label = "e = 0.1")
plt.legend()
plt.xlabel('Time step')
plt.ylabel('Average reward')
