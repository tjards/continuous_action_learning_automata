#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 19:45:47 2024

This program implements Continuous Action Learning Automata (CALA) to learn
    an arbitrary set of actions for n-states

@author: tjards

"""

#%% Import stuff
# ---------------
import numpy as np
import matplotlib.pyplot as plt

#%% Simulations parameters
# ---------------------
num_states      = 3             # number of states
action_min      = -1            # minimum of action space
action_max      = 1             # maximum of action space
target_actions = np.random.uniform(action_min, action_max, num_states) # randomly assigned target actions

#%% Hyperparameters
# -----------------
learning_rate   = 0.1       # rate at which policy updates
variance        = 0.2       # initial variance
variance_ratio  = 1         # default 1, permits faster/slower variance updates
variance_min    = 0.001     # default 0.001, makes sure variance doesn't go too low

# initial means and variances
means = np.random.uniform(action_min, action_max, num_states)
variances = np.full(num_states, variance)

#%% Learning Class
# ----------------
class CALA:
    
    # initialize
    def __init__(self, num_states, action_min, action_max, learning_rate, means, variances):
        
        # load parameters into class
        self.num_states     = num_states
        self.action_min     = action_min
        self.action_max     = action_max
        self.learning_rate  = learning_rate
        self.means          = means
        self.variances      = variances
        
        # store stuff
        self.mean_history       = []
        self.variance_history   = []
        self.reward_history     = []

    # select action
    def select_action(self, state):
        
        # pull mean and variance for given state
        mean        = self.means[state]
        variance    = self.variances[state]
        
        # select action from normal distribution
        action = np.random.normal(mean, np.sqrt(variance))
        
        # return the action, onstrained using clip()
        return np.clip(action, self.action_min, self.action_max)
    
    # update policy 
    def update_policy(self, state, action, reward):
 
        # pull mean and variance for given state
        mean        = self.means[state]
        variance    = self.variances[state]
        
        # update mean and variance based on reward signal
        self.means[state]       += self.learning_rate * reward * (action - mean)
        self.variances[state]   += variance_ratio * self.learning_rate * reward * ((action - mean) ** 2 - variance)
        
        # constrain the variance 
        self.variances[state] = max(variance_min, self.variances[state])

    # run the simulation
    def run(self, num_episodes, environment):
  
        # note: 'environment' is a function (substitute with actual environment feedback)      
  
        # for the desired number of episodes
        for _ in range(num_episodes):
            
            # initialize local storage 
            mean_store      = []
            variance_store  = []
            reward_store    = []
            
            # for each state
            for state in range(0, self.num_states):
                
                # select the action (based on current mean/variance)
                action = self.select_action(state)
                
                # collect reward (based on feedback from environment)
                reward = environment(state, action)
                
                # update the policy (based on reward and hyperparameters)
                self.update_policy(state, action, reward)
                
                # store 
                mean_store.append(self.means[state])
                variance_store.append(self.variances[state])
                reward_store.append(reward)
            
            # append local storage to history
            self.mean_history.append(mean_store)
            self.variance_history.append(variance_store)
            self.reward_history.append(reward_store)

    def plots(self):
 
        time_steps  = len(self.mean_history)
        fig, axs    = plt.subplots(3, 1, figsize=(10, 12))
        
        # arrayerize the history lists
        mean_array      = np.array(self.mean_history)
        variance_array  = np.array(self.variance_history)
        reward_array    = np.array(self.reward_history)

        # Means
        # ----
        for state in range(self.num_states):
            # plot the means
            line, = axs[0].plot(range(time_steps), mean_array[:, state])
            line_color = line.get_color()
            axs[0].axhline(y=target_actions[state], color = line_color, linestyle='--')
            std_devs = np.sqrt(variance_array[:, state])
            axs[0].fill_between(np.arange(time_steps), mean_array[:, state] - std_devs, mean_array[:, state] + std_devs, color=line_color, alpha=0.3)
        # format the plots    
        axs[0].set_title('Action means over time')
        axs[0].set_xlabel('Episodes')
        axs[0].set_ylabel('Mean with standard deviation')
        axs[0].set_ylim(action_min, action_max)
        axs[0].legend()

        # Variances
        # ---------
        for state in range(self.num_states):
            axs[1].plot(range(time_steps), variance_array[:, state])
        axs[1].set_title('Action variance over time')
        axs[1].set_xlabel('Episodes')
        axs[1].set_ylabel('Variance')
        axs[1].legend()

        # Rewards
        # -------
        for state in range(self.num_states):
            axs[2].plot(range(time_steps), reward_array[:, state], label=f"state {state}")
        axs[2].set_title('Reward over time')
        axs[2].set_xlabel('Episodes')
        axs[2].set_ylabel('Reward')
        axs[2].legend()

        plt.tight_layout()
        plt.show()

#%% Example
# --------
def environment(state, action):

    # reward gets exponentially higher, the closer action is to target action
    reward = np.exp(-np.abs(target_actions[state] - action))
    
    return reward

# run the simulation
automata = CALA(num_states, action_min, action_max, learning_rate, means, variances)
automata.run(num_episodes=1000, environment=environment)
automata.plots()