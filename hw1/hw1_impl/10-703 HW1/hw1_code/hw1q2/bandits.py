import numpy as np
import matplotlib.pyplot as plt
import math
import random 


## PROBLEM 2 : BANDITS
## In this section, we have given you a template for coding each of the 
## exploration algorithms: epsilon-greedy, optimistic initialization, UCB exploration, 
## and Boltzmann Exploration 

## You will be implementing these algorithms as described in the “10-armed Testbed” in Sutton+Barto Section 2.3
## Please refer to the textbook or office hours if you have any confusion.

## note: you are free to change the template as you like, do not think of this as a strict guideline
## as long as the algorithm is implemented correctly and the reward plots are correct, you will receive full credit

# This is the optional wrapper for exploration algorithm we have provided to get you started
# this returns the expected rewards after running an exploration algorithm in the K-Armed Bandits problem
# we have already specified a number of parameters specific to the 10-armed testbed for guidance
# iterations is the number of times you run your algorithm

# WRAPPER FUNCTION
def explorationAlgorithm(explorationAlgorithm, param, iters):
    cumulativeRewards = []
    for i in range(iters):
        # number of time steps
        t = 1000
        # number of arms, 10 in this instance
        k = 10
        # real reward distribution across K arms
        rewards = np.random.normal(1,1,k)
        # counts for each arm
        n = np.zeros(k)
        # extract expected rewards by running specified exploration algorithm with the parameters above
        # param is the different, specific parameter for each exploration algorithm
        # this would be epsilon for epsilon greedy, initial values for optimistic intialization, c for UCB, and temperature for Boltmann 
        currentRewards = explorationAlgorithm(param, t, k, rewards, n)
        cumulativeRewards.append(currentRewards)
    # TO DO: CALCULATE AVERAGE REWARDS ACROSS EACH ITERATION TO PRODUCE EXPECTED REWARDS
    expectedRewards = "<TODO>"
    return expectedRewards

# EPSILON GREEDY TEMPLATE
def epsilonGreedy(epsilon, steps, k, realRewards, n):
    # TO DO: initialize structure to hold expected rewards per step
    
    # TO DO: initialize an initial q value for each arm
    
    # TO DO: implement the epsilon-greedy algorithm over all steps and return the expected rewards across all steps
    raise NotImplementedError()

# OPTIMISTIC INTIALIZATION TEMPLATE
def optimisticInitialization(value, steps, k, realRewards, n):
    # TO DO: initialize structure to hold expected rewards per step
    
    # TO DO: initialize optimistic initial q values per arm specified by parameter
    
    # TO DO: implement the optimistic initializaiton algorithm over all steps and return the expected rewards across all steps
    raise NotImplementedError()

# UCB EXPLORATION TEMPLATE
def ucbExploration(c, steps, k, realRewards, n):
    # TO DO: initialize structure to hold expected rewards per step

    # TO DO: initialize q values per arm 

    # TO DO: implement the UCB exploration algorithm over all steps and return the expected rewards across all steps
    raise NotImplementedError()


# BOLTZMANN EXPLORATION TEMPLATE
def boltzmannE(temperature, steps, k, realRewards, n):
    # TO DO: initialize structure to hold expected rewards per step

    # TO DO: initialize q values per arm 

    # TO DO: initialize probability values for each arm

    # TO DO: implement the Boltzmann Exploration algorithm over all steps and return the expected rewards across all steps
    raise NotImplementedError()

# PLOT TEMPLATE
def plotExplorations(paramList, explorationAlgorithm):
    # TO DO: for each parameter in the param list, plot the returns from the exploration Algorithm from each param on the same plot
    x = np.arange(1,1001)
    # calculate your Ys (expected rewards) per each parameter value
    # plot all the Ys on the same plot
    # include correct labels on your plot!
    raise NotImplementedError()