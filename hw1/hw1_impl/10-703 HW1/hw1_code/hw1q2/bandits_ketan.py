import enum
from turtle import pos
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
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
    for _ in range(iters):
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
    expectedRewards = np.average(np.stack(cumulativeRewards, axis=0), axis=0)
    return expectedRewards

# EPSILON GREEDY TEMPLATE
def epsilonGreedy(epsilon, steps, k, realRewards, n):
    # TO DO: initialize structure to hold expected rewards per step
    results = []
    # TO DO: initialize an initial q value for each arm
    q = np.zeros((k))
    
    # TO DO: implement the epsilon-greedy algorithm over all steps and return the expected rewards across all steps
    # raise NotImplementedError()
    for _ in range(steps):
        prob = random.random()
        if prob >= epsilon:
            action = np.argmax(q)
        else:
            action = random.randint(0, n.shape[0]-1)

        n[action] = n[action] + 1
        q[action] = q[action] + (random.normalvariate(realRewards[action], 1)-q[action])/n[action]


        total = 0
        for action_j in range(k):
            if action_j == np.argmax(q):
                total = total + realRewards[action_j] * (1-epsilon+epsilon/k)
            else:
                total = total + realRewards[action_j] * (epsilon/k)

        results.append(total)

    return results

# OPTIMISTIC INTIALIZATION TEMPLATE
def optimisticInitialization(value, steps, k, realRewards, n):
    # TO DO: initialize structure to hold expected rewards per step
    
    # TO DO: initialize optimistic initial q values per arm specified by parameter
    
    # TO DO: implement the optimistic initializaiton algorithm over all steps and return the expected rewards across all steps
    results = []
    # TO DO: initialize an initial q value for each arm
    q = np.ones((k))*value
    
    # TO DO: implement the epsilon-greedy algorithm over all steps and return the expected rewards across all steps
    # raise NotImplementedError()
    for _ in range(steps):
        
        action = np.argmax(q)
        

        n[action] = n[action] + 1
        q[action] = q[action] + (random.normalvariate(realRewards[action], 1)-q[action])/n[action]


        total = 0
        for action_j in range(k):
            if action_j == np.argmax(q):
                total = total + realRewards[action_j]# * (1-epsilon+epsilon/k)
            # else:
            #     total = total + realRewards[action_j] * (epsilon/k)

        results.append(total)

    return results
    # raise NotImplementedError()

# UCB EXPLORATION TEMPLATE
def ucbExploration(c, steps, k, realRewards, n):
    # TO DO: initialize structure to hold expected rewards per step

    # TO DO: initialize q values per arm 

    # TO DO: implement the UCB exploration algorithm over all steps and return the expected rewards across all steps
    results = []
    # TO DO: initialize an initial q value for each arm
    q = np.zeros((k))
    
    # TO DO: implement the epsilon-greedy algorithm over all steps and return the expected rewards across all steps
    # raise NotImplementedError()
    for time_ctr in range(steps):
        
        flag = 0
        for possible_action in range(k):
            if n[possible_action] == 0:
                action = possible_action
                flag = 1
                break
        
        if flag == 0:
            num_arr = np.ones((k)) * math.log(time_ctr)
            total_arr = c * np.sqrt(num_arr/n)
            action = np.argmax(q+total_arr)
        
        n[action] = n[action] + 1
        q[action] = q[action] + (random.normalvariate(realRewards[action], 1)-q[action])/n[action]

        total = 0
        for action_j in range(k):
            if action_j == np.argmax(q):
                total = total + realRewards[action_j]# * (1-epsilon+epsilon/k)
            # else:
            #     total = total + realRewards[action_j] * (epsilon/k)

        results.append(total)

    return results
    # raise NotImplementedError()


# BOLTZMANN EXPLORATION TEMPLATE
def boltzmannE(temperature, steps, k, realRewards, n):
    # TO DO: initialize structure to hold expected rewards per step

    # TO DO: initialize q values per arm 

    # TO DO: initialize probability values for each arm

    # TO DO: implement the Boltzmann Exploration algorithm over all steps and return the expected rewards across all steps
    # raise NotImplementedError()
    results = []
    prob = np.ones((k))/k
    # TO DO: initialize an initial q value for each arm
    q = np.zeros((k))
    
    # TO DO: implement the epsilon-greedy algorithm over all steps and return the expected rewards across all steps
    # raise NotImplementedError()
    for _ in range(steps):
        
        action = np.random.choice(np.arange(k), 1, p=prob)[0]
        
        n[action] = n[action] + 1
        q[action] = q[action] + (random.normalvariate(realRewards[action], 1)-q[action])/n[action]

        total = 0
        for action_j in range(k):
            total = total + realRewards[action_j] * prob[action_j]

        results.append(total)
        new_arr = np.copy(q)*temperature
        # new_arr = new_arr - np.max(new_arr)
        prob = np.exp(new_arr)/np.sum(np.exp(new_arr))

    return results

# PLOT TEMPLATE
def plotExplorations(paramList, exploration, legend, figname):
    # TO DO: for each parameter in the param list, plot the returns from the exploration Algorithm from each param on the same plot
    x = np.arange(1,1001)
    # calculate your Ys (expected rewards) per each parameter value
    # plot all the Ys on the same plot
    # include correct labels on your plot!
    # raise NotImplementedError()
    # epsilons = [0, 0.001, 0.01, 0.1, 1.0]
    # colors = 
    max_value, max_num = 0, 0
    best_res = None
    for ctr, epsilon in enumerate(paramList):
        res = explorationAlgorithm(exploration, epsilon, 100)
        if res[-1] > max_value:
            max_value = res[-1]
            max_num = legend[ctr]
            best_res = res
        plt.plot(np.arange(1000), res)
    plt.legend(legend)
    plt.ylabel('Expected Rewards')
    plt.xlabel('Time steps')
    # plt.show()
    plt.savefig(figname)
    plt.clf()
    return max_num, best_res

if __name__ == "__main__":

    res, res_legend = [], []

    epsilons = [0, 0.001, 0.01, 0.1, 1.0]
    legend = ['epsilon=0', 'epsilon=0.001', 'epsilon=0.01', 'epsilon=0.1', 'epsilon=1.0']
    max_param_1, max_value_1 = plotExplorations(epsilons, epsilonGreedy, legend, '2a')
    print(max_param_1)

    epsilons = [0, 1, 2, 5, 10]
    legend = ['initialize=0', 'initialize=1', 'initialize=2', 'initialize=5', 'initialize=10']
    max_param_2, max_value_2 = plotExplorations(epsilons, optimisticInitialization, legend, '2b')
    print(max_param_2)

    epsilons = [0, 1, 2, 5]
    legend = ['UCB c=0', 'UCB c=1', 'UCB c=2', 'UCB c=5']
    max_param_3, max_value_3 = plotExplorations(epsilons, ucbExploration, legend, '2c')
    print(max_param_3)

    epsilons = [1, 3, 10, 30, 100]
    legend = ['temperature=1', 'temperature=3', 'temperature=10', 'temperature=30', 'temperature=100']
    max_param_4, max_value_4 = plotExplorations(epsilons, boltzmannE, legend, '2d')
    print(max_param_4)

    # plt.clf()
    plt.plot(max_value_1)
    plt.plot(max_value_2)
    plt.plot(max_value_3)
    plt.plot(max_value_4)
    plt.legend([max_param_1, max_param_2, max_param_3, max_param_4])
    plt.ylabel('Expected Rewards')
    plt.xlabel('Time steps')
    # plt.show()
    plt.savefig('2e')

    # plotExplorations([1, 3, 10, 30, 100], boltzmannE)