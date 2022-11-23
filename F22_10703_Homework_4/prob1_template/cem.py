import numpy as np
import scipy.stats as stats


class CEMOptimizer:
    def __init__(self, cost_fn, out_dim, popsize, num_elites, max_iters, 
                 upper_bound=None, lower_bound=None, epsilon=0.001):
        """ Cross Entropy Method (CEM) Class
        Arguments:
            cost_fn: cost function to evaluate the fitness of different states and action pairs
            out_dim (int): dimension of the problem space, here, dimension of action space
            popsize (int): number of candidates to be sampled at each iteration
            num_elites (int): number of top candidates selected to estimate the distribution for next iteration
            max_iters (int): maximum number of iterations for optimization
            epsilon (float): minimum variance. If the maximum variance drops below epsilon, optimization is stopped.
        """
        super().__init__()
        self.cost_fn = cost_fn
        self.out_dim = out_dim 
        self.popsize = popsize
        self.num_elites = num_elites 
        self.max_iters = max_iters
        self.ub = upper_bound.reshape([1, out_dim])
        self.lb = lower_bound.reshape([1, out_dim])
        self.epsilon = epsilon

        # Sanity checks
        assert (num_elites <= popsize), "Error: Number of elites must <= population size!"

    def solve(self, init_mean, init_var):
        """ Optimizes the cost function using the provided initial candidate distribution

        Hint: Use self.cost_fn to get the cost of each trajectory

        Arguments:
            init_mean (np.ndarray): mean of the initial candidate dist, shape: (action_dim * plan_horizon,) 
            init_var (np.ndarray): variance of the initial candidate dist, shape: (action_dim * plan_horizon,) 
        """
        mean, var, itr = init_mean, init_var, 0
        X = stats.norm(loc=np.zeros_like(mean), scale=np.ones_like(mean))

        while (itr < self.max_iters) and np.max(var) > self.epsilon:
            # Sample and select top (elites)
            samples = X.rvs(size=[self.popsize, self.out_dim]) * np.sqrt(var) + mean
            samples = np.clip(samples, self.lb, self.ub)  # np.clip(samples, -1, 1)
            costs = self.cost_fn(samples)
            elites = samples[np.argsort(costs)[:self.num_elites]]

            # Update distribution statistics for next iteration
            mean = np.mean(elites, axis=0)
            var = np.var(elites, axis=0)
            itr += 1
        return mean