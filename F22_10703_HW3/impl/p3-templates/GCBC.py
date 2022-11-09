from collections import OrderedDict, deque
from typing import Tuple 
import gym
from gym import spaces
import matplotlib.pyplot as plt
import numpy as np
import random
from queue import Queue

import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from tqdm import tqdm
# Import make_model here from the approptiate model_*.py file
from model_pytorch import make_model
# This model should be the same as problem 2

class GCDataset(torch.utils.data.Dataset):
	def __init__(self, train_states, train_actions) -> None:
		super().__init__()
		self.train_states = train_states
		self.train_actions = train_actions

	def __len__(self):
		return len(self.train_states)
	
	def __getitem__(self, idx):

		input = torch.tensor(self.train_states[idx]).float()
		out = torch.tensor(self.train_actions[idx]).long()
		return {
			'input': input,
			'output': out
		}


### 2.1 Build Goal-Conditioned Task
class FourRooms:
	def __init__(self, l=5, T=30):
		'''
		FourRooms Environment for pedagogic purposes
		Each room is a l*l square gridworld, 
		connected by four narrow corridors,
		the center is at (l+1, l+1).
		There are two kinds of walls:
		- borders: x = 0 and 2*l+2 and y = 0 and 2*l+2 
		- central walls
		T: maximum horizion of one episode
			should be larger than O(4*l)
		'''
		assert l % 2 == 1 and l >= 5
		self.l = l
		self.total_l = 2 * l + 3
		self.T = T

		# create a map: zeros (walls) and ones (valid grids)
		self.map = np.ones((self.total_l, self.total_l), dtype=np.bool)
		# build walls
		self.map[0, :] = self.map[-1, :] = self.map[:, 0] = self.map[:, -1] = False
		self.map[l+1, [1,2,-3,-2]] = self.map[[1,2,-3,-2], l+1] = False
		self.map[l+1, l+1] = False

		# define action mapping (go right/up/left/down, counter-clockwise)
		# e.g [1, 0] means + 1 in x coordinate, no change in y coordinate hence
		# hence resulting in moving right
		self.act_set = np.array([
			[1, 0], [0, 1], [-1, 0], [0, -1] 
		], dtype=np.int)
		self.action_space = spaces.Discrete(4)

		# you may use self.act_map in search algorithm 
		self.act_map = {}
		self.act_map[(1, 0)] = 0
		self.act_map[(0, 1)] = 1
		self.act_map[(-1, 0)] = 2
		self.act_map[(0, -1)] = 3

	def render_map(self):
		plt.imshow(self.map)
		plt.xlabel('y')
		plt.ylabel('x')
		plt.savefig('p2_map.png', 
					bbox_inches='tight', pad_inches=0.1, dpi=300)
		plt.show()
	
	def sample_sg(self) -> Tuple[np.array, np.array]:
		# sample s
		while True:
			s = [np.random.randint(self.total_l), 
				np.random.randint(self.total_l)]
			if self.map[s[0], s[1]]:
				break

		# sample g
		while True:
			g = [np.random.randint(self.total_l), 
				np.random.randint(self.total_l)]
			if self.map[g[0], g[1]] and \
				(s[0] != g[0] or s[1] != g[1]):
				break
		return s, g

	def reset(self, s=None, g=None):
		'''
		Args:
			s: starting position, np.array((2,))
			g: goal, np.array((2,))
		Return:
			obs: np.cat(s, g)
		'''
		if s is None or g is None:
			s, g = self.sample_sg()
		else:
			assert 0 < s[0] < self.total_l - 1 and 0 < s[1] < self.total_l - 1
			assert 0 < g[0] < self.total_l - 1 and 0 < g[1] < self.total_l - 1
			assert (s[0] != g[0] or s[1] != g[1])
			assert self.map[s[0], s[1]] and self.map[g[0], g[1]]
		
		self.s = s
		self.g = g
		self.t = 1

		return self._obs()
	
	def step(self, a):
		'''
		Args:
			a: action, index into act_set
		Return obs, reward, done, info:
			done: whether the state has reached the goal
			info: succ if the state has reached the goal, fail otherwise 
		'''
		assert self.action_space.contains(a)

		# WRITE CODE HERE
		reached_goal = np.allclose(self.s, self.g)
		already_done = reached_goal or self.t >= self.T
		if already_done:
			print("[warning] already done, cannot step")
			return self._obs(), 1 if reached_goal else 0, True, reached_goal

		if self.map[self.s[0], self.s[1]] == 0:
			# "step into a wall?!"
			# self.s = self.s
			pass
		else:
			self.s = self.s + self.act_set[a]
		reward = 0.0 if reached_goal else 1
		reached_goal = np.allclose(self.s, self.g)
		done = False
		self.t += 1
		done = reached_goal or self.t >= self.T
		# END
		
		return self._obs(), reward, done, reached_goal

	def _obs(self):
		return np.concatenate([self.s, self.g])


def plot_traj(env, ax, traj, goal=None):
	traj_map = env.map.copy().astype(np.float)
	traj_map[traj[:, 0], traj[:, 1]] = 2 # visited states
	traj_map[traj[0, 0], traj[0, 1]] = 1.5 # starting state
	traj_map[traj[-1, 0], traj[-1, 1]] = 2.5 # ending state
	if goal is not None:
		traj_map[goal[0], goal[1]] = 3 # goal
	ax.imshow(traj_map)
	ax.set_xlabel('y')
	ax.set_label('x')

### A uniformly random policy's trajectory
def test_step(env: FourRooms):
	s_g = np.array([1, 1])
	g = np.array([2*l+1, 2*l+1])
	s_g = env.reset(s_g, g)
	done = False
	traj = [s_g]
	while not done:
		s_g, _, done, _ = env.step(env.action_space.sample())
		traj.append(s_g)
	traj = np.array(traj)

	ax = plt.subplot()
	plot_traj(env, ax, traj, g)
	plt.savefig('p2_random_traj.png', 
			bbox_inches='tight', pad_inches=0.1, dpi=300)
	plt.show()


def compute_shortest_path(env: FourRooms, start=None, goal=None):
	shortest_traj = None
	shortest_action = None
	visited = np.zeros(env.map.shape, dtype=bool)
	if start is None or goal is None:
		start, goal = env.s, env.g
	_map = env.map
	_act_set = env.act_set
	# act_map = env.act_map

	done = False
	bfs_queue = deque()
	bfs_queue.append(([start], []))
	while len(bfs_queue) > 0:
		states, actions = bfs_queue.popleft()
		cur_s = states[-1]
		if np.allclose(cur_s, goal):
			done = True
			shortest_traj = states
			shortest_action = actions
			break
		for a in range(4):
			s_next = cur_s + _act_set[a]
			if _map[s_next[0], s_next[1]] and not visited[s_next[0], s_next[1]]:
				bfs_queue.append((states + [s_next], actions + [a]))
				visited[s_next[0], s_next[1]] = True
	# in four rooms, there is always a path; if not, check env.T setting.
	assert done, "goal not reached"
	return np.array(shortest_traj, dtype=int), np.array(shortest_action, dtype=int)

def shortest_path_expert(env: FourRooms, render=False):
	""" 
	Implement a shortest path algorithm and collect N trajectories for N goal reaching tasks
	"""
	N = 1000
	expert_trajs = []
	expert_actions = []

	# WRITE CODE HERE
	for i in range(N):
		env.reset()
		traj, actions = compute_shortest_path(env)
		expert_trajs.append(traj)
		expert_actions.append(actions)
	# END
	# You should obtain expert_trajs, expert_actions from search algorithm

	fig, axes = plt.subplots(5,5, figsize=(10,10))
	axes = axes.reshape(-1)
	for idx, ax in enumerate(axes):
		plot_traj(env, ax, expert_trajs[idx])
	
	# Plot a subset of expert state trajectories
	plt.savefig('p2_expert_trajs.png', 
			bbox_inches='tight', pad_inches=0.1, dpi=300)
	if render:
		plt.show()
	return expert_trajs, expert_actions


class GCBC:

	def __init__(self, env, expert_trajs, expert_actions, num_workers=1):
		self.env = env
		self.expert_trajs = expert_trajs
		self.expert_actions = expert_actions
		self.transition_num = sum(map(len, expert_actions))
		self.model = make_model(input_dim=4, output_dim=4)
		# state_dim + goal_dim = 4
		# action_choices = 4
		self.model_optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
		self.num_workers = num_workers
	def generate_behavior_cloning_data(self):
		# training state should be a concatenation of state and goal
		self._train_states = []
		self._train_actions = []
		
		# WRITE CODE HERE
		for traj, actions in zip(self.expert_trajs, self.expert_actions):
			for idx in range(len(actions)):
				self._train_states.append(np.concatenate([traj[idx], traj[-1]]))
				self._train_actions.append(actions[idx])
		# END

		self._train_states = np.array(self._train_states).astype(np.float) # size: (*, 4)
		self._train_actions = np.array(self._train_actions) # size: (*, )
		
	def generate_relabel_data(self):
		# apply expert data goal relabelling trick
		self._train_states = []
		self._train_actions = []

		# WRITE CODE HERE
		for traj, actions in zip(self.expert_trajs, self.expert_actions):
			for idx in range(len(actions)):
				for relabeled_goal in traj[idx+1:]:
					self._train_states.append(np.concatenate([traj[idx], relabeled_goal]))
					self._train_actions.append(actions[idx])
		# END

		self._train_states = np.array(self._train_states).astype(np.float) # size: (*, 4)
		self._train_actions = np.array(self._train_actions) # size: (*, 4)

	# def train_epoch(self):
	# 	out = self.model(self._train_states)


	def train(self, num_epochs=20, batch_size=256):
		""" 
		Trains the model on training data generated by the expert policy.
		Args:
			num_epochs: number of epochs to train on the data generated by the expert.
			batch_size
		Return:
			loss: (float) final loss of the trained policy.
			acc: (float) final accuracy of the trained policy
		"""
		# WRITE CODE HERE
		# END
		dataset = GCDataset(self._train_states, self._train_actions)
		dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=self.num_workers)
		for epoch in range(num_epochs):
			epoch_loss = 0
			epoch_acc = 0
			for i, data in enumerate(dataloader):
				states, actions = data["input"], data["output"]
				out = self.model(states)
				self.model_optimizer.zero_grad()
				loss = F.cross_entropy(out, actions)
				acc = (out.argmax(dim=1) == actions).float().mean()
				loss.backward()
				self.model_optimizer.step()
				self.model_optimizer.zero_grad()

				epoch_loss += loss.item()
				epoch_acc += acc.item()
			# print(f"Epoch {epoch}: loss {epoch_loss / len(dataloader)}, acc {epoch_acc / len(dataloader)}")

		epoch_loss = epoch_loss / len(dataloader)
		epoch_acc = epoch_acc / len(dataloader)
		return epoch_loss, epoch_acc


def gcbc_policy(gcbc: GCBC):
	def _policy(state_goal_vec):
		input = torch.tensor(state_goal_vec).float()
		out = gcbc.model(input)
		action = out.argmax(dim=0)
		return action.item()
	return _policy

def evaluate_gc(env, policy, n_episodes=50):
	succs = 0
	for _ in range(n_episodes):
		goal_reached = generate_gc_episode(env, policy)
		if goal_reached:
			succs += 1
		# WRITE CODE HERE
		# END
	succs /= n_episodes
	return succs


def generate_gc_episode(env, policy):
	"""Collects one rollout from the policy in an environment. The environment
	should implement the OpenAI Gym interface. A rollout ends when done=True. The
	number of states and actions should be the same, so you should not include
	the final state when done=True.
	Args:
		env: an OpenAI Gym environment.
		policy: a trained model
	Returns:
	"""
	done = False
	s_g = env.reset()

	while not done:
		action = policy(s_g)
		s_g, reward, done, goal_reached = env.step(action)
		# WRITE CODE HERE
		# END
	return goal_reached

def generate_random_trajs():
	N = 1000
	random_trajs = []
	random_actions = []
	random_goals = []

	# WRITE CODE HERE

	# END
	# You should obtain random_trajs, random_actions, random_goals from random policy

	# train GCBC based on the previous code
	# WRITE CODE HERE

def run_GCBC(expert_trajs, expert_actions, env, mode = 'relabel', num_seeds=5, num_iters=150, num_epochs=2, batch_size=256, num_workers=1):
	# mode = 'vanilla'
	loss_vecs = []
	acc_vecs = []
	succ_vecs = []

	for i in range(num_seeds):
		print('*' * 50)
		print('seed: %d' % i)
		loss_vec = []
		acc_vec = []
		succ_vec = []
		# generate new set of trajectories
		# obtain either expert or random trajectories
		gcbc = GCBC(env, expert_trajs, expert_actions, num_workers=num_workers)
		if mode == 'vanilla':
			gcbc.generate_behavior_cloning_data()
		else:
			gcbc.generate_relabel_data()
		print("total train samples:", len(gcbc._train_states))
		for e in tqdm(range(num_iters)):
			loss, acc = gcbc.train(num_epochs=num_epochs, batch_size=batch_size)
			succ = evaluate_gc(env, gcbc_policy(gcbc))
			loss_vec.append(loss)
			acc_vec.append(acc)
			succ_vec.append(succ)
			print(e, round(loss,3), round(acc,3), succ)
		loss_vecs.append(loss_vec)
		acc_vecs.append(acc_vec)
		succ_vecs.append(succ_vec)

	loss_vec = np.mean(np.array(loss_vecs), axis = 0).tolist()
	acc_vec = np.mean(np.array(acc_vecs), axis = 0).tolist()
	succ_vec = np.mean(np.array(succ_vecs), axis = 0).tolist()

	### Plot the results
	from scipy.ndimage import uniform_filter
	# you may use uniform_filter(succ_vec, 5) to smooth succ_vec
	# plt.figure(figsize=(12, 3))
	figure, axes = plt.subplots(1, 3, figsize=(12, 3))
	# WRITE CODE HERE
	for ax, vec, title in zip(axes, [loss_vec, acc_vec, succ_vec], ['loss', 'acc', 'succ']):
		ax.plot(vec)
		ax.set_title(title)
	# END
	plt.savefig('p2_gcbc_%s.png' % mode, dpi=300)
	plt.show()


if __name__ == '__main__':
	# build env
	l, T = 5, 30
	env = FourRooms(l, T)
	env.reset()
	print("env action space: ", env.action_space)
	print("env map: ", env.map)
	env.s = np.array([1, 1], dtype=int)
	env.g = np.array([0, 8], dtype=int)
	print("env sample goal", env.g)
	print("env sample state", env.s)

	# test shortest traj
	# shortest_traj, shortest_actions = compute_shortest_path(env)
	# print("shortest_traj: ", shortest_traj)
	# print("shortest_actions: ", shortest_actions)
	# shortest_path_expert(env, render=True)

	### Visualize the map

	# env.render_map()
	# run_GCBC()