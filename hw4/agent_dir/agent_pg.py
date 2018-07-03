import math
import pdb
import torch
from torch.autograd import Variable
import scipy.misc
import numpy as np


class Agent_PG():
	def __init__(self, env, args):
		self.env = env
		self.log_file = 'pg_baseline_log.txt'
		self._max_iters = 100000
		self._iter = 1

		self._model = Policy(self.env.observation_space.shape,6)
		self._use_cuda = torch.cuda.is_available()
		if self._use_cuda:
			self._model = self._model.cuda()
			torch.cuda.manual_seed_all(0)

		self.reward_history = []
		self._optimizer = torch.optim.Adam(self._model.parameters(),
											  lr=1e-4)
		self.model_filename ='model/pg-baseline-ck-pt-2' 
		if True:
			print('loading trained model')
			if self._use_cuda:
				ckp = torch.load(self.model_filename)
			else:
				ckp = torch.load(self.model_filename,
								 map_location=lambda storage, loc: storage)

			self._model.load_state_dict(ckp['model'])

		self._prev_obs = 0

	def init_game_setting(self):
		pass

	def _preprocess_obs(self, obs):
		obs = obs[34:194]
		obs[obs[:, :, 0] == 144] = 0
		obs[obs[:, :, 0] == 109] = 0
		obs = 0.2126 * obs[:, :, 0] \
			  + 0.7152 * obs[:, :, 1] \
			  + 0.0722 * obs[:, :, 2]
		obs = obs.astype(np.uint8)
		obs = scipy.misc.imresize(obs, (80, 80)).astype(float)
		processed = obs - self._prev_obs
		self._prev_obs = obs
		return processed

	def get_discount_reward(self,reward , N , discount = 0.99):
		r = [reward]
		for i in range(N-1):
			r.append(r[-1] * discount)
		return r[::-1]
	def _train_iteration(self , mean,std):
		obs = self.env.reset()
		n_steps = 0
		done = False
		rewards = []
		eps_reward = 0
		actions_prob = []
		self._optimizer.zero_grad()
		while not done:
			total_log_probs = 0
			total_entropy = 0
			reward = 0
			while reward == 0:
				# calculate action probability
				var_obs = Variable(torch.from_numpy(self._preprocess_obs(obs)).float().unsqueeze(0))
				if self._use_cuda:
					var_obs = var_obs.cuda()
				action_probs = self._model.forward(var_obs)
				
				# sample action
				action = torch.multinomial(action_probs, 1).data[0, 0] 
				obs, reward, done, _ = self.env.step(action)

				# accumulate reward and probability
				total_log_probs += action_probs[:, action].log()
				actions_prob.append( action_probs[:, action].log())
				n_steps += 1
			this_reward = self.get_discount_reward(reward,len(actions_prob) - len(rewards))
			rewards += this_reward
			eps_reward += reward
		rewards=np.array(rewards)
		loss = 0
		discounted_reward = (rewards-mean)

		for w,a in zip(discounted_reward,actions_prob):
			loss += - a*w
		loss.backward()
		torch.nn.utils.clip_grad_norm(self._model.parameters(),
									  5, 'inf')
		self._optimizer.step()
		self._optimizer.zero_grad()

		return eps_reward ,rewards
	def train(self):
		if self.log_file is not None:
			fp_log = open(self.log_file, 'w', buffering=1)

		total_steps = 0
		rewards = [-1.0]
		patience = 10000
		while self._iter < self._max_iters:
			if len(rewards) > patience:
				rewards = rewards[-patience:]
			reward , reward_list = self._train_iteration(np.mean(rewards) , np.std(rewards))
			rewards += reward_list.tolist()
			self.reward_history.append(reward)
			
			
			if len(self.reward_history) > 30:
				self.reward_history = self.reward_history[-30:]
			print(self._iter ,reward ,np.mean(self.reward_history) , file = fp_log)
			if self._iter % 10 == 0:
				print(self._iter , reward ,np.mean(self.reward_history))
	
			if self._iter % 100 == 0:
				print('saving model!')
				torch.save({'model': self._model.state_dict(),
							'iter': self._iter}, self.model_filename)

			self._iter += 1

	def make_action(self, observation, test=True):
		var_obs = Variable(torch.from_numpy(self._preprocess_obs(observation))
						   .float().unsqueeze(0))
		if self._use_cuda:
			var_obs = var_obs.cuda()
		action_probs = self._model.forward(var_obs)
		action = torch.multinomial(action_probs, 1).data[0, 0]
		return action


class Policy(torch.nn.Module):
	def __init__(self, input_shape, n_actions):
		super(Policy, self).__init__()
		self.mlp = torch.nn.Sequential(
			torch.nn.Linear(6400,128),
			torch.nn.ReLU(),
			torch.nn.Linear(128, n_actions),
			torch.nn.Softmax()
		)

	def forward(self, frames):
		frames = frames.unsqueeze(-3)
		x = frames
		x = x.view(x.size(0), -1)
		return self.mlp(x)
