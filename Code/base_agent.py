import torch
import torch.nn as nn
import numpy as np
import os
import time
from collections import deque
from torch.utils.tensorboard import SummaryWriter
from gae_replay_buffer import GaeSampleMemory
from replay_buffer import ReplayMemory
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
	from ppo_agent import GraphEnv

class PPOBaseAgent(ABC):
	def __init__(self, config):
		self.env: "GraphEnv" = None
		self.gpu = config["gpu"]
		self.device = torch.device("cuda" if self.gpu and torch.cuda.is_available() else "cpu")
		self.total_time_step = 0
		self.training_steps = int(config["training_steps"])
		self.update_sample_count = int(config["update_sample_count"])
		self.discount_factor_gamma = config["discount_factor_gamma"]
		self.discount_factor_lambda = config["discount_factor_lambda"]
		self.clip_epsilon = config["clip_epsilon"]
		self.max_gradient_norm = config["max_gradient_norm"]
		self.batch_size = int(config["batch_size"])
		self.value_coefficient = config["value_coefficient"]
		self.entropy_coefficient = config["entropy_coefficient"]
		self.eval_interval = config["eval_interval"]
		self.eval_episode = config["eval_episode"]

		self.gae_replay_buffer = GaeSampleMemory({
			"horizon" : config["horizon"],
			"use_return_as_advantage": False,
			"agent_count": 1
			})
		# actions: {assign to a c.g, add a c.g.}

		self.writer = SummaryWriter(config["logdir"])

	@abstractmethod
	def decide_agent_actions(self, observation):
		# add batch dimension in observation
		# get action, value, logp from net

		return NotImplementedError

	@abstractmethod
	def update(self):
		# sample a minibatch of transitions (of size 128)
		obs, actions, rewards, values, log_probs, dones = self.gae_replay_buffer.sample()
		
		# calculate GAE
		advantages, returns = self.gae_replay_buffer.compute_gae(
			rewards, values, dones, gamma = self.discount_factor_gamma, lam = self.discount_factor_lambda)

		# calculate the loss and update the behavior network

		return NotImplementedError


	def train(self):
		episode_idx = 0
		while self.total_time_step <= self.training_steps:
			observation = self.env.reset()
			episode_reward = 0
			episode_len = 0
			episode_idx += 1
			while True:
				data, action, value, logp_pi = self.decide_agent_actions(observation)
				next_observation, reward, terminate, truncate, pat_id = self.env.step(action)
				# observation must be dict before storing into gae_replay_buffer
				# dimension of reward, value, logp_pi, done must be the same
				obs = data
				# print(data)
				obs = torch.tensor(obs.x, dtype = torch.float32, device = self.device)
				print("obs:", obs)
				action = torch.tensor(action, dtype = torch.int64, device = self.device).unsqueeze(0).detach().cpu().numpy()
				print(action.shape)
				logp_pi = torch.tensor(logp_pi, dtype = torch.float32, device = self.device).detach().cpu().numpy()
				self.gae_replay_buffer.append(0, {
								"observation": obs,
								"action": action,
								"reward": reward, 
								"value": value, 
								"logp_pi": logp_pi, 
								"done": terminate})

				if len(self.gae_replay_buffer) >= self.update_sample_count:
					self.update()
					self.gae_replay_buffer.clear_buffer()

				print("Episode reward", episode_reward, ". Reward:", reward)

				episode_reward += reward
				episode_len += 1
				
				if terminate or truncate:
					self.writer.add_scalar('Train/Episode Reward', episode_reward, self.total_time_step)
					self.writer.add_scalar('Train/Episode Len', episode_len, self.total_time_step)
					print(f"[{len(self.gae_replay_buffer)}/{self.update_sample_count}][{self.total_time_step}/{self.training_steps}]  episode: {episode_idx}  episode reward: {episode_reward}  episode len: {episode_len}")
					break
					
				observation = next_observation
				self.total_time_step += 1
				
			if episode_idx % self.eval_interval == 0:
				# save model checkpoint
				avg_score = self.evaluate()
				self.save(os.path.join(self.writer.log_dir, f"model_{self.total_time_step}_{int(avg_score)}.pth"))
				self.writer.add_scalar('Evaluate/Episode Reward', avg_score, self.total_time_step)

	def evaluate(self):
		print("==============================================")
		print("Evaluating...")
		all_rewards = []
		for i in range(self.eval_episode):
			observation, info = self.test_env.reset()
			total_reward = 0
			while True:
				# self.test_env.render()
				action, _, _ = self.decide_agent_actions(observation, eval=True)
				next_observation, reward, terminate, truncate = self.test_env.step(action[0])
				total_reward += reward
				if terminate or truncate:
					print(f"episode {i+1} reward: {total_reward}")
					all_rewards.append(total_reward)
					break

				observation = next_observation
			

		avg = sum(all_rewards) / self.eval_episode
		print(f"average score: {avg}")
		print("==============================================")
		return avg
	
	# save model
	def save(self, save_path):
		torch.save(self.net.state_dict(), save_path)

	# load model
	def load(self, load_path):
		self.net.load_state_dict(torch.load(load_path, map_location=torch.device('cpu')))

	# load model weights and evaluate
	def load_and_evaluate(self, load_path):
		self.load(load_path)
		self.evaluate()


	

