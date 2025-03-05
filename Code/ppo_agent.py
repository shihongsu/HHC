import torch
import torch.nn as nn
import numpy as np
import os
import time
from collections import deque
from torch.utils.tensorboard import SummaryWriter
from gae_replay_buffer import GaeSampleMemory
from base_agent import PPOBaseAgent
from ppo_model import PPONet
from torch_geometric.utils import from_networkx
import gym
import cv2

import networkx as nx
from gym import spaces
from torch_geometric.data import Data


# graph environment
class GraphEnv(gym.Env):
	def __init__(self, patients, distance, caregivers):
		super(GraphEnv, self).__init__()
		
		self.graph = nx.Graph()
		self.patients = patients # list of requests
		self.distance = distance
		self.caregivers = caregivers # c.g. info
		self.assignments = {} # {pat_id: c.g._id}
		self.caregiver_counter = len(caregivers) # c.g. count

		self._build_graph()

		# action space
		num_patients = len(self.patients)
		num_caregivers = len(self.caregivers) + 1
		self.action_space = spaces.MultiDiscrete([num_patients, num_caregivers])

		# observation space
		self.observation_space = spaces.Box(low = 0, high = 1, shape = (num_patients + num_caregivers, num_patients + num_caregivers), dtype = np.float32)

	
	def _build_graph(self):
		# Add pat
		num_patients = len(self.patients)
		for i, patient in enumerate(self.patients):
			self.graph.add_node(i, 
					   			type = 1, 
					   			level = patient[2], 
								time_window_start = patient[0], 
								time_window_end = patient[1], 
								service_time = patient[3], 
								workload = -1, 
								is_add = False)

		# build edge_index and edge_attr
		edge_index_list = []
		edge_attr_list = []

		# Add dist
		for i in range(num_patients):
			for j in range(num_patients):
				if i != j:
					self.graph.add_edge(i, j, distance = self.distance[i][j])

					edge_index_list.append([i, j])
					edge_attr_list.append([self.distance[i][j]])

		self.graph.edge_index = torch.tensor(edge_index_list, dtype = torch.long).T # transpose to shape (2, num_edges)
		self.graph.edge_attr = torch.tensor(edge_attr_list, dtype = torch.float) # shape (num_edges, features)

		# Add "add_cg"
		self.graph.add_node(num_patients, 
					  		type = 3, 
							level = -1,
							time_window_start = -1,
							time_window_end = -1, 
							service_time = -1,
							workload = -1,
							is_add = True)
	

	def step(self, action):
		patient_id, caregiver_id = action
		patient_node = self.graph.nodes[patient_id]

		if caregiver_id["type"] == "add_caregiver":
			# add cg with lv = pat lv
			self.graph.add_node(len(self.patients) + len(self.caregivers), 
					   			type = 2, 
								level = patient_node["level"], 
								time_window_start = -1, 
								time_window_end = -1, 
								service_time = -1, 
								workload = 480 - patient_node["service_time"], 
								is_add = False)
			
			self.assignments[patient_id] = len(self.patients) + len(self.caregivers)
			self.caregiver_counter += 1
			reward = 2 # successful assignment
		else:
			# check if assignment valid (time window + lv)
			caregiver_node = self.graph.nodes[caregiver_id]

			if caregiver_node["level"] >= patient_node["level"]:
				self.assignments[patient_id] = caregiver_id
				reward = 5
			else:
				reward = -10

		done = (len(self.assignments) == len(self.patients))
		return self._get_observation, reward, done, {}


	def _get_observation(self):
		return self.graph


	def reset(self):
		"""Reset environment."""
		self.graph.clear()
		self.caregivers.clear()
		self.assignments.clear()
		self.caregiver_counter = 0
		self._build_graph()
		
		return self._get_observation()


class PPOAgent(PPOBaseAgent):
	def __init__(self, config, patients, distance, caregivers):
		super(PPOAgent, self).__init__(config)
		### TODO ###
		# initialize env
		self.env = GraphEnv(patients, distance, caregivers)
		### TODO ###
		# initialize test_env
		self.test_env = GraphEnv(patients, distance, caregivers)

		# Set up PPO net

		"""
		Node with attribute: 
			# index: Node index
			type: patient (1) / caregiver (2) / add caregiver (3)
			level:  Skill level
			time_window_start: Time window start (for patients)
			time_window_end: Time window end (for patients)
			service_time:  Service time (for patients)
			workload:  Workload (for caregivers)
			is_add: Check if whether is the add caregiver node
		
		Edge with attribute:
			dist: distance between nodes		
		"""
		self.node_attr = ["type", 
						  "level", 
						  "time_window_start", 
						  "time_window_end", 
						  "service_time", 
						  "workload", 
						  "is_add"]
		self.edge_attr = ["distance"]


		self.net = PPONet(len(self.node_attr), len(self.edge_attr))

		self.net.to(self.device)
		self.lr = config["learning_rate"]
		self.update_count = config["update_ppo_epoch"]
		self.optim = torch.optim.Adam(self.net.parameters(), lr=self.lr)
		
	def decide_agent_actions(self, observation, eval=False):
		### TODO ###
		# add batch dimension in observation
		data = from_networkx(observation, group_node_attrs = self.node_attr, group_edge_attrs = self.edge_attr).to(self.device)

		# get action, value, logp from net
		if eval:
			with torch.no_grad():
				action, prob, value, _ = self.net(data.x, data.edge_index, data.edge_attr, eval=True)
		else:
			action, prob, value, _ = self.net(data.x, data.edge_index, data.edge_attr)
		
		return action.detach().cpu().numpy(), value.item(), prob.squeeze().item()
	
	def update(self):
		loss_counter = 0.0001
		total_surrogate_loss = 0
		total_v_loss = 0
		total_entropy = 0
		total_loss = 0

		batches = self.gae_replay_buffer.extract_batch(self.discount_factor_gamma, self.discount_factor_lambda)
		sample_count = len(batches["action"])
		batch_index = np.random.permutation(sample_count)
		
		observation_batch = {}
		for key in batches["observation"]:
			observation_batch[key] = batches["observation"][key][batch_index]
		action_batch = batches["action"][batch_index]
		return_batch = batches["return"][batch_index]
		adv_batch = batches["adv"][batch_index]
		v_batch = batches["value"][batch_index]
		logp_pi_batch = batches["logp_pi"][batch_index]

		for _ in range(self.update_count):
			for start in range(0, sample_count, self.batch_size):
				ob_train_batch = {}
				for key in observation_batch:
					ob_train_batch[key] = observation_batch[key][start:start + self.batch_size]
				ac_train_batch = action_batch[start:start + self.batch_size]
				return_train_batch = return_batch[start:start + self.batch_size]
				adv_train_batch = adv_batch[start:start + self.batch_size]
				v_train_batch = v_batch[start:start + self.batch_size]
				logp_pi_train_batch = logp_pi_batch[start:start + self.batch_size]

				ob_train_batch = torch.tensor(ob_train_batch, dtype = torch.float32).to(self.device)
				ac_train_batch = torch.tensor(ac_train_batch, dtype = torch.long).to(self.device)
				adv_train_batch = torch.tensor(adv_train_batch, dtype = torch.float32).to(self.device)
				logp_pi_train_batch = torch.tensor(logp_pi_train_batch, dtype = torch.float32).to(self.device)
				return_train_batch = torch.tensor(return_train_batch, dtype = torch.float32).to(self.device)

				### TODO ###
				# calculate loss and update network
				action, prob, value, entropy = self.net(ob_train_batch, eval = False, a = ac_train_batch.squeeze())

				# size of entropy
				entropy = torch.mean(entropy)

				# calculate policy loss
				ratio = torch.exp(prob - logp_pi_train_batch)
				clip = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon)
				# check if clipped
				surrogate_loss = -torch.mean(torch.min(ratio * adv_train_batch, clip * adv_train_batch))

				# calculate value loss
				value_criterion = nn.MSELoss()
				v_loss = value_criterion(value, return_train_batch)
				
				# calculate total loss
				loss = surrogate_loss + self.value_coefficient * v_loss - self.entropy_coefficient * entropy

				# update network
				self.optim.zero_grad()
				loss.backward()
				nn.utils.clip_grad_norm_(self.net.parameters(), self.max_gradient_norm)
				self.optim.step()

				total_surrogate_loss += surrogate_loss.item()
				total_v_loss += v_loss.item()
				total_entropy += entropy.item()
				total_loss += loss.item()
				loss_counter += 1

		self.writer.add_scalar('PPO/Loss', total_loss / loss_counter, self.total_time_step)
		self.writer.add_scalar('PPO/Surrogate Loss', total_surrogate_loss / loss_counter, self.total_time_step)
		self.writer.add_scalar('PPO/Value Loss', total_v_loss / loss_counter, self.total_time_step)
		self.writer.add_scalar('PPO/Entropy', total_entropy / loss_counter, self.total_time_step)
		print(f"Loss: {total_loss / loss_counter}\
			\tSurrogate Loss: {total_surrogate_loss / loss_counter}\
			\tValue Loss: {total_v_loss / loss_counter}\
			\tEntropy: {total_entropy / loss_counter}\
			")
	
