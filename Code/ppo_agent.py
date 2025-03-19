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
import torch_geometric.utils.convert as tg_convert
from torch_geometric.data import Batch
import matplotlib.pyplot as plt
import gym
import cv2

import networkx as nx
from gym import spaces
from torch_geometric.data import Data

import networkx as nx
# print("NetworkX version:", nx.__version__)
# print("nx.is_directed:", nx.is_directed)


# graph environment
class GraphEnv(gym.Env):
	def __init__(self, patients, distance, caregivers):
		super(GraphEnv, self).__init__()
		
		self.graph = nx.Graph()
		self.patients = patients # list of requests
		self.distance = distance
		self.caregivers = caregivers # c.g. info
		self.assignments = [] # {pat_id: c.g._id}
		self.caregiver_counter = len(caregivers) # c.g. count

		self._build_graph()

		# action space
		num_patients = len(self.patients)
		print("Pat", self.patients, "Pat#", num_patients)
		print("Cg", self.caregivers)
		num_caregivers = len(self.caregivers) + 1
		self.action_space = spaces.MultiDiscrete([num_patients, num_caregivers])

		# observation space
		self.observation_space = spaces.Box(low = 0, high = 1, shape = (num_patients + num_caregivers, num_patients + num_caregivers), dtype = np.float32)

	
	def _build_graph(self):
		# Add pat
		num_patients = len(self.patients)
		for i, patient in enumerate(self.patients):
			self.graph.add_node(i, 
					   			index = i, 
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
					  		index = num_patients, 
					  		type = 3, 
							level = -1,
							time_window_start = -1,
							time_window_end = -1, 
							service_time = -1,
							workload = -1,
							is_add = True)
		# caregivers [current time, worktime remain, level]
		self.caregivers.append([0, 0, -1])

		# give 1st caregiver
		self.graph.add_node(num_patients + 1,
					  		index = num_patients + 1, 
					  		type = 2,
							level = self.patients[0][2],
							time_window_start = -1,
							time_window_end = -1,
							service_time = -1,
							workload = 480,
							is_add = False)
		
		pos = nx.spring_layout(self.graph)
		nx.draw(self.graph, pos = pos, with_labels = True)
		# plt.show()
	

	def step(self, action):
		caregiver_id = int(action)
		# print("action:", action, "type:", type(action))
		patient_id = len(self.assignments)
		# print("PatID:", patient_id, ", CgID:", caregiver_id)
		patient_node = self.graph.nodes[patient_id]

		# print("Caregiver type:", nx.get_node_attributes(self.graph, "type")[caregiver_id])
		# if type == add caregiver (3)
		if self.graph.nodes[caregiver_id]["type"] == 3:
			# add cg with lv = pat lv
			self.graph.add_node(len(self.patients) + len(self.caregivers), 
					   			index = len(self.patients) + len(self.caregivers), 
					   			type = 2, 
								level = patient_node["level"], 
								time_window_start = -1, 
								time_window_end = -1, 
								service_time = -1, 
								workload = 480 - patient_node["service_time"], 
								is_add = False)
			self.caregivers.append([patient_node["time_window_start"], 480 - patient_node["service_time"], patient_node["level"]])
			self.assignments.append([patient_id, caregiver_id])
			# print("Pat", self.patients)
			print("Assignment", self.assignments)
			self.caregiver_counter += 1
			reward = 2 # successful assignment
		else:
			# check if assignment valid (time window + lv)
			caregiver_node = self.graph.nodes[caregiver_id]

			if caregiver_node["level"] >= patient_node["level"] and caregiver_node["workload"] - patient_node["service_time"] >= 0:
				self.assignments.append([patient_id, caregiver_id])
				caregiver_node["workload"] = caregiver_node["workload"] - patient_node["service_time"]
				reward = 5
			else:
				reward = -10

		done = (len(self.assignments) == len(self.patients))
		if done == 1:
			print(self.assignments)
		return self._get_observation(), reward, done, patient_id


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
		
		self.node_attr = ["index", 
						  "type", 
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
		print("Observation:", observation)

		# print("Observation type:", type(observation))

		# print("nx.is_directed(observation):", nx.is_directed(observation))

		data = from_networkx(observation, group_node_attrs = self.node_attr, group_edge_attrs = self.edge_attr)

		pre_action_mask = from_networkx(observation, group_node_attrs = ["type"]).x.squeeze(-1).tolist()
		action_mask = []
		for i in pre_action_mask:
			if i == 1:
				action_mask.append(0)
			else: 
				action_mask.append(1)
		action_mask = np.array(action_mask)

		print("Data:", data)
		# print("Data type:", type(data))

		# get action, value, logp from net		

		if eval:
			with torch.no_grad():
				action, prob, value, _ = self.net(data.x, data.edge_index, data.edge_attr, action_mask = action_mask, eval=True)
		else:
			action, prob, value, _ = self.net(data.x, data.edge_index, data.edge_attr, action_mask = action_mask)

		# print("Action:", action.detach().cpu().numpy())
		
		return data, action.detach().cpu().numpy(), value.item(), prob.squeeze().item(), action_mask
	
	def update(self):
		loss_counter = 0.0001
		total_surrogate_loss = 0
		total_v_loss = 0
		total_entropy = 0
		total_loss = 0

		batches = self.gae_replay_buffer.extract_batch(self.discount_factor_gamma, self.discount_factor_lambda)

		print("Keys in observation_batch:", batches["observation"])  # check if contain edge_index, edge_attr

		sample_count = len(batches["action"])
		batch_index = np.random.permutation(sample_count)
		
		# observation_batch = {}
		# for key in batches["observation"]:
		# 	# print(key)
		# 	observation_batch[key] = batches["observation"][key][batch_index]

		# for key, value in observation_batch.items():
		# 	print("Key:", key)
		# 	print("Value type:", type(value))
		# 	print("Value sample:", value[:5] if isinstance(value, (list, np.ndarray, torch.Tensor)) else value)

		observation_batch = [batches["observation"][idx] for idx in batch_index]
		action_mask_batch = [batches["action_mask"][idx] for idx in batch_index]
		action_batch = batches["action"][batch_index]
		return_batch = batches["return"][batch_index]
		adv_batch = batches["adv"][batch_index]
		v_batch = batches["value"][batch_index]
		logp_pi_batch = batches["logp_pi"][batch_index]

		for _ in range(self.update_count):
			for start in range(0, sample_count, self.batch_size):
				# ob_train_batch = {}
				# for key in observation_batch:
				# 	ob_train_batch[key] = observation_batch[key][start:start + self.batch_size]
				ob_train_batch = Batch.from_data_list(observation_batch[start:start + self.batch_size])
				am_train_batch = action_mask_batch[start:start + self.batch_size]
				# print("am: ", am_train_batch)
				# raise RuntimeError("123")
				ac_train_batch = action_batch[start:start + self.batch_size]
				return_train_batch = return_batch[start:start + self.batch_size]
				adv_train_batch = adv_batch[start:start + self.batch_size]
				v_train_batch = v_batch[start:start + self.batch_size]
				logp_pi_train_batch = logp_pi_batch[start:start + self.batch_size]

				# print("ob_train_batch:", type(ob_train_batch), ob_train_batch)

				ob_train_batch = ob_train_batch.to(self.device)
				ac_train_batch = torch.tensor(ac_train_batch, dtype = torch.long).to(self.device)
				adv_train_batch = torch.tensor(adv_train_batch, dtype = torch.float32).to(self.device)
				logp_pi_train_batch = torch.tensor(logp_pi_train_batch, dtype = torch.float32).to(self.device)
				return_train_batch = torch.tensor(return_train_batch, dtype = torch.float32).to(self.device)

				### TODO ###
				# calculate loss and update network
				action, prob, value, entropy = self.net(ob_train_batch.x, ob_train_batch.edge_index,
					ob_train_batch.edge_attr, am_train_batch, eval = False, a = ac_train_batch.squeeze())

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
	
