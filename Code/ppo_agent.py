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
from const import *

import networkx as nx
from gym import spaces
from torch_geometric.data import Data

import networkx as nx
# print("NetworkX version:", nx.__version__)
# print("nx.is_directed:", nx.is_directed)


# graph environment
class GraphEnv(gym.Env):
	def __init__(self, patients, distance):
		super(GraphEnv, self).__init__()
		
		self.graph = nx.Graph()
		self.patients = patients # list of requests
		self.distance = distance
		self.assignments = {} # {pat_id: cg_id}

		self.maskedgraph = nx.Graph()

		self._build_graph()

		# # action space
		# num_patients = len(self.patients)
		# num_caregivers = len(self.graph.nodes()) - num_patients
		# self.action_space = spaces.MultiDiscrete([num_patients, num_caregivers])
		# # observation space
		# self.observation_space = spaces.Box(low = 0, high = 1, shape = (num_patients + num_caregivers, num_patients + num_caregivers), dtype = np.float32)
	
	def _build_graph(self):
		# Add pat
		num_patients = len(self.patients)
		for i, patient in enumerate(self.patients):
			self.graph.add_node(i, 
					   			# index = i, 
					   			type = 1, 
					   			level = patient[2], 
								time_window_start = patient[0], 
								time_window_end = patient[1], 
								service_time = patient[3], 
								workload = -1,
								current_time = -1,
								is_add = False)

		# build edge_index and edge_attr
		# edge_index_list = []
		# edge_attr_list = []

		# Add dist and edge type (1 for dist, 0 for assignment)
		for i in range(num_patients):
			for j in range(num_patients):
				if i != j:
					self.graph.add_edge(i, j, distance = self.distance[i][j], edge_type = 1)

		# self.graph.edge_index = torch.tensor(edge_index_list, dtype = torch.long).T # transpose to shape (2, num_edges)
		# self.graph.edge_attr = torch.tensor(edge_attr_list, dtype = torch.float) # shape (num_edges, features)

		# Add "add_cg"
		self.graph.add_node(num_patients, 
					  		# index = num_patients, 
					  		type = 3, 
							level = -1,
							time_window_start = -1,
							time_window_end = -1, 
							service_time = -1,
							workload = WLUB,
							current_time = -1,
							is_add = True)
		
		# for i in range(num_patients):
		# 	self.graph.add_edge(i, num_patients, distance = 0, edge_type = 0)

		# # give 1st caregiver
		# self.graph.add_node(num_patients + 1,
		# 			  		# index = num_patients + 1, 
		# 			  		type = 2,
		# 					level = self.patients[0][2],
		# 					time_window_start = -1,
		# 					time_window_end = -1,
		# 					service_time = -1,
		# 					workload = 0,
		# 					is_add = False)
		# self.caregivers.append([self.patients[0][0] + self.patients[0][3], self.patients[0][3], self.patients[0][2]])

		# for i in range(num_patients):
		# 	self.graph.add_edge(i, num_patients + 1, distance = 0, edge_type = 0)

		# self.assignments.append([0, num_patients + 1])
		
		# only for action mask
		self.maskedgraph = self.graph.__class__()
		self.maskedgraph.add_nodes_from(self.graph.nodes())
		for i in range(num_patients):
			for j in range(num_patients, len(self.graph.nodes())):
				if self.checkmate(i, j):
					self.maskedgraph.add_edge(i, j)
		
		# self.plot_graph(self.graph)
		# self.plot_graph(self.maskedgraph)
	
	def checkmate(self, i, j):
		"""
		i, j: node id
		cg rear
		"""
		for cg in self.assignments.keys():
			if i in self.assignments[cg]:
				return False				
		if (self.graph.nodes[i]["level"] <= self.graph.nodes[j]["level"] or 
			self.graph.nodes[j]["level"] == -1):
			return True
		else: 
			return False

	@staticmethod
	def plot_graph(graph):
		pos = nx.spring_layout(graph)
		nx.draw(graph, pos = pos, with_labels = True)
		plt.show()

	# find the last assignment in caregiver's schedule
	def find_last_assignment(self, caregiver_id):
		return self.assignments[caregiver_id][-1]
	
	def check_time_window(self, cg_id, pat_id, last_pat_id):
		cg = self.graph.nodes[cg_id]
		pat = self.graph.nodes[pat_id]
		dist = self.graph[pat_id][last_pat_id]["distance"]
		serve = pat["service_time"]
		if (cg["level"] >= pat["level"] and 
			cg["workload"] + dist + serve <= WLUB and
			pat["time_window_start"] <= cg["current_time"] + dist and
			cg["current_time"] + dist + pat["service_time"] <= pat["time_window_end"]):
			return True
		return False

	def step(self, action):
		patient_id, caregiver_id = action
		patient_node = self.graph.nodes[patient_id]

		# if type == add caregiver (3)
		if self.graph.nodes[caregiver_id]["type"] == 3:
			# add cg with lv = pat lv
			new_cg_id = len(self.graph.nodes())
			self.graph.add_node(len(self.graph.nodes()), 
					   			# index = len(self.patients) + len(self.caregivers), 
					   			type = 2, 
								level = patient_node["level"], 
								time_window_start = -1, 
								time_window_end = -1, 
								service_time = -1, 
								workload = patient_node["service_time"], 
								current_time = patient_node["time_window_start"] + patient_node["service_time"],
								is_add = False)
			self.assignments[new_cg_id] = [patient_id]
			self.graph.add_edge(patient_id, new_cg_id, distance = 0, edge_type = 0)
			if patient_node["level"] == 0:
				reward = -LV1CG
			elif patient_node["level"] == 0.5:
				reward = -LV2CG
			elif patient_node["level"] == 1:
				reward = -LV3CG

			# for mask
			self.maskedgraph.add_node(new_cg_id)
			for i in range(len(self.patients)):
				# if loop need more checking
				if (self.checkmate(i, new_cg_id) and # check lv and not assigned
					self.check_time_window(new_cg_id, i, patient_id)): # check time window (dist + serve)
						self.maskedgraph.add_edge(i, new_cg_id)
			self.maskedgraph.remove_edges_from(list(self.maskedgraph.edges(patient_id)))			
		# successful assignment
		else:
			caregiver_node = self.graph.nodes[caregiver_id]
			last_assignment = self.find_last_assignment(caregiver_id)
			dist = self.graph[last_assignment][patient_id]["distance"]
			serve = patient_node["service_time"]
			caregiver_node["workload"] += (dist + serve)
			caregiver_node["current_time"] += (dist + serve)
			reward = -dist * TR

			# for mask
			self.maskedgraph.remove_edges_from(list(self.maskedgraph.edges(patient_id)))
			# remove other unqualified edges
			temp = []
			for i in self.maskedgraph.neighbors(caregiver_id):
				if not self.check_time_window(caregiver_id, i, patient_id):
					temp.append(i)
			for i in temp:
				self.maskedgraph.remove_edge(i, caregiver_id)

			self.assignments[caregiver_id].append(patient_id)
			self.graph.add_edge(patient_id, caregiver_id, distance = 0, edge_type = 0)

		# self.plot_graph(self.graph)
		# self.plot_graph(self.maskedgraph)

		num_assignments = sum([len(self.assignments[i]) for i in self.assignments.keys()])

		done = (num_assignments == len(self.patients))
		# if done == 1:
		# 	print("Assignments:", self.assignments.items())
		return self._get_observation(), reward, done, patient_id


	def _get_observation(self):
		return self.graph


	def reset(self):
		"""Reset environment."""
		self.graph.clear()
		self.assignments = {}
		self._build_graph()
		
		return self._get_observation()


class PPOAgent(PPOBaseAgent):
	def __init__(self, config, patients, distance):
		super(PPOAgent, self).__init__(config)
		### TODO ###
		# initialize env
		self.env = GraphEnv(patients, distance)
		### TODO ###
		# initialize test_env
		self.test_env = GraphEnv(patients, distance)

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
			current_time: Current time (for caregivers)
			is_add: Check if whether is the add caregiver node
		
		Edge with attribute:
			dist: distance between nodes
			edge_type: 1: dist, 0: assignment		
		"""
		
		self.node_attr = [# "index", 
						  "type", 
						  "level", 
						  "time_window_start", 
						  "time_window_end", 
						  "service_time", 
						  "workload",
						  "current_time",
						  "is_add"]
		self.edge_attr = ["distance", "edge_type"]


		self.net = PPONet(len(self.node_attr), len(self.edge_attr))
		self.net.to(self.device)
		self.lr = config["learning_rate"]
		self.update_count = config["update_ppo_epoch"]
		self.optim = torch.optim.Adam(self.net.parameters(), lr=self.lr)
		
	def decide_agent_actions(self, observation, masked_graph, eval=False):
		### TODO ###
		# add batch dimension in observation
		# print("Observation:", observation)

		# print("Observation type:", type(observation))
		# print("nx.is_directed(observation):", nx.is_directed(observation))
		
		data = from_networkx(observation, group_node_attrs = self.node_attr, group_edge_attrs = self.edge_attr)
		data = data.to(self.device)
		# print("Data:", data)

		masked_edge_index = from_networkx(masked_graph).edge_index
		edge_tuples = set()
		for u, v in masked_edge_index.T.tolist():
			small, large = min(u, v), max(u, v)
			edge_tuples.add((small, large))
		cleaned_edges = torch.tensor(list(edge_tuples)).T
		masked_edge_index = cleaned_edges
		
		# print("***\n", masked_edge_index)

		# pre_action_mask = from_networkx(observation, group_node_attrs = ["type", "level"]).x.squeeze(-1).tolist()
		# print("PRE:", pre_action_mask)
		# print("ASS:", self.env.assignments)
		# n = len(self.env.patients)
		# m = len(self.env.caregivers)
		# action_mask = []

		# for j in range(m):
		# 	for i in range(n):
		# 		assigned = 0
		# 		for k in self.env.assignments:
		# 			if k[0] == i:
		# 				assigned = 1
		# 				break
		# 		if assigned == 0:				
		# 			if (pre_action_mask[i][1] <= pre_action_mask[n + j][1] or 
		#  				pre_action_mask[n + j][1] == -1):
		# 				action_mask.append(1)
		# 			else: 
		# 				action_mask.append(0)
		# 		else:
		# 			action_mask.append(0)
		# action_mask = np.array(action_mask)
		# print("ACTMASK:", action_mask)

		# node_attributes = observation.nodes(data=True)
		# for node, attributes in node_attributes:
		# 	print(f"Node {node}: {attributes}")

		# edge_attributes = observation.edges(data=True)
		# for u, v, attributes in edge_attributes:
		# 	print(f"Edge ({u}, {v}): {attributes}")


		# get action, value, logp from net		

		if eval:
			with torch.no_grad():
				action, prob, value, _ = self.net(data.x, data.edge_index, data.edge_attr, masked_edge_index, eval=True)
		else:
			action, prob, value, _ = self.net(data.x, data.edge_index, data.edge_attr, masked_edge_index)

		# print("Action:", action.detach().cpu().numpy())
		# action.detach().cpu().numpy()
		return data, action, value.item(), prob.item(), masked_edge_index
	
	def update(self):
		loss_counter = 0.0001
		total_surrogate_loss = 0
		total_v_loss = 0
		total_entropy = 0
		total_loss = 0

		batches = self.gae_replay_buffer.extract_batch(self.discount_factor_gamma, self.discount_factor_lambda)

		# print("Keys in observation_batch:", batches["observation"])  # check if contain edge_index, edge_attr

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
		graph_size_batch = batches["graph_size"][batch_index]
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
				graph_size_train_batch = graph_size_batch[start:start + self.batch_size]

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
					ob_train_batch.edge_attr, am_train_batch, eval = False, a = ac_train_batch.squeeze(), graph_size = graph_size_train_batch)

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
	
