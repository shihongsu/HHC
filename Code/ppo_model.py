import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from torch_scatter import scatter_mean
from torch_geometric.nn import GATConv
import copy

class PPONet(nn.Module):
    def __init__(self, in_channels, edge_attr_dim, hidden_channels = 32, out_channels = 256, heads = 4, init_weights = True):
        super().__init__()

        # 1st GAT
        self.gat1 = GATConv(in_channels, hidden_channels, heads = heads, concat = True, edge_dim = edge_attr_dim)
        # ELU
        self.activation = nn.ELU()
        # 2nd GAT
        self.gat2 = GATConv(hidden_channels * heads, out_channels, heads = 1, concat = False, edge_dim = edge_attr_dim)
        
        self.edge_mlp = nn.Sequential(nn.Linear(2 * out_channels, 256), 
                                      nn.ReLU(), 
                                      nn.Linear(256, 1))
        
        self.value_mlp = nn.Sequential(nn.Linear(out_channels, 256), 
                                       nn.ReLU(), 
                                       nn.Linear(256, 1))

        if init_weights:
            self._initialize_weights()

    def forward(self, x, edge_index, edge_attr, masked_edge_index, eval=False, a=[], graph_size = []):
        
        """
        since graph is undirected
        so edge index need to be double direction
        """
        x = x.float()
        edge_attr = edge_attr.float()

        x = self.gat1(x, edge_index, edge_attr)
        x = self.activation(x)
        node_embeddings = self.gat2(x, edge_index, edge_attr)

        if len(graph_size) != 0:
            flattened_masked_edge_index = copy.deepcopy(masked_edge_index)
            shift = np.concatenate([np.array([0]), graph_size[:-1]])
            for i in range(1, len(shift)):
                shift[i] += shift[i-1]
            
            for i in range(len(flattened_masked_edge_index)):
                for j in range(len(flattened_masked_edge_index[i])):
                    flattened_masked_edge_index[i][j] += shift[i]

            flattened_masked_edge_index = np.concatenate(flattened_masked_edge_index, axis = 1)  
            row, col = flattened_masked_edge_index
        else:
            row, col = masked_edge_index

        edge_embeddings = torch.cat([node_embeddings[row], node_embeddings[col]], dim=-1)
        edge_logits = self.edge_mlp(edge_embeddings).squeeze(-1)
        

        if len(graph_size) != 0:
            # print("!", edge_logits.shape)

            row_lengths = [e.shape[1] for e in masked_edge_index]
            # print("RL:", row_lengths)
            max_row = max(row_lengths)
            # print("MR:", max_row)
            # Define padding per row (optional, or computed from max row length)
            pad_per_row = [(max_row - e) for e in row_lengths]
            # print("PPR:", pad_per_row)

            rows = []
            idx = 0
            for length, pad in zip(row_lengths, pad_per_row):
                row_data = edge_logits[idx : idx + length]
                padding = torch.full((pad,), float("-inf"), dtype = torch.float32)
                padded_row = torch.cat([row_data, padding])
                rows.append(padded_row)
                idx += length

            edge_logits = torch.stack(rows)

            # print("!", edge_logits.shape)
            # print("*", edge_logits[:3])
            # raise RuntimeError("!")


        if len(graph_size) != 0:
            row_lengths = copy.deepcopy(graph_size)
            # print("OO", node_embeddings.shape)
            rows = []
            idx = 0
            for length in row_lengths:
                # print("1", node_embeddings[idx : idx + length].shape)
                rows.append(torch.mean(node_embeddings[idx : idx + length], dim = 0))
                idx += length
     
            node_embeddings = torch.stack(rows)
            # print("NES:", node_embeddings.shape)
            value = self.value_mlp(node_embeddings).squeeze(-1)
            # print("VS", value.shape)
        else:
            value = self.value_mlp(node_embeddings.mean(dim = 0)).squeeze(-1)

        # action mask
        # if action_mask is not None:
        #     # print("AM:", action_mask)
        #     # action_mask = action_mask.view(-1, 1) 
        #     masked_logits = node_logits.clone()
        #     # print("Masked_logits:", masked_logits)
        #     masked_logits[action_mask == 0] = float("-inf")
        # else:
        #     masked_logits = node_logits

        dist = Categorical(logits = edge_logits)
        # print("PROBS:", edge_logits) 
        # print("DISTPROBS:", dist.probs)
        
        ### TODO ###
        # Finish the forward function
        # Return action, action probability, value, entropy

        # evaluation or not
        if eval:
            action = torch.argmax(edge_logits).unsqueeze(0)
        else:
            # action set empty or not
            if len(a) == 0:
                action = dist.sample()
            else:
                action = a
        # print("action:", action)
        prob = dist.log_prob(action)
        # print("***", len(dist.probs))
        # action = torch.tensor(np.random.randint(0, len(dist.probs)))
        entropy = dist.entropy()

        return action, prob, value, entropy

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, GATConv):
                nn.init.xavier_uniform_(m.lin.weight)
                nn.init.xavier_uniform_(m.att_dst)
                nn.init.xavier_uniform_(m.att_edge)
                nn.init.xavier_uniform_(m.att_src)
            elif isinstance(m, nn.Conv2d):
                nn.init.orthogonal(m.weight, np.sqrt(2))
                nn.init.constant(m.bias, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.orthogonal(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
