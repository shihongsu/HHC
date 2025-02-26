import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from torch_scatter import scatter_mean
from torch_geometric.nn import GATConv

class PPONet(nn.Module):
    def __init__(self, in_channels, edge_attr_dim, hidden_channels = 32, out_channels = 256, heads = 4, init_weights = True):
        super().__init__()

        # 1st GAT
        self.gat1 = GATConv(in_channels, hidden_channels, heads = heads, concat = True, edge_dim = edge_attr_dim)
        # ELU
        self.activation = nn.ELU()
        # 2nd GAT
        self.gat2 = GATConv(hidden_channels * heads, out_channels, heads = 1, concat = False)
        
        self.edge_mlp = nn.Sequential(nn.Linear(2 * out_channels, 256), 
                                      nn.ReLU(), 
                                      nn.Linear(256, 1))
        
        self.value_mlp = nn.Sequential(nn.Linear(out_channels, 256), 
                                       nn.ReLU(), 
                                       nn.Linear(256, 1))

        if init_weights:
            self._initialize_weights()

    def forward(self, x, edge_index, edge_attr, action_mask = None, eval=False, a=[]):
        
        """
        since graph is undirected
        so edge index need to be double direction
        """
        
        x = x.float()
        edge_attr = edge_attr.float()

        x = self.gat1(x, edge_index, edge_attr)
        x = F.elu(x)
        node_embeddings = self.gat2(x, edge_index, edge_attr)

        # edge logits
        row, col = edge_index
        edge_embeddings = torch.cat([node_embeddings[row], node_embeddings[col]], dim = -1)
        edge_logits = self.edge_mlp(edge_embeddings).squeeze(-1)

        # action mask
        if action_mask is not None:
            masked_logits = edge_logits.clone()
            masked_logits[action_mask == 0] = float("-inf")
        else:
            masked_logits = edge_logits

        # aggregate double direction of same edge
        edge_pairs = torch.sort(edge_index, dim = 0).values
        unique_edges, unique_indices = torch.unique(edge_pairs, dim = 1, return_inverse = True)
        aggregated_logits = scatter_mean(masked_logits, unique_indices, dim = 0)

        dist = Categorical(logits = aggregated_logits)
        
        ### TODO ###
        # Finish the forward function
        # Return action, action probability, value, entropy

        # evaluation or not
        if eval:
            action = torch.argmax(aggregated_logits).unsqueeze(0)
        else:
            # action set empty or not
            if len(a) == 0:
                action = dist.sample()
            else:
                action = a
        
        prob = dist.log_prob(action)
        value = self.value_mlp(node_embeddings.mean(dim = 0)).squeeze(-1)
        entropy = dist.entropy()

        return unique_edges, action, prob, value, entropy

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.orthogonal_(m.weight, np.sqrt(2))
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, np.sqrt(2))
                nn.init.constant_(m.bias, 0.0)
                
