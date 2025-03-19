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
        self.gat2 = GATConv(hidden_channels * heads, out_channels, heads = 1, concat = False, edge_dim = edge_attr_dim)
        
        self.node_mlp = nn.Sequential(nn.Linear(out_channels, 256), 
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

        # print("x shape:", x.shape)
        # print("edge_attr shape:", edge_attr.shape)

        x = self.gat1(x, edge_index, edge_attr)
        x = self.activation(x)
        node_embeddings = self.gat2(x, edge_index, edge_attr)

        # node logits
        node_logits = self.node_mlp(node_embeddings).squeeze(-1)

        # print("Node logits shape:", node_logits.shape)

        # action mask
        if action_mask is not None:
            print("AM:", action_mask)
            # action_mask = action_mask.view(-1, 1) 
            masked_logits = node_logits.clone()
            # print("Masked_logits:", masked_logits)
            masked_logits[action_mask == 0] = float("-inf")
        else:
            masked_logits = node_logits

        dist = Categorical(logits = masked_logits)
        print("prob:", dist.probs)
        
        ### TODO ###
        # Finish the forward function
        # Return action, action probability, value, entropy

        # evaluation or not
        if eval:
            action = torch.argmax(masked_logits).unsqueeze(0)
        else:
            # action set empty or not
            if len(a) == 0:
                action = dist.sample()
            else:
                action = a
        # print("action:", action)
        prob = dist.log_prob(action)
        value = self.value_mlp(node_embeddings.mean(dim = 0)).squeeze(-1)
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
