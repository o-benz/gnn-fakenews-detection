import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class GCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2, dropout=0.5):
        super(GCN, self).__init__()
        self.num_layers = num_layers
        self.convs = nn.ModuleList()
        
        # Input layer
        self.convs.append(GCNConv(input_dim, hidden_dim))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
        
        # Output layer
        self.convs.append(GCNConv(hidden_dim, output_dim))
        
        self.dropout = dropout

    def forward(self, x, edge_index):
        """
        Forward pass of the GCN model
        Args:
            x: Node features matrix (N x input_dim)
            edge_index: Graph connectivity in COO format (2 x E)
        Returns:
            Output node features
        """
        for i in range(self.num_layers - 1):
            x = self.convs[i](x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        x = self.convs[-1](x, edge_index)
        return F.log_softmax(x, dim=1) 