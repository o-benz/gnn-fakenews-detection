import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv

class GAT(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_heads=8, num_layers=2, dropout=0.5):
        super(GAT, self).__init__()
        self.num_layers = num_layers
        self.convs = nn.ModuleList()
        
        # Input layer with multiple attention heads
        self.convs.append(GATConv(input_dim, hidden_dim, heads=num_heads, dropout=dropout))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(
                GATConv(hidden_dim * num_heads, hidden_dim, heads=num_heads, dropout=dropout)
            )
        
        # Output layer (we use 1 attention head for the final prediction)
        self.convs.append(
            GATConv(hidden_dim * num_heads, output_dim, heads=1, concat=False, dropout=dropout)
        )
        
        self.dropout = dropout

    def forward(self, x, edge_index):
        """
        Forward pass of the GAT model
        Args:
            x: Node features matrix (N x input_dim)
            edge_index: Graph connectivity in COO format (2 x E)
        Returns:
            Output node features
        """
        for i in range(self.num_layers - 1):
            x = self.convs[i](x, edge_index)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        x = self.convs[-1](x, edge_index)
        return F.log_softmax(x, dim=1) 