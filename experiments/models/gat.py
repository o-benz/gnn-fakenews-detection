import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv

class GAT(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_heads=8, num_layers=2, 
                 dropout=0.5, attention_dropout=0.2, residual=True):
        super(GAT, self).__init__()
        self.num_layers = num_layers
        self.residual = residual
        self.convs = nn.ModuleList()
        
        # Input layer with multiple attention heads
        self.convs.append(
            GATConv(input_dim, hidden_dim, heads=num_heads, 
                   dropout=attention_dropout, add_self_loops=True)
        )
        
        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(
                GATConv(hidden_dim * num_heads, hidden_dim, heads=num_heads,
                       dropout=attention_dropout, add_self_loops=True)
            )
        
        # Output layer (we use 1 attention head for the final prediction)
        self.convs.append(
            GATConv(hidden_dim * num_heads, output_dim, heads=1,
                   concat=False, dropout=attention_dropout, add_self_loops=True)
        )
        
        # Residual linear transformations if needed
        if self.residual:
            self.residual_layers = nn.ModuleList()
            self.residual_layers.append(
                nn.Linear(input_dim, hidden_dim * num_heads)
            )
            for _ in range(num_layers - 2):
                self.residual_layers.append(
                    nn.Linear(hidden_dim * num_heads, hidden_dim * num_heads)
                )
            self.residual_layers.append(
                nn.Linear(hidden_dim * num_heads, output_dim)
            )
        
        self.dropout = dropout
        self.attention_dropout = attention_dropout

    def forward(self, x, edge_index):
        """
        Forward pass of the GAT model
        Args:
            x: Node features matrix (N x input_dim)
            edge_index: Graph connectivity in COO format (2 x E)
        Returns:
            Output node features
        """
        # Store original input for residual
        previous_x = x
        
        for i in range(self.num_layers - 1):
            # GAT layer
            x = self.convs[i](x, edge_index)
            
            # Add residual if enabled
            if self.residual:
                res = self.residual_layers[i](previous_x)
                if x.shape == res.shape:
                    x = x + res
            
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            previous_x = x
        
        # Final layer
        x = self.convs[-1](x, edge_index)
        if self.residual:
            res = self.residual_layers[-1](previous_x)
            if x.shape == res.shape:
                x = x + res
                
        return F.log_softmax(x, dim=1) 