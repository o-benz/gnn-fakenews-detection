import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv

class DualChannelAttention(nn.Module):
    def __init__(self, dim):
        super(DualChannelAttention, self).__init__()
        self.content_attention = nn.Linear(dim, 1)
        self.social_attention = nn.Linear(dim, 1)
        
    def forward(self, content_features, social_features):
        content_score = torch.sigmoid(self.content_attention(content_features))
        social_score = torch.sigmoid(self.social_attention(social_features))
        
        attention_weights = torch.cat([content_score, social_score], dim=1)
        attention_weights = F.softmax(attention_weights, dim=1)
        
        return attention_weights

class DHGAT(nn.Module):
    def __init__(self, content_dim, social_dim, hidden_dim, output_dim, num_heads=8, num_layers=2, dropout=0.5):
        super(DHGAT, self).__init__()
        self.num_layers = num_layers
        
        # Content channel
        self.content_convs = nn.ModuleList()
        self.content_convs.append(GATConv(content_dim, hidden_dim, heads=num_heads, dropout=dropout))
        for _ in range(num_layers - 1):
            self.content_convs.append(
                GATConv(hidden_dim * num_heads, hidden_dim, heads=num_heads, dropout=dropout)
            )
            
        # Social channel
        self.social_convs = nn.ModuleList()
        self.social_convs.append(GATConv(social_dim, hidden_dim, heads=num_heads, dropout=dropout))
        for _ in range(num_layers - 1):
            self.social_convs.append(
                GATConv(hidden_dim * num_heads, hidden_dim, heads=num_heads, dropout=dropout)
            )
            
        # Dual-channel attention
        self.channel_attention = DualChannelAttention(hidden_dim * num_heads)
        
        # Final classification layers
        self.fc1 = nn.Linear(hidden_dim * num_heads * 2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        
        self.dropout = dropout

    def forward(self, content_x, social_x, content_edge_index, social_edge_index):
        """
        Forward pass of the DHGAT model
        Args:
            content_x: Content features matrix
            social_x: Social features matrix
            content_edge_index: Content graph connectivity
            social_edge_index: Social graph connectivity
        Returns:
            Output node features
        """
        # Process content channel
        for conv in self.content_convs:
            content_x = conv(content_x, content_edge_index)
            content_x = F.elu(content_x)
            content_x = F.dropout(content_x, p=self.dropout, training=self.training)
            
        # Process social channel
        for conv in self.social_convs:
            social_x = conv(social_x, social_edge_index)
            social_x = F.elu(social_x)
            social_x = F.dropout(social_x, p=self.dropout, training=self.training)
            
        # Apply dual-channel attention
        attention_weights = self.channel_attention(content_x, social_x)
        content_attended = content_x * attention_weights[:, 0].unsqueeze(1)
        social_attended = social_x * attention_weights[:, 1].unsqueeze(1)
        
        # Concatenate and classify
        x = torch.cat([content_attended, social_attended], dim=1)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.elu(self.fc1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.fc2(x)
        
        return F.log_softmax(x, dim=1) 