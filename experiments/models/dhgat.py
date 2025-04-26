import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv

class DualChannelAttention(nn.Module):
    def __init__(self, dim):
        super(DualChannelAttention, self).__init__()
        self.content_attention = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.ReLU(),
            nn.Linear(dim // 2, 1)
        )
        self.social_attention = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.ReLU(),
            nn.Linear(dim // 2, 1)
        )
        
    def forward(self, content_features, social_features):
        content_score = torch.sigmoid(self.content_attention(content_features))
        social_score = torch.sigmoid(self.social_attention(social_features))
        
        attention_weights = torch.cat([content_score, social_score], dim=1)
        attention_weights = F.softmax(attention_weights, dim=1)
        
        return attention_weights

class DHGAT(nn.Module):
    def __init__(self, content_dim, social_dim, hidden_dim, output_dim, num_heads=8, num_layers=2, 
                 dropout=0.5, attention_dropout=0.2):
        super(DHGAT, self).__init__()
        self.num_layers = num_layers
        
        # Content channel with larger capacity
        self.content_convs = nn.ModuleList()
        self.content_convs.append(
            GATConv(content_dim, hidden_dim * 2, heads=num_heads, 
                   dropout=attention_dropout, add_self_loops=True)
        )
        for _ in range(num_layers - 2):
            self.content_convs.append(
                GATConv(hidden_dim * 2 * num_heads, hidden_dim * 2, heads=num_heads,
                       dropout=attention_dropout, add_self_loops=True)
            )
        self.content_convs.append(
            GATConv(hidden_dim * 2 * num_heads, hidden_dim, heads=num_heads,
                   dropout=attention_dropout, add_self_loops=True)
        )
            
        # Social channel
        self.social_convs = nn.ModuleList()
        self.social_convs.append(
            GATConv(social_dim, hidden_dim, heads=num_heads,
                   dropout=attention_dropout, add_self_loops=True)
        )
        for _ in range(num_layers - 2):
            self.social_convs.append(
                GATConv(hidden_dim * num_heads, hidden_dim, heads=num_heads,
                       dropout=attention_dropout, add_self_loops=True)
            )
        self.social_convs.append(
            GATConv(hidden_dim * num_heads, hidden_dim, heads=num_heads,
                   dropout=attention_dropout, add_self_loops=True)
        )
            
        # Dual-channel attention
        self.channel_attention = DualChannelAttention(hidden_dim * num_heads)
        
        # Additional feature processing
        self.content_bn = nn.BatchNorm1d(hidden_dim * num_heads)
        self.social_bn = nn.BatchNorm1d(hidden_dim * num_heads)
        
        # Final classification layers with residual connections
        self.fc1 = nn.Linear(hidden_dim * num_heads * 2, hidden_dim * 2)
        self.fc2 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        
        self.dropout = dropout
        self.attention_dropout = attention_dropout

    def forward(self, content_x, social_x, content_edge_index, social_edge_index):
        # Process content channel
        content_x_orig = content_x
        for i, conv in enumerate(self.content_convs):
            content_x = conv(content_x, content_edge_index)
            content_x = F.elu(content_x)
            content_x = F.dropout(content_x, p=self.dropout, training=self.training)
            if i < len(self.content_convs) - 1:  # Skip connection except for last layer
                if content_x.shape == content_x_orig.shape:
                    content_x = content_x + content_x_orig
                content_x_orig = content_x
            
        # Process social channel
        social_x_orig = social_x
        for i, conv in enumerate(self.social_convs):
            social_x = conv(social_x, social_edge_index)
            social_x = F.elu(social_x)
            social_x = F.dropout(social_x, p=self.dropout, training=self.training)
            if i < len(self.social_convs) - 1:  # Skip connection except for last layer
                if social_x.shape == social_x_orig.shape:
                    social_x = social_x + social_x_orig
                social_x_orig = social_x
            
        # Apply batch normalization
        content_x = self.content_bn(content_x)
        social_x = self.social_bn(social_x)
        
        # Apply dual-channel attention
        attention_weights = self.channel_attention(content_x, social_x)
        content_attended = content_x * attention_weights[:, 0].unsqueeze(1)
        social_attended = social_x * attention_weights[:, 1].unsqueeze(1)
        
        # Concatenate and classify with residual connections
        x = torch.cat([content_attended, social_attended], dim=1)
        x1 = F.dropout(F.elu(self.fc1(x)), p=self.dropout, training=self.training)
        x2 = F.dropout(F.elu(self.fc2(x1)), p=self.dropout, training=self.training)
        if x2.shape == x1.shape:
            x2 = x2 + x1
        x = self.fc3(x2)
        
        return F.log_softmax(x, dim=1) 