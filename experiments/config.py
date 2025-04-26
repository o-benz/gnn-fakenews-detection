class Config:
    # General training parameters
    seed = 42
    batch_size = 32
    learning_rate = 0.0005
    weight_decay = 0.003
    early_stopping_patience = 20
    
    # Model specific parameters
    class GCN:
        input_dim = 310  # 300 (content) + 10 (social)
        hidden_dim = 96
        output_dim = 6
        num_layers = 2
        dropout = 0.3
        num_epochs = 50
    
    class GAT:
        input_dim = 310
        hidden_dim = 48
        output_dim = 6
        num_layers = 2
        dropout = 0.3
        num_heads = 8
        num_epochs = 50
        
    class DHGAT:
        content_dim = 300
        social_dim = 10
        hidden_dim = 64  # Base hidden dimension
        output_dim = 6
        num_layers = 2
        dropout = 0.3
        attention_dropout = 0.2
        num_heads = 8
        num_epochs = 100  # Increased epochs for better convergence 