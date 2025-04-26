class Config:
    # General training parameters
    seed = 42
    batch_size = 64
    learning_rate = 0.0005
    weight_decay = 0.01
    early_stopping_patience = 10  # Increased patience to allow more training
    
    # Model specific parameters
    class GCN:
        input_dim = 310  # 300 (content) + 10 (social)
        hidden_dim = 32  # Reduced complexity
        output_dim = 6
        num_layers = 2
        dropout = 0.3  # Reduced dropout
        num_epochs = 18  # Specific number of epochs for GCN
    
    class GAT:
        input_dim = 310  # 300 (content) + 10 (social)
        hidden_dim = 16  # Reduced complexity
        output_dim = 6
        num_layers = 2
        dropout = 0.3  # Reduced dropout
        num_heads = 4
        num_epochs = 18  # Specific number of epochs for GAT
        
    class DHGAT:
        content_dim = 300
        social_dim = 10
        hidden_dim = 32
        output_dim = 6
        num_layers = 2
        dropout = 0.3
        num_heads = 4
        num_epochs = 30  # Specific number of epochs for DHGAT 