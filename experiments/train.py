import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import os
from tqdm import tqdm
import json
from datetime import datetime
import time
import torch.nn.functional as F
from torch_geometric.loader import NeighborLoader

from models.gcn import GCN
from models.gat import GAT
from models.dhgat import DHGAT
from utils.data import load_data
from utils.visualize import visualize_results
from config import Config

def count_parameters(model):
    """Count number of trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def save_experiment_details(model_name, model, results_dir, train_data, val_data, test_data):
    """Save experiment configuration and dataset details"""
    details = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'model_name': model_name,
        'parameters_count': count_parameters(model),
        'dataset_sizes': {
            'train': len(train_data.y),
            'val': len(val_data.y),
            'test': len(test_data.y)
        },
        'hyperparameters': {
            'learning_rate': Config.learning_rate,
            'weight_decay': Config.weight_decay,
            'num_epochs': Config.num_epochs,
            'early_stopping_patience': Config.early_stopping_patience,
        }
    }
    
    # Add model-specific hyperparameters
    if model_name == 'gcn':
        details['model_config'] = {
            'input_dim': Config.GCN.input_dim,
            'hidden_dim': Config.GCN.hidden_dim,
            'output_dim': Config.GCN.output_dim,
            'num_layers': Config.GCN.num_layers,
            'dropout': Config.GCN.dropout
        }
    elif model_name == 'gat':
        details['model_config'] = {
            'input_dim': Config.GAT.input_dim,
            'hidden_dim': Config.GAT.hidden_dim,
            'output_dim': Config.GAT.output_dim,
            'num_heads': Config.GAT.num_heads,
            'num_layers': Config.GAT.num_layers,
            'dropout': Config.GAT.dropout
        }
    else:  # dhgat
        details['model_config'] = {
            'content_dim': Config.DHGAT.content_dim,
            'social_dim': Config.DHGAT.social_dim,
            'hidden_dim': Config.DHGAT.hidden_dim,
            'output_dim': Config.DHGAT.output_dim,
            'num_heads': Config.DHGAT.num_heads,
            'num_layers': Config.DHGAT.num_layers,
            'dropout': Config.DHGAT.dropout
        }
    
    with open(os.path.join(results_dir, 'experiment_details.json'), 'w') as f:
        json.dump(details, f, indent=4)
    
    # Print summary
    print("\nExperiment Details:")
    print(f"Model: {model_name.upper()}")
    print(f"Number of trainable parameters: {details['parameters_count']:,}")
    print("\nDataset sizes:")
    print(f"Train: {details['dataset_sizes']['train']:,}")
    print(f"Validation: {details['dataset_sizes']['val']:,}")
    print(f"Test: {details['dataset_sizes']['test']:,}")
    print("\nHyperparameters:")
    for k, v in details['hyperparameters'].items():
        print(f"{k}: {v}")
    print("\nModel Configuration:")
    for k, v in details['model_config'].items():
        print(f"{k}: {v}")
    print("\n" + "="*50 + "\n")

def train_model(model, train_data, val_data, optimizer, device, epochs=100, patience=10):
    model.train()
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []
    best_val_loss = float('inf')
    patience_counter = 0
    start_time = time.time()

    # Move data to device
    train_data = train_data.to(device)
    val_data = val_data.to(device)

    for epoch in range(epochs):
        # Training
        model.train()
        optimizer.zero_grad()
        
        # Handle different model types
        if isinstance(model, DHGAT):
            out = model(train_data.content_x, train_data.social_x,
                       train_data.content_edge_index, train_data.social_edge_index)
        else:  # GCN or GAT
            out = model(train_data.x, train_data.edge_index)
        
        loss = F.cross_entropy(out, train_data.y)
        loss.backward()
        optimizer.step()
        
        # Calculate training metrics
        pred = out.argmax(dim=1)
        train_acc = pred.eq(train_data.y).sum().item() / train_data.y.size(0)
        train_loss = loss.item()
        
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)

        # Validation
        model.eval()
        with torch.no_grad():
            if isinstance(model, DHGAT):
                out = model(val_data.content_x, val_data.social_x,
                          val_data.content_edge_index, val_data.social_edge_index)
            else:  # GCN or GAT
                out = model(val_data.x, val_data.edge_index)
            
            val_loss = F.cross_entropy(out, val_data.y).item()
            pred = out.argmax(dim=1)
            val_acc = pred.eq(val_data.y).sum().item() / val_data.y.size(0)
            
            val_losses.append(val_loss)
            val_accuracies.append(val_acc)

        print(f'Epoch {epoch:03d}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, '
              f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), f'results/best_model_{model.__class__.__name__.lower()}.pt')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f'Early stopping triggered at epoch {epoch}')
                break

    training_time = time.time() - start_time
    return {
        'train_losses': train_losses,
        'train_accuracies': train_accuracies,
        'val_losses': val_losses,
        'val_accuracies': val_accuracies,
        'training_time': training_time
    }

def evaluate_model(model, test_data, device):
    model.eval()
    test_data = test_data.to(device)
    
    with torch.no_grad():
        # Handle different model types
        if isinstance(model, DHGAT):
            out = model(test_data.content_x, test_data.social_x,
                       test_data.content_edge_index, test_data.social_edge_index)
        else:  # GCN or GAT
            out = model(test_data.x, test_data.edge_index)
        
        test_loss = F.cross_entropy(out, test_data.y).item()
        pred = out.argmax(dim=1)
        test_accuracy = pred.eq(test_data.y).sum().item() / test_data.y.size(0)
        
        # Get predictions and labels for confusion matrix
        all_preds = pred.cpu().numpy()
        all_labels = test_data.y.cpu().numpy()
        cm = confusion_matrix(all_labels, all_preds)
    
    return {
        'test_loss': test_loss,
        'test_accuracy': test_accuracy,
        'confusion_matrix': cm,
        'predictions': all_preds,
        'true_labels': all_labels
    }

def main():
    # Set random seed for reproducibility
    torch.manual_seed(Config.seed)
    np.random.seed(Config.seed)
    
    # Create results directory
    os.makedirs('results', exist_ok=True)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # Load data
    train_data, val_data, test_data, num_features, num_classes = load_data()
    
    # Initialize results dictionary
    results = {}
    
    # Train and evaluate models
    models = {
        'gcn': GCN(
            input_dim=Config.GCN.input_dim,
            hidden_dim=Config.GCN.hidden_dim,
            output_dim=Config.GCN.output_dim,
            num_layers=Config.GCN.num_layers,
            dropout=Config.GCN.dropout
        ),
        'gat': GAT(
            input_dim=Config.GAT.input_dim,
            hidden_dim=Config.GAT.hidden_dim,
            output_dim=Config.GAT.output_dim,
            num_layers=Config.GAT.num_layers,
            dropout=Config.GAT.dropout,
            num_heads=Config.GAT.num_heads
        ),
        'dhgat': DHGAT(
            content_dim=Config.DHGAT.content_dim,
            social_dim=Config.DHGAT.social_dim,
            hidden_dim=Config.DHGAT.hidden_dim,
            output_dim=Config.DHGAT.output_dim,
            num_layers=Config.DHGAT.num_layers,
            dropout=Config.DHGAT.dropout,
            num_heads=Config.DHGAT.num_heads
        )
    }
    
    for model_name, model in models.items():
        print(f'\nTraining {model_name.upper()}...')
        model = model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=Config.learning_rate, 
                                   weight_decay=Config.weight_decay)
        
        # Train
        training_results = train_model(
            model, train_data, val_data, optimizer, device,
            epochs=100,  # Let early stopping determine when to stop
            patience=Config.early_stopping_patience
        )
        
        # Load best model for evaluation
        model.load_state_dict(torch.load(f'results/best_model_{model_name}.pt'))
        
        # Evaluate
        test_results = evaluate_model(model, test_data, device)
        
        # Combine results
        results[model_name] = {
            **training_results,
            **test_results,
            'parameters_count': sum(p.numel() for p in model.parameters())
        }
        
        print(f'{model_name.upper()} Test Loss: {test_results["test_loss"]:.4f}, '
              f'Test Accuracy: {test_results["test_accuracy"]:.4f}')
    
    # Generate visualizations
    visualize_results(results)

if __name__ == '__main__':
    main() 