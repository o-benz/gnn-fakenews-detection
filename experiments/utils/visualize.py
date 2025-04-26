import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import json
from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd

def plot_model_comparison(results):
    """Create a bar plot comparing model performances."""
    df = pd.DataFrame({
        'Model': list(results.keys()),
        'Accuracy': [data['test_accuracy'] for data in results.values()],
        'Loss': [data['test_loss'] for data in results.values()]
    })
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot accuracy
    sns.barplot(x='Model', y='Accuracy', data=df, ax=ax1)
    ax1.set_title('Test Accuracy Comparison')
    ax1.set_ylim(0, 1)
    
    # Plot loss
    sns.barplot(x='Model', y='Loss', data=df, ax=ax2)
    ax2.set_title('Test Loss Comparison')
    
    plt.tight_layout()
    os.makedirs('results/comparison', exist_ok=True)
    plt.savefig('results/comparison/model_comparison.png')
    plt.close()

def plot_training_curves_comparison(results):
    """Plot training and validation curves for all models."""
    plt.figure(figsize=(15, 5))
    
    # Define colors for each model
    colors = {
        'gcn': '#2ecc71',    # Green
        'gat': '#e74c3c',    # Red
        'dhgat': '#3498db'   # Blue
    }
    
    # Plot training loss
    plt.subplot(1, 2, 1)
    for model_name, data in results.items():
        color = colors.get(model_name.lower(), '#000000')
        plt.plot(data['train_losses'], color=color, label=f'{model_name.upper()} (Train)')
        plt.plot(data['val_losses'], color=color, linestyle='--', label=f'{model_name.upper()} (Val)')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot training accuracy
    plt.subplot(1, 2, 2)
    for model_name, data in results.items():
        color = colors.get(model_name.lower(), '#000000')
        plt.plot(data['train_accuracies'], color=color, label=f'{model_name.upper()} (Train)')
        plt.plot(data['val_accuracies'], color=color, linestyle='--', label=f'{model_name.upper()} (Val)')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    os.makedirs('results/comparison', exist_ok=True)
    plt.savefig('results/comparison/training_curves.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_confusion_matrices_comparison(results):
    """Plot confusion matrices for all models."""
    n_models = len(results)
    fig, axes = plt.subplots(1, n_models, figsize=(5*n_models, 4))
    if n_models == 1:
        axes = [axes]
    
    for ax, (model_name, data) in zip(axes, results.items()):
        cm = data['confusion_matrix']
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='Blues', ax=ax)
        ax.set_title(f'{model_name.upper()} Confusion Matrix')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
    
    plt.tight_layout()
    os.makedirs('results/comparison', exist_ok=True)
    plt.savefig('results/comparison/confusion_matrices.png')
    plt.close()

def create_performance_table(results):
    """Create a performance comparison table."""
    metrics = {
        'Model': [],
        'Test Accuracy': [],
        'Test Loss': [],
        'Training Time (s)': [],
        'Parameters Count': []
    }
    
    for model_name, data in results.items():
        metrics['Model'].append(model_name.upper())
        metrics['Test Accuracy'].append(data['test_accuracy'])
        metrics['Test Loss'].append(data['test_loss'])
        metrics['Training Time (s)'].append(data['training_time'])
        metrics['Parameters Count'].append(data['parameters_count'])
    
    df = pd.DataFrame(metrics)
    
    # Save as CSV
    os.makedirs('results/comparison', exist_ok=True)
    df.to_csv('results/comparison/performance_metrics.csv', index=False)
    
    # Save as LaTeX table
    latex_table = df.to_latex(index=False, float_format=lambda x: '{:.4f}'.format(x))
    with open('results/comparison/performance_metrics.tex', 'w') as f:
        f.write(latex_table)
    
    return df

def plot_class_performance_comparison(results):
    """Plot per-class performance metrics for all models."""
    # Define class names
    class_names = ['pants-fire', 'false', 'barely-true', 'half-true', 'mostly-true', 'true']
    metrics = ['precision', 'recall', 'f1-score']
    
    # Create figure with subplots for each metric
    fig, axes = plt.subplots(len(metrics), 1, figsize=(12, 4*len(metrics)))
    
    # Define colors for each model
    colors = {
        'gcn': '#2ecc71',    # Green
        'gat': '#e74c3c',    # Red
        'dhgat': '#3498db'   # Blue
    }
    
    # For each metric
    for idx, metric in enumerate(metrics):
        data_dict = {
            'Class': class_names * len(results),
            'Model': [model.upper() for model in results.keys() for _ in class_names],
            'Score': []
        }
        
        # Collect scores for each model
        for model_name, model_data in results.items():
            class_report = model_data['classification_report']
            for class_name in class_names:
                data_dict['Score'].append(class_report[class_name][metric])
        
        # Create DataFrame and plot
        df = pd.DataFrame(data_dict)
        sns.barplot(x='Class', y='Score', hue='Model', data=df, ax=axes[idx])
        
        axes[idx].set_title(f'Per-class {metric.capitalize()}')
        axes[idx].set_xlabel('Class')
        axes[idx].set_ylabel(metric.capitalize())
        axes[idx].tick_params(axis='x', rotation=45)
        axes[idx].grid(True, alpha=0.3)
    
    plt.tight_layout()
    os.makedirs('results/comparison', exist_ok=True)
    plt.savefig('results/comparison/class_performance.png', dpi=300, bbox_inches='tight')
    plt.close()

def visualize_results(results):
    """Generate all visualizations for the results."""
    os.makedirs('results/comparison', exist_ok=True)
    
    plot_training_curves_comparison(results)
    plot_confusion_matrices_comparison(results)
    plot_model_comparison(results)
    plot_class_performance_comparison(results)
    create_performance_table(results) 