import pandas as pd
import os
import torch
from torch_geometric.data import Data
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder, StandardScaler
from utils.util import create_folder

# Global scalers to ensure consistent normalization across splits
content_scaler = None
social_scaler = None

def check_features(features, name):
    """Helper function to check for NaN values and extreme values"""
    if np.isnan(features).any():
        print(f"Warning: NaN values found in {name} features")
    if np.isinf(features).any():
        print(f"Warning: Infinite values found in {name} features")
    print(f"{name} features stats - min: {np.min(features)}, max: {np.max(features)}, mean: {np.mean(features)}, std: {np.std(features)}")

def load_and_prepare_LIAR():
    """
    Load and prepare the LIAR dataset from local TSV files
    Returns:
        pd.DataFrame: The prepared dataset
    """
    ds_dir = 'experiments/data/liar_dataset'
    ds_file = os.path.join('data', 'liar_df.csv')

    if os.path.isfile(ds_file):
        print("LIAR processed dataset already exists, loading from file :)\n")
        return pd.read_csv(ds_file)

    # Column names for the TSV files
    columns = ['id', 'label', 'statement', 'subject', 'speaker', 'speaker_job', 
               'state', 'party', 'barely_true_counts', 'false_counts', 'half_true_counts', 
               'mostly_true_counts', 'pants_on_fire_counts', 'context']

    print('Loading LIAR dataset from TSV files...')
    # Load train, validation and test sets
    df_train = pd.read_csv(os.path.join(ds_dir, 'train.tsv'), sep='\t', names=columns)
    df_val = pd.read_csv(os.path.join(ds_dir, 'valid.tsv'), sep='\t', names=columns)
    df_test = pd.read_csv(os.path.join(ds_dir, 'test.tsv'), sep='\t', names=columns)

    # Add split type
    df_train['df_type'] = 'train'
    df_val['df_type'] = 'val'
    df_test['df_type'] = 'test'

    # Merge all splits
    print('Merging Train, Val and Test in one dataset file...')
    df_full = pd.concat([df_train, df_val, df_test], axis=0)
    df_full.reset_index(drop=True, inplace=True)
    print(f'Merged dataset shape: {df_full.shape}')

    # Save processed dataset
    print('Saving processed LIAR dataset to CSV file...')
    create_folder('data')  # Ensure the data directory exists
    df_full.to_csv(ds_file, sep=',', encoding='utf-8', index=False)

    print('load_and_prepare_LIAR: DONE :)\n')
    return df_full

def create_content_features(df):
    """
    Create content features using TF-IDF on statement text
    Args:
        df: DataFrame containing the dataset
    Returns:
        np.ndarray: Content features matrix
        int: Dimension of content features
    """
    tfidf = TfidfVectorizer(max_features=300, stop_words='english')
    content_features = tfidf.fit_transform(df['statement']).toarray()
    return content_features, content_features.shape[1]

def create_social_features(df):
    """
    Create social features from speaker metadata
    Args:
        df: DataFrame containing the dataset
    Returns:
        np.ndarray: Social features matrix
        int: Dimension of social features
    """
    # Fill missing values with 'unknown' for categorical columns
    categorical_columns = ['speaker', 'speaker_job', 'state', 'party', 'context']
    for col in categorical_columns:
        df[col] = df[col].fillna('unknown')
    
    # Fill missing values with 0 for numerical columns
    numerical_columns = ['barely_true_counts', 'false_counts', 'half_true_counts', 
                        'mostly_true_counts', 'pants_on_fire_counts']
    for col in numerical_columns:
        df[col] = df[col].fillna(0)
    
    # Encode categorical columns
    le = LabelEncoder()
    social_features = []
    
    # Encode categorical features
    for col in categorical_columns:
        encoded = le.fit_transform(df[col])
        social_features.append(encoded.reshape(-1, 1))
    
    # Add numerical features
    numerical_features = df[numerical_columns].values
    social_features.append(numerical_features)
    
    # Concatenate all features
    social_features = np.concatenate(social_features, axis=1)
    
    # Ensure no NaN or infinite values
    social_features = np.nan_to_num(social_features, nan=0.0, posinf=0.0, neginf=0.0)
    
    return social_features, social_features.shape[1]

def create_edge_index(features, k=5):
    """
    Create edge index based on k-nearest neighbors in feature space
    Args:
        features: Feature matrix
        k: Number of neighbors
    Returns:
        torch.Tensor: Edge index in COO format
    """
    # Compute pairwise distances
    dist = torch.cdist(torch.tensor(features), torch.tensor(features))
    
    # Get k nearest neighbors for each node
    _, indices = torch.topk(dist, k=k+1, dim=1, largest=False)
    
    # Create edge index (excluding self-loops)
    rows = torch.arange(len(features)).unsqueeze(1).repeat(1, k)
    cols = indices[:, 1:k+1]  # Exclude the first neighbor (self)
    
    edge_index = torch.stack([rows.flatten(), cols.flatten()])
    return edge_index

def prepare_graph_data(df, split='train'):
    """
    Prepare graph data for training/validation/testing
    Args:
        df: DataFrame containing the dataset
        split: One of 'train', 'val', 'test'
    Returns:
        Data: PyTorch Geometric Data object with connected nodes
    """
    global content_scaler, social_scaler
    
    # Filter data by split
    df_split = df[df['df_type'] == split]
    
    # Create features
    content_features, content_dim = create_content_features(df_split)
    social_features, social_dim = create_social_features(df_split)
    
    # Check raw features
    if split == 'train':
        print("\nFeature statistics before normalization:")
        check_features(content_features, "Content")
        check_features(social_features, "Social")
    
    # Handle potential infinite values
    content_features = np.clip(content_features, -1e6, 1e6)
    social_features = np.clip(social_features, -1e6, 1e6)
    
    # Initialize scalers on training data, apply to all splits
    if split == 'train':
        content_scaler = StandardScaler().fit(content_features)
        social_scaler = StandardScaler().fit(social_features)
    
    # Transform features using fitted scalers
    content_features = content_scaler.transform(content_features)
    social_features = social_scaler.transform(social_features)
    
    # Check normalized features
    if split == 'train':
        print("\nFeature statistics after normalization:")
        check_features(content_features, "Content")
        check_features(social_features, "Social")
    
    # Handle any remaining numerical instabilities
    content_features = np.nan_to_num(content_features, nan=0.0, posinf=1.0, neginf=-1.0)
    social_features = np.nan_to_num(social_features, nan=0.0, posinf=1.0, neginf=-1.0)
    
    # Convert labels (map string labels to integers)
    label_mapping = {
        'pants-fire': 0, 'false': 1, 'barely-true': 2,
        'half-true': 3, 'mostly-true': 4, 'true': 5
    }
    labels = df_split['label'].map(label_mapping).values
    
    # Convert features to torch tensors
    content_features = torch.FloatTensor(content_features)
    social_features = torch.FloatTensor(social_features)
    labels = torch.LongTensor(labels)
    
    # Create edge indices based on feature similarity
    k = 10  # Number of neighbors for each node
    
    # Create edges based on content similarity
    content_edge_index = create_edge_index(content_features, k=k)
    
    # Create edges based on social similarity
    social_edge_index = create_edge_index(social_features, k=k)
    
    # Combine content and social features
    combined_features = torch.cat([content_features, social_features], dim=1)
    
    # Create a single PyG Data object for the entire graph
    data = Data(
        x=combined_features,
        edge_index=content_edge_index,  # Use content-based edges by default for GCN/GAT
        content_x=content_features,
        social_x=social_features,
        content_edge_index=content_edge_index,
        social_edge_index=social_edge_index,
        y=labels
    )
    
    return data

def load_data():
    """
    Main function to load and prepare all data
    Returns:
        tuple: (train_data, val_data, test_data, num_features, num_classes)
    """
    # Load raw data
    df = load_and_prepare_LIAR()
    
    # Prepare graph data for each split
    train_data = prepare_graph_data(df, 'train')
    val_data = prepare_graph_data(df, 'val')
    test_data = prepare_graph_data(df, 'test')
    
    # Get feature dimensions
    num_features = train_data.x.size(1)
    num_classes = 6  # Number of label classes
    
    return train_data, val_data, test_data, num_features, num_classes 