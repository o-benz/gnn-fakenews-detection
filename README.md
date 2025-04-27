# Graph Neural Networks for Fake News Detection

This project implements and evaluates several Graph Neural Network (GNN) architectures for the task of fake news detection using the LIAR dataset. The models leverage both textual content and social metadata to classify statements into fine-grained truthfulness categories.

## Dataset

The [LIAR dataset](https://www.cs.ucsb.edu/~william/data/liar_dataset.zip) contains 12,791 manually labeled short statements collected from PolitiFact. Each statement is annotated with one of six veracity levels:
- Pants-fire
- False
- Barely-true
- Half-true
- Mostly-true
- True

Each record also includes:
- The text of the statement
- The speaker's name, political party, and state
- The context in which the statement was made
- Historical credibility counts for the speaker

## Approach

The problem is formulated as a graph-based classification task:
- Each **statement** is treated as a **node**.
- **Edges** are constructed based on **$k$-nearest neighbors** similarity (with $k=10$) in the feature space.
- Two separate graphs are built:
  - A **content graph** based on TF-IDF textual similarity.
  - A **social graph** based on speaker metadata similarity.

Node features include:
- **Content features**: 300-dimensional TF-IDF vectors of statement text.
- **Social features**: 10-dimensional vectors encoding speaker metadata.

## Implemented Models

- **GCN (Graph Convolutional Network)**  
  A baseline model applying graph convolutions over the combined content and social features.

- **GAT (Graph Attention Network)**  
  A baseline model incorporating attention mechanisms over neighbors to improve feature aggregation.

- **DHGAT (Decision-based Heterogeneous Graph Attention Network, adapted)**  
  Our proposed architecture:
  - Processes content and social graphs through separate GAT-based pipelines.
  - Applies a learned dual-channel attention mechanism to dynamically weight the two information sources.
  - Combines the attended features for final classification.

## Installation

1. Clone the repository:
```bash
git clone https://github.com/o-benz/gnn-fakenews-detection.git
cd gnn-fakenews-detection
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

## Usage

To train and evaluate all models (GCN, GAT, DHGAT):
```bash
python experiments/train.py
```

Trained model checkpoints, training curves, evaluation reports, and confusion matrices will be saved under the `results/` directory.

## Results

| Model | Test Accuracy | Test Loss | Training Time |
|:-----|:--------------|:---------|:--------------|
| GCN  | 18.8%          | 1.794    | ~18 seconds   |
| GAT  | 20.3%          | 1.856    | ~34 seconds   |
| DHGAT | 20.3%         | 1.748    | ~36 seconds   |

- DHGAT achieves the best test loss among the models.
- Visualization outputs include training curves and confusion matrices for deeper error analysis.

## Future Work

- Extending DHGAT to full heterogeneous graphs with multiple node and edge types.
- Modeling dynamic propagation patterns using temporal GNNs.
- Evaluating on richer datasets capturing real user engagement networks (e.g., Twitter, Weibo).

---

## License

This project is licensed under the [MIT License](LICENSE).

---

## Acknowledgments

This project was developed as part of the INF8225 course on Probabilistic and Learning Techniques in Artificial Intelligence at Polytechnique Montréal, under the guidance of our professor Christopher J. Pal and with support from our TA Anthony Gosselin.

### Team Members
- Ely Cheikh Abass
- Omar Benzekri
- Anis Abdeladim
