
# CPSC 483 Final Project
**By Maryanne Xu and Leo deJong**

## Description
This project explores various initialization methods for Graph Neural Networks (GNNs), focusing particularly on MLPInit and its impact on training efficiency and accuracy. We compare MLPInit with Xavier and random initialization method across different GNN architectures. Additionally, we implement a Graph Attention Network (GAT) to see whether it outperforms other architectures.

Based on following paper: "Embarrassingly Simple GNN Training Acceleration with MLP Initialization" by Han et al.

## Repository Contents
- `requirements.txt` or `environment.yml`: Lists of all the dependencies.
- `MLPwGAT.sh`: Script to reproduce results for the implemented GAT.
- `README.md`: Instructions for reproducing the results.

## Setup and Installation
To set up the environment and install the necessary dependencies, follow these steps:

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/leo-dejong/MLPinit.git
   ```

2. **Install Dependencies:**
   - Using `requirements.txt`:
     ```bash
     pip install -r requirements.txt
     ```
   - Using `environment.yml` (for Conda environments):
     ```bash
     conda env create -f environment.yml
     conda activate mlpinit
     ```

## Reproducing Results

### Training Examples
- **For GraphSAGE Training with MLPInit:**
  ```bash
  python -u src/main.py --batch_size 1000 --dataset ogbn-arxiv --dim_hidden 512 --dropout 0.5 --epochs 50 --eval_steps 1 --lr 0.001 --num_layers 4 --random_seed 31415 --save_dir . --train_model_type gnn --gnn_model GraphSAGE --weight_decay 0 --pretrained_checkpoint ./ogbn-arxiv_GraphSAGE_mlp_512_4_31415.pt
  ```

- **For PeerMLP Training:**
  ```bash
  python -u src/main.py --batch_size 1000 --dataset ogbn-arxiv --dim_hidden 512 --dropout 0.5 --epochs 50 --eval_steps 1 --lr 0.001 --num_layers 4 --random_seed 31415 --save_dir . --train_model_type mlp --gnn_model GraphSAGE --weight_decay 0
  ```

### Reproducing GAT Results
Execute the `MLPwGAT.sh` script to reproduce the GAT results:
```bash
sh MLPwGAT.sh
```

## Codebase Sources
The project is based on:
- [MLPInit for GNNs](https://github.com/snap-research/MLPInit-for-GNNs/tree/main)
- The GNN architectures and training code from [Large_Scale_GCN_Benchmarking](https://github.com/VITA-Group/Large_Scale_GCN_Benchmarking).

---
