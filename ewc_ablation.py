#!/usr/bin/env python3
"""
ewc_ablation.py

Rigorous ablation study for PrimerForge.
Compares EWC-regularized fine-tuning against naive fine-tuning
and a frozen baseline to demonstrate how EWC prevents catastrophic forgetting
when adapting a primer design classifier from general (Task A) to specialized (Task B) datasets.
"""

import os
import random
import copy
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim

# ---------------------------------------------------------------------------
# Styling and Colors for Publication-Ready Figures
# ---------------------------------------------------------------------------
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans', 'Liberation Sans']
plt.rcParams['axes.edgecolor'] = '#cccccc'
plt.rcParams['axes.linewidth'] = 0.8
plt.rcParams['grid.color'] = '#eeeeee'
plt.rcParams['grid.linewidth'] = 0.5

# Colorblind-friendly palette (Okabe-Ito inspired)
COLOR_FROZEN = '#0072B2'  # Blue
COLOR_NAIVE = '#D55E00'   # Red
COLOR_EWC = '#009E73'     # Green

# ---------------------------------------------------------------------------
# Reproducibility Helper
# ---------------------------------------------------------------------------
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# ---------------------------------------------------------------------------
# Custom StandardScaler (zero external dependencies)
# ---------------------------------------------------------------------------
class StandardScaler:
    def __init__(self):
        self.mean = None
        self.std = None
        
    def fit(self, X):
        self.mean = np.mean(X, axis=0)
        self.std = np.std(X, axis=0)
        # Avoid division by zero for constant features
        self.std[self.std == 0.0] = 1.0
        
    def transform(self, X):
        return (X - self.mean) / self.std
        
    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

# ---------------------------------------------------------------------------
# PyTorch MLP Classifier Architecture
# ---------------------------------------------------------------------------
class MLPClassifier(nn.Module):
    def __init__(self, input_dim):
        super(MLPClassifier, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(16, 1)
        )
        
    def forward(self, x):
        return self.net(x)

# ---------------------------------------------------------------------------
# Elastic Weight Consolidation (EWC) Regularizer
# ---------------------------------------------------------------------------
class EWCRegularizer:
    def __init__(self, model, lambda_ewc=400.0):
        self.lambda_ewc = lambda_ewc
        self.anchor_params = {}
        self.fisher_diag = {}
        
        # Snapshot the base model parameters
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.anchor_params[name] = param.data.clone()
                self.fisher_diag[name] = torch.zeros_like(param.data)
                
    def estimate_fisher(self, model, X, y):
        """
        Estimates the empirical Fisher Information Matrix diagonal on Task A.
        Uses squared gradients of the log-likelihood (negative binary cross-entropy).
        """
        model.eval()
        criterion = nn.BCEWithLogitsLoss(reduction='none')
        
        # Reset Fisher diagonal
        for name in self.fisher_diag:
            self.fisher_diag[name].zero_()
            
        num_samples = len(X)
        for i in range(num_samples):
            model.zero_grad()
            inputs = X[i : i + 1]
            targets = y[i : i + 1]
            
            outputs = model(inputs)
            # Log-likelihood is negative BCE loss
            loss = criterion(outputs, targets)
            loss.backward()
            
            for name, param in model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    self.fisher_diag[name] += param.grad.data.pow(2)
                    
        # Average over samples
        for name in self.fisher_diag:
            self.fisher_diag[name] /= num_samples
            
        # Update the anchor parameters to current weights
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.anchor_params[name] = param.data.clone()
                
    def ewc_loss(self, model):
        """
        Returns EWC regularization penalty: \lambda \cdot \sum F_i(\theta_i - \theta*_i)^2
        """
        loss = 0.0
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.fisher_diag:
                fisher = self.fisher_diag[name]
                anchor = self.anchor_params[name]
                loss += torch.sum(fisher * (param - anchor).pow(2))
        return self.lambda_ewc * loss

# ---------------------------------------------------------------------------
# Dataset Simulation
# ---------------------------------------------------------------------------
def load_and_simulate_tasks():
    df = pd.read_csv("data/train_cluster_split.csv")
    
    # Compute GC content average
    if 'f_gc' in df.columns and 'r_gc' in df.columns:
        gc_content = (df['f_gc'] + df['r_gc']) / 2.0
    else:
        def compute_gc(seq):
            if not isinstance(seq, str) or len(seq) == 0:
                return 0.0
            return sum(1 for b in seq.upper() if b in "GC") / len(seq) * 100.0
        gc_content = df['sequence_fwd'].apply(compute_gc)
        
    df['gc_content'] = gc_content
    
    # Task A: rows where GC content is between 40-60% (balanced general dataset)
    task_a_df = df[(df['gc_content'] >= 40.0) & (df['gc_content'] <= 60.0)].copy()
    
    # Task B: rows where GC content is outside 45-55% range (simulating extreme lab-specific dataset)
    # We simulate a label shift in Task B: high-GC primers (GC > 55%) have label=1 due to optimized conditions
    # (e.g., addition of DMSO or betaine, specialized polymerases like Q5), which would normally fail under general Taq protocols.
    task_b_df = df[(df['gc_content'] < 45.0) | (df['gc_content'] > 55.0)].copy()
    task_b_df.loc[task_b_df['gc_content'] > 55.0, 'label'] = 1.0
    
    # Feature columns to extract
    feature_cols = [
        "f_tm", "r_tm", "tm_diff", "f_hairpin_dg", "r_hairpin_dg",
        "f_homodimer_dg", "r_homodimer_dg", "cross_dimer_dg", "f_gc", "r_gc",
        "f_len", "r_len", "f_clamp_gc", "r_clamp_gc", "f_poly_run", "r_poly_run",
        "f_3_dinuc_gc", "r_3_dinuc_gc", "f_3_dinuc_aa", "f_3_dinuc_tt",
        "r_3_dinuc_aa", "r_3_dinuc_tt", "f_3_stability", "r_3_stability",
        "target_mfe", "target_gc", "target_len", "primer_overlap",
        "f_off_targets", "r_off_targets", "f_var_dist", "r_var_dist",
        "salt_monovalent_mm", "salt_divalent_mm", "dntp_conc_mm", "polymerase_encoded"
    ]
    
    available_features = [col for col in feature_cols if col in df.columns]
    
    return task_a_df, task_b_df, available_features

# ---------------------------------------------------------------------------
# Main Evaluation Loop
# ---------------------------------------------------------------------------
def run_ablation_study():
    seeds = [42, 123, 456]
    epochs_ft = 30
    
    # Dictionary to collect results across seeds
    # Keys: (condition, epoch) -> list of [task_a_acc, task_b_acc, task_a_loss, task_b_loss]
    results_by_seed = {
        'Frozen': {epoch: [] for epoch in range(epochs_ft + 1)},
        'Naive': {epoch: [] for epoch in range(epochs_ft + 1)},
        'EWC': {epoch: [] for epoch in range(epochs_ft + 1)}
    }
    
    for seed in seeds:
        print(f"\n--- Running Seed {seed} ---")
        set_seed(seed)
        
        # Load and simulate datasets
        task_a_df, task_b_df, features = load_and_simulate_tasks()
        
        X_task_a = task_a_df[features].fillna(0.0).values
        y_task_a = task_a_df['label'].values
        
        X_task_b = task_b_df[features].fillna(0.0).values
        y_task_b = task_b_df['label'].values
        
        # Standardize features (scale using Task A parameters)
        scaler = StandardScaler()
        X_task_a_scaled = scaler.fit_transform(X_task_a)
        X_task_b_scaled = scaler.transform(X_task_b)
        
        # Split Task A into 80/20 train/validation for tracking base training metrics
        indices_a = np.arange(len(X_task_a_scaled))
        np.random.shuffle(indices_a)
        split_idx = int(0.8 * len(X_task_a_scaled))
        train_idx, val_idx = indices_a[:split_idx], indices_a[split_idx:]
        
        # Create PyTorch datasets
        X_train_a = torch.tensor(X_task_a_scaled[train_idx], dtype=torch.float32)
        y_train_a = torch.tensor(y_task_a[train_idx], dtype=torch.float32).unsqueeze(1)
        
        X_val_a = torch.tensor(X_task_a_scaled[val_idx], dtype=torch.float32)
        y_val_a = torch.tensor(y_task_a[val_idx], dtype=torch.float32).unsqueeze(1)
        
        # Full task datasets for tracking
        X_task_a_tensor = torch.tensor(X_task_a_scaled, dtype=torch.float32)
        y_task_a_tensor = torch.tensor(y_task_a, dtype=torch.float32).unsqueeze(1)
        
        X_task_b_tensor = torch.tensor(X_task_b_scaled, dtype=torch.float32)
        y_task_b_tensor = torch.tensor(y_task_b, dtype=torch.float32).unsqueeze(1)
        
        # 1. Train Base MLP Classifier on Task A
        input_dim = len(features)
        base_model = MLPClassifier(input_dim)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(base_model.parameters(), lr=0.001)
        
        print("Training base model on Task A (50 epochs)...")
        for epoch in range(1, 51):
            base_model.train()
            optimizer.zero_grad()
            outputs = base_model(X_train_a)
            loss = criterion(outputs, y_train_a)
            loss.backward()
            optimizer.step()
            
            # Record val stats on final epoch
            if epoch == 50:
                base_model.eval()
                with torch.no_grad():
                    val_outputs = base_model(X_val_a)
                    val_loss = criterion(val_outputs, y_val_a).item()
                    val_preds = (torch.sigmoid(val_outputs) >= 0.5).float()
                    val_acc = (val_preds == y_val_a).float().mean().item() * 100.0
                print(f"Base model final validation loss: {val_loss:.4f}, accuracy: {val_acc:.2f}%")
        
        # Evaluate pre-fine-tuning metrics (Epoch 0)
        base_model.eval()
        with torch.no_grad():
            # Task A
            out_a = base_model(X_task_a_tensor)
            loss_a_pre = criterion(out_a, y_task_a_tensor).item()
            acc_a_pre = ((torch.sigmoid(out_a) >= 0.5).float() == y_task_a_tensor).float().mean().item() * 100.0
            
            # Task B
            out_b = base_model(X_task_b_tensor)
            loss_b_pre = criterion(out_b, y_task_b_tensor).item()
            acc_b_pre = ((torch.sigmoid(out_b) >= 0.5).float() == y_task_b_tensor).float().mean().item() * 100.0
            
        print(f"Pre-fine-tuning Task A accuracy: {acc_a_pre:.2f}%, Task B accuracy: {acc_b_pre:.2f}%")
        
        # -------------------------------------------------------------------
        # Condition A: Frozen Baseline (no weight updates)
        # -------------------------------------------------------------------
        for epoch in range(epochs_ft + 1):
            results_by_seed['Frozen'][epoch].append([acc_a_pre, acc_b_pre, loss_a_pre, loss_b_pre])
            
        # -------------------------------------------------------------------
        # Condition B: Naive Fine-Tuning
        # -------------------------------------------------------------------
        print("Running Naive Fine-Tuning on Task B...")
        model_naive = copy.deepcopy(base_model)
        optimizer_naive = optim.Adam(model_naive.parameters(), lr=0.001)
        
        # Record epoch 0
        results_by_seed['Naive'][0].append([acc_a_pre, acc_b_pre, loss_a_pre, loss_b_pre])
        
        for epoch in range(1, epochs_ft + 1):
            model_naive.train()
            optimizer_naive.zero_grad()
            outputs = model_naive(X_task_b_tensor)
            loss = criterion(outputs, y_task_b_tensor)
            loss.backward()
            optimizer_naive.step()
            
            # Evaluate
            model_naive.eval()
            with torch.no_grad():
                out_a = model_naive(X_task_a_tensor)
                loss_a = criterion(out_a, y_task_a_tensor).item()
                acc_a = ((torch.sigmoid(out_a) >= 0.5).float() == y_task_a_tensor).float().mean().item() * 100.0
                
                out_b = model_naive(X_task_b_tensor)
                loss_b = criterion(out_b, y_task_b_tensor).item()
                acc_b = ((torch.sigmoid(out_b) >= 0.5).float() == y_task_b_tensor).float().mean().item() * 100.0
                
            results_by_seed['Naive'][epoch].append([acc_a, acc_b, loss_a, loss_b])
            
        # -------------------------------------------------------------------
        # Condition C: EWC Fine-Tuning
        # -------------------------------------------------------------------
        print("Running EWC Fine-Tuning on Task B...")
        model_ewc = copy.deepcopy(base_model)
        optimizer_ewc = optim.Adam(model_ewc.parameters(), lr=0.001)
        
        # Estimate Fisher Information Matrix diagonal on Task A using the base model
        ewc_reg = EWCRegularizer(model_ewc, lambda_ewc=400.0)
        ewc_reg.estimate_fisher(base_model, X_task_a_tensor, y_task_a_tensor)
        
        # Record epoch 0
        results_by_seed['EWC'][0].append([acc_a_pre, acc_b_pre, loss_a_pre, loss_b_pre])
        
        for epoch in range(1, epochs_ft + 1):
            model_ewc.train()
            optimizer_ewc.zero_grad()
            outputs = model_ewc(X_task_b_tensor)
            bce_loss = criterion(outputs, y_task_b_tensor)
            penalty = ewc_reg.ewc_loss(model_ewc)
            total_loss = bce_loss + penalty
            total_loss.backward()
            optimizer_ewc.step()
            
            # Evaluate
            model_ewc.eval()
            with torch.no_grad():
                out_a = model_ewc(X_task_a_tensor)
                loss_a = criterion(out_a, y_task_a_tensor).item()
                acc_a = ((torch.sigmoid(out_a) >= 0.5).float() == y_task_a_tensor).float().mean().item() * 100.0
                
                out_b = model_ewc(X_task_b_tensor)
                loss_b = criterion(out_b, y_task_b_tensor).item()
                acc_b = ((torch.sigmoid(out_b) >= 0.5).float() == y_task_b_tensor).float().mean().item() * 100.0
                
            results_by_seed['EWC'][epoch].append([acc_a, acc_b, loss_a, loss_b])

    # ---------------------------------------------------------------------------
    # Calculate Means and Standard Deviations
    # ---------------------------------------------------------------------------
    conditions = ['Frozen', 'Naive', 'EWC']
    summary_data = []  # To write to CSV
    
    # Structure to hold statistical results for plotting
    plot_stats = {cond: {
        'epoch': list(range(epochs_ft + 1)),
        'task_a_acc_mean': [], 'task_a_acc_std': [],
        'task_b_acc_mean': [], 'task_b_acc_std': [],
        'task_a_loss_mean': [], 'task_a_loss_std': [],
        'task_b_loss_mean': [], 'task_b_loss_std': []
    } for cond in conditions}
    
    for cond in conditions:
        for epoch in range(epochs_ft + 1):
            runs = np.array(results_by_seed[cond][epoch])  # Shape: (3, 4) -> columns: [acc_a, acc_b, loss_a, loss_b]
            mean_vals = np.mean(runs, axis=0)
            std_vals = np.std(runs, axis=0)
            
            plot_stats[cond]['task_a_acc_mean'].append(mean_vals[0])
            plot_stats[cond]['task_a_acc_std'].append(std_vals[0])
            plot_stats[cond]['task_b_acc_mean'].append(mean_vals[1])
            plot_stats[cond]['task_b_acc_std'].append(std_vals[1])
            plot_stats[cond]['task_a_loss_mean'].append(mean_vals[2])
            plot_stats[cond]['task_a_loss_std'].append(std_vals[2])
            plot_stats[cond]['task_b_loss_mean'].append(mean_vals[3])
            plot_stats[cond]['task_b_loss_std'].append(std_vals[3])
            
            summary_data.append({
                'condition': cond,
                'epoch': epoch,
                'task_a_acc': mean_vals[0],
                'task_b_acc': mean_vals[1],
                'task_a_loss': mean_vals[2],
                'task_b_loss': mean_vals[3]
            })
            
    # Save results to CSV
    os.makedirs("data", exist_ok=True)
    results_df = pd.DataFrame(summary_data)
    results_df.to_csv("data/ewc_ablation_results.csv", index=False)
    print("\nSaved study results to data/ewc_ablation_results.csv")
    
    # ---------------------------------------------------------------------------
    # Generate Publication-Ready Figures
    # ---------------------------------------------------------------------------
    os.makedirs("plots", exist_ok=True)
    epochs = np.arange(epochs_ft + 1)
    
    # Fig A: Task A Accuracy Retention
    fig_a, ax_a = plt.subplots(figsize=(6.5, 4.5), dpi=300)
    
    # Base accuracy reference
    base_acc_mean = plot_stats['Frozen']['task_a_acc_mean'][0]
    ax_a.axhline(base_acc_mean, color='#888888', linestyle='--', linewidth=1.0, label='Pre-fine-tuning Baseline')
    
    for cond, color, marker in zip(['Frozen', 'Naive', 'EWC'], [COLOR_FROZEN, COLOR_NAIVE, COLOR_EWC], ['o', 's', '^']):
        mean = np.array(plot_stats[cond]['task_a_acc_mean'])
        std = np.array(plot_stats[cond]['task_a_acc_std'])
        ax_a.plot(epochs, mean, label=f'{cond}', color=color, linewidth=1.5, marker=marker, markersize=4, markevery=3)
        ax_a.fill_between(epochs, mean - std, mean + std, color=color, alpha=0.15)
        
    ax_a.set_title("Catastrophic Forgetting vs EWC Retention on Task A", fontsize=11, fontweight='bold', pad=12)
    ax_a.set_xlabel("Fine-tuning Epoch", fontsize=10, labelpad=8)
    ax_a.set_ylabel("Task A Accuracy (%)", fontsize=10, labelpad=8)
    ax_a.set_xlim(-0.5, epochs_ft + 0.5)
    ax_a.set_ylim(40, 105)
    ax_a.grid(True, linestyle=':', alpha=0.6)
    ax_a.legend(loc='lower left', frameon=True, facecolor='white', edgecolor='#dddddd', fontsize=9)
    plt.tight_layout()
    fig_a.savefig("plots/ewc_fig_task_a_retention.pdf", format='pdf', bbox_inches='tight')
    fig_a.savefig("plots/ewc_fig_task_a_retention.png", format='png', bbox_inches='tight', dpi=300)
    plt.close(fig_a)
    
    # Fig B: Task B Adaptation Rate
    fig_b, ax_b = plt.subplots(figsize=(6.5, 4.5), dpi=300)
    
    for cond, color, marker in zip(['Naive', 'EWC'], [COLOR_NAIVE, COLOR_EWC], ['s', '^']):
        mean = np.array(plot_stats[cond]['task_b_acc_mean'])
        std = np.array(plot_stats[cond]['task_b_acc_std'])
        ax_b.plot(epochs, mean, label=f'{cond}', color=color, linewidth=1.5, marker=marker, markersize=4, markevery=3)
        ax_b.fill_between(epochs, mean - std, mean + std, color=color, alpha=0.15)
        
    ax_b.set_title("Task B Adaptation Rate: Naive Fine-Tuning vs EWC", fontsize=11, fontweight='bold', pad=12)
    ax_b.set_xlabel("Fine-tuning Epoch", fontsize=10, labelpad=8)
    ax_b.set_ylabel("Task B Accuracy (%)", fontsize=10, labelpad=8)
    ax_b.set_xlim(-0.5, epochs_ft + 0.5)
    ax_b.set_ylim(40, 105)
    ax_b.grid(True, linestyle=':', alpha=0.6)
    ax_b.legend(loc='lower right', frameon=True, facecolor='white', edgecolor='#dddddd', fontsize=9)
    plt.tight_layout()
    fig_b.savefig("plots/ewc_fig_task_b_adaptation.pdf", format='pdf', bbox_inches='tight')
    fig_b.savefig("plots/ewc_fig_task_b_adaptation.png", format='png', bbox_inches='tight', dpi=300)
    plt.close(fig_b)
    
    print("Generated retention and adaptation figures in plots/ (PDF and PNG).")
    
    # ---------------------------------------------------------------------------
    # Print Summary Table to stdout
    # ---------------------------------------------------------------------------
    print("\n" + "=" * 85)
    print("                 ELASTIC WEIGHT CONSOLIDATION (EWC) ABLATION STUDY SUMMARY")
    print("=" * 85)
    print(f"{'Condition':<20} | {'Task A Acc (Pre)':<18} | {'Task A Acc (Post)':<18} | {'Task B Acc (Post)':<18}")
    print("-" * 85)
    
    for cond in conditions:
        pre_acc_mean = plot_stats[cond]['task_a_acc_mean'][0]
        pre_acc_std = plot_stats[cond]['task_a_acc_std'][0]
        
        post_a_mean = plot_stats[cond]['task_a_acc_mean'][-1]
        post_a_std = plot_stats[cond]['task_a_acc_std'][-1]
        
        post_b_mean = plot_stats[cond]['task_b_acc_mean'][-1]
        post_b_std = plot_stats[cond]['task_b_acc_std'][-1]
        
        pre_str = f"{pre_acc_mean:.2f}% ± {pre_acc_std:.2f}%"
        post_a_str = f"{post_a_mean:.2f}% ± {post_a_std:.2f}%"
        post_b_str = f"{post_b_mean:.2f}% ± {post_b_std:.2f}%"
        
        print(f"{cond:<20} | {pre_str:<18} | {post_a_str:<18} | {post_b_str:<18}")
        
    print("=" * 85)

if __name__ == "__main__":
    run_ablation_study()
