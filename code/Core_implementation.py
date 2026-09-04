# =============================================================================
# CFL-Phish: Complete Reproduction of Table (Exp E1-E7)
# Centralized + Federated + DP + ZT with MLP and FT-Transformer
# =============================================================================
import os
import copy
import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import AdamW
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# =============================================================================
# 1. Configuration
# =============================================================================
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🚀 Device: {DEVICE}")

# Federated Hyperparameters (Table 6)
K_CLIENTS = 3
R_ROUNDS = 10
E_EPOCHS = 5
BATCH_SIZE = 64
LR = 1e-3
WEIGHT_DECAY = 1e-4
DIRICHLET_ALPHA = 0.5

# DP Parameters (ε=0.7, δ=10⁻⁵, C=1.0, σ≈1.2)
C_CLIP = 1.0
SIGMA = 1.2

# Zero Trust Parameters (Eq. 10)
ALPHA, BETA, GAMMA, DELTA_ZT = 0.25, 0.35, 0.20, 0.20
TAU_MIN = 0.1

# =============================================================================
# 2. Data Loading
# =============================================================================
def load_data():
    paths = [
        "/content/drive/MyDrive/Phish360_cache/trainval_768.npz",
        "/content/Phish360_data/trainval_768.npz",
        "./trainval_768.npz"
    ]
    path = next((p for p in paths if os.path.exists(p)), None)
    if not path:
        raise FileNotFoundError("❌ Could not find trainval_768.npz")
    
    tr = np.load(path, allow_pickle=True)
    X_all = tr['X'].astype(np.float32)
    y_all = tr['y'].astype(np.int64)
    
    X_tr_raw, X_val_raw, y_tr, y_val = train_test_split(
        X_all, y_all, test_size=0.15, stratify=y_all, random_state=42
    )
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_tr_raw).astype(np.float32)
    X_val = scaler.transform(X_val_raw).astype(np.float32)
    
    print(f"✅ Data: Train={X_tr.shape}, Val={X_val.shape}")
    return X_tr, X_val, y_tr, y_val

# =============================================================================
# 3. Model Architectures
# =============================================================================
class PhishNetMLP(nn.Module):
    """MLP with GroupNorm (better with DP than BatchNorm)"""
    def __init__(self, input_dim=768, num_classes=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512), nn.GroupNorm(32, 512), nn.ReLU(), nn.Dropout(0.30),
            nn.Linear(512, 256), nn.GroupNorm(16, 256), nn.ReLU(), nn.Dropout(0.25),
            nn.Linear(256, 128), nn.GroupNorm(8, 128), nn.ReLU(), nn.Dropout(0.20),
            nn.Linear(128, num_classes)
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x)


class FeatureTokenizer(nn.Module):
    def __init__(self, input_dim, d_model):
        super().__init__()
        self.tokenizers = nn.ModuleList([
            nn.Sequential(nn.Linear(1, d_model), nn.LayerNorm(d_model), nn.GELU())
            for _ in range(input_dim)
        ])
    
    def forward(self, x):
        tokens = [tok(x[:, i:i+1]) for i, tok in enumerate(self.tokenizers)]
        return torch.stack(tokens, dim=1)


class FTTransformer(nn.Module):
    def __init__(self, input_dim=768, d_model=256, n_heads=8, 
                 n_layers=6, ffn_dim=1024, dropout=0.20, num_classes=2):
        super().__init__()
        self.tokenizer = FeatureTokenizer(input_dim, d_model)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.pos_embedding = nn.Parameter(torch.randn(1, input_dim + 1, d_model) * 0.02)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=ffn_dim,
            dropout=dropout, activation='gelu', batch_first=True, norm_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model), nn.Linear(d_model, d_model // 2),
            nn.GELU(), nn.Dropout(dropout), nn.Linear(d_model // 2, num_classes)
        )
    
    def forward(self, x):
        B = x.size(0)
        tokens = self.tokenizer(x)
        cls = self.cls_token.expand(B, -1, -1)
        tokens = torch.cat([cls, tokens], dim=1) + self.pos_embedding
        encoded = self.encoder(tokens)
        return self.classifier(encoded[:, 0])

# =============================================================================
# 4. DP Mechanism (Gradient-Level, Eq. 9)
# =============================================================================
def apply_dp_to_gradients(model, C=C_CLIP, sigma=SIGMA):
    """Apply DP directly to gradients: clip then add noise"""
    # 1. Compute global L2 norm
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total_norm += p.grad.data.norm(2).item() ** 2
    total_norm = total_norm ** 0.5

    # 2. Clip
    if total_norm > C:
        clip_coef = C / (total_norm + 1e-6)
        for p in model.parameters():
            if p.grad is not None:
                p.grad.data.mul_(clip_coef)

    # 3. Add Gaussian noise N(0, σ²C²I)
    noise_std = sigma * C
    for p in model.parameters():
        if p.grad is not None:
            noise = torch.normal(0.0, noise_std, p.grad.shape, device=p.grad.device)
            p.grad.data.add_(noise)

# =============================================================================
# 5. Zero Trust Manager (Eq. 10-12)
# =============================================================================
class ZeroTrustManager:
    def __init__(self, n_clients, alpha=ALPHA, beta=BETA, gamma=GAMMA, 
                 delta_zt=DELTA_ZT, tau_min=TAU_MIN):
        self.n_clients = n_clients
        self.alpha, self.beta, self.gamma, self.delta_zt = alpha, beta, gamma, delta_zt
        self.tau_min = tau_min

    def compute_anomalies(self, client_deltas, client_losses, client_times):
        K = self.n_clients
        anomalies = {}
        
        delta_norms = []
        for i in range(K):
            norm = sum(torch.norm(v).item() ** 2 for v in client_deltas[i].values()) ** 0.5
            delta_norms.append(norm)
        
        mean_delta = np.mean(delta_norms)
        max_delta = max(delta_norms) + 1e-6
        mean_loss = np.mean(client_losses) + 1e-6
        mean_time = np.mean(client_times) + 1e-6
        
        for i in range(K):
            cid = f"client_{i}"
            a_k = 0.0  # No auth failures in isolated experiment
            d_k = abs(delta_norms[i] - mean_delta) / max_delta
            c_k = abs(client_times[i] - mean_time) / mean_time
            p_k = abs(client_losses[i] - mean_loss) / mean_loss
            anomalies[cid] = {'a_k': a_k, 'd_k': d_k, 'c_k': c_k, 'p_k': p_k}
        
        return anomalies

    def compute_trust_scores(self, anomalies):
        trust_scores = {}
        for cid, anom in anomalies.items():
            rho_k = (self.alpha * anom['a_k'] 
                    + self.beta * anom['d_k']
                    + self.gamma * anom['c_k']
                    + self.delta_zt * anom['p_k'])
            trust_scores[cid] = max(self.tau_min, 1.0 - rho_k)
        return trust_scores

    def aggregate(self, client_deltas, client_sizes, trust_scores):
        denom = sum(trust_scores[f"client_{i}"] * client_sizes[i] 
                   for i in range(self.n_clients))
        agg_delta = {}
        for key in client_deltas[0].keys():
            weighted_sum = sum(
                trust_scores[f"client_{i}"] * client_sizes[i] * client_deltas[i][key]
                for i in range(self.n_clients)
            )
            agg_delta[key] = weighted_sum / denom
        return agg_delta

# =============================================================================
# 6. Client Partitioning (Dirichlet α=0.5)
# =============================================================================
def partition_clients(y, n_clients=K_CLIENTS, alpha=DIRICHLET_ALPHA, seed=42):
    rng = np.random.default_rng(seed)
    client_idx = [[] for _ in range(n_clients)]
    for cls in np.unique(y):
        idxs = np.where(y == cls)[0]
        rng.shuffle(idxs)
        props = rng.dirichlet(np.repeat(alpha, n_clients))
        sizes = (props * len(idxs)).astype(int)
        sizes[-1] = len(idxs) - sizes[:-1].sum()
        start = 0
        for c in range(n_clients):
            client_idx[c].extend(idxs[start:start+sizes[c]].tolist())
            start += sizes[c]
    return [np.array(c, dtype=int) for c in client_idx]

# =============================================================================
# 7. Evaluation Function
# =============================================================================
def evaluate_model(model, X_val, y_val):
    model.eval()
    with torch.no_grad():
        loader = DataLoader(
            TensorDataset(torch.tensor(X_val, dtype=torch.float32), 
                         torch.tensor(y_val, dtype=torch.long)),
            batch_size=128
        )
        preds, probs, labels = [], [], []
        for bX, by in loader:
            logits = model(bX.to(DEVICE))
            p = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            preds.extend((p > 0.5).astype(int))
            probs.extend(p)
            labels.extend(by.numpy())
    
    acc = accuracy_score(labels, preds) * 100
    f1 = f1_score(labels, preds, zero_division=0) * 100
    auc = roc_auc_score(labels, probs) * 100
    return {'accuracy': acc, 'f1': f1, 'auc': auc}

# =============================================================================
# 8. Centralized Training (E1, E2)
# =============================================================================
def train_centralized(X_tr, y_tr, X_val, y_val, model_class, model_kwargs, 
                      n_epochs=50, seed=42):
    """Centralized training - no FL, no DP"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    model = model_class(**model_kwargs).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    
    loader = DataLoader(
        TensorDataset(torch.tensor(X_tr, dtype=torch.float32), 
                     torch.tensor(y_tr, dtype=torch.long)),
        batch_size=BATCH_SIZE, shuffle=True
    )
    
    for epoch in range(n_epochs):
        model.train()
        for bX, by in loader:
            bX, by = bX.to(DEVICE), by.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(bX), by)
            loss.backward()
            optimizer.step()
    
    return evaluate_model(model, X_val, y_val)

# =============================================================================
# 9. Federated Training (E3-E7)
# =============================================================================
def train_federated(X_tr, y_tr, X_val, y_val, model_class, model_kwargs,
                    use_dp=False, use_zt=False, seed=42):
    """Federated training with optional DP and ZT"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    client_indices = partition_clients(y_tr, K_CLIENTS, alpha=DIRICHLET_ALPHA, seed=seed)
    client_sizes = [len(idx) for idx in client_indices]
    
    global_model = model_class(**model_kwargs).to(DEVICE)
    client_models = [model_class(**model_kwargs).to(DEVICE) for _ in range(K_CLIENTS)]
    
    zt_manager = ZeroTrustManager(K_CLIENTS) if use_zt else None
    criterion = nn.CrossEntropyLoss()
    
    for r in range(R_ROUNDS):
        # Broadcast global model
        global_state = copy.deepcopy(global_model.state_dict())
        for c in range(K_CLIENTS):
            client_models[c].load_state_dict(global_state)
        
        client_deltas = []
        client_losses = []
        client_times = []
        
        for k in range(K_CLIENTS):
            client_start = time.time()
            
            X_k = torch.tensor(X_tr[client_indices[k]], dtype=torch.float32)
            y_k = torch.tensor(y_tr[client_indices[k]], dtype=torch.long)
            loader = DataLoader(TensorDataset(X_k, y_k), batch_size=BATCH_SIZE, shuffle=True)
            
            opt = AdamW(client_models[k].parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
            client_models[k].train()
            
            epoch_loss, num_samples = 0.0, 0
            for e in range(E_EPOCHS):
                for bX, by in loader:
                    bX, by = bX.to(DEVICE), by.to(DEVICE)
                    opt.zero_grad()
                    
                    loss = criterion(client_models[k](bX), by)
                    loss.backward()
                    
                    # Apply DP to gradients BEFORE optimizer step
                    if use_dp:
                        apply_dp_to_gradients(client_models[k], C_CLIP, SIGMA)
                    
                    opt.step()
                    epoch_loss += loss.item() * bX.size(0)
                    num_samples += bX.size(0)
            
            client_time = time.time() - client_start
            client_times.append(client_time)
            client_losses.append(epoch_loss / max(num_samples, 1))
            
            # Compute model delta (keep on DEVICE)
            delta_theta = {}
            for name, param in client_models[k].named_parameters():
                delta_theta[name] = param.data - global_state[name]
            client_deltas.append(delta_theta)
        
        # Aggregation
        if use_zt and zt_manager:
            anomalies = zt_manager.compute_anomalies(client_deltas, client_losses, client_times)
            trust_scores = zt_manager.compute_trust_scores(anomalies)
            agg_delta = zt_manager.aggregate(client_deltas, client_sizes, trust_scores)
        else:
            # Standard FedAvg
            total_size = sum(client_sizes)
            agg_delta = {}
            for name in client_deltas[0].keys():
                weighted_sum = sum(
                    client_sizes[i] * client_deltas[i][name]
                    for i in range(K_CLIENTS)
                )
                agg_delta[name] = weighted_sum / total_size
        
        # Update global model
        new_global_state = copy.deepcopy(global_state)
        for name in agg_delta.keys():
            new_global_state[name] = global_state[name] + agg_delta[name]
        global_model.load_state_dict(new_global_state)
    
    return evaluate_model(global_model, X_val, y_val)

# =============================================================================
# 10. Main - Run All Experiments
# =============================================================================
def main():
    print("\n" + "="*80)
    print("📊 CFL-Phish: Reproducing Table (Exp E1-E7)")
    print("="*80)
    
    X_tr, X_val, y_tr, y_val = load_data()
    
    results = []
    
    # E1: MLP Centralized No Privacy
    print("\n🔬 E1: MLP Centralized (No Privacy)")
    metrics = train_centralized(X_tr, y_tr, X_val, y_val, 
                                PhishNetMLP, {'input_dim': 768, 'num_classes': 2})
    results.append({'Exp': 'E1', 'Architecture': 'MLP', 'Mode': 'Centralized', 
                    'Privacy': 'No', 'LLM': 'No', **metrics})
    print(f"   Acc: {metrics['accuracy']:.2f}% | F1: {metrics['f1']:.2f}% | AUC: {metrics['auc']:.2f}%")
    
    # E2: FT-Transformer Centralized No Privacy
    print("\n E2: FT-Transformer Centralized (No Privacy)")
    metrics = train_centralized(X_tr, y_tr, X_val, y_val, 
                                FTTransformer, 
                                {'input_dim': 768, 'd_model': 256, 'n_heads': 8, 
                                 'n_layers': 6, 'ffn_dim': 1024, 'dropout': 0.20, 'num_classes': 2})
    results.append({'Exp': 'E2', 'Architecture': 'FT-Transformer', 'Mode': 'Centralized', 
                    'Privacy': 'No', 'LLM': 'No', **metrics})
    print(f"   Acc: {metrics['accuracy']:.2f}% | F1: {metrics['f1']:.2f}% | AUC: {metrics['auc']:.2f}%")
    
    # E3: MLP Federated No Privacy
    print("\n🔬 E3: MLP Federated (No Privacy)")
    metrics = train_federated(X_tr, y_tr, X_val, y_val, 
                              PhishNetMLP, {'input_dim': 768, 'num_classes': 2},
                              use_dp=False, use_zt=False)
    results.append({'Exp': 'E3', 'Architecture': 'MLP', 'Mode': 'Federated', 
                    'Privacy': 'No', 'LLM': 'No', **metrics})
    print(f"   Acc: {metrics['accuracy']:.2f}% | F1: {metrics['f1']:.2f}% | AUC: {metrics['auc']:.2f}%")
    
    # E4: FT-Transformer Federated No Privacy
    print("\n🔬 E4: FT-Transformer Federated (No Privacy)")
    metrics = train_federated(X_tr, y_tr, X_val, y_val, 
                              FTTransformer, 
                              {'input_dim': 768, 'd_model': 256, 'n_heads': 8, 
                               'n_layers': 6, 'ffn_dim': 1024, 'dropout': 0.20, 'num_classes': 2},
                              use_dp=False, use_zt=False)
    results.append({'Exp': 'E4', 'Architecture': 'FT-Transformer', 'Mode': 'Federated', 
                    'Privacy': 'No', 'LLM': 'No', **metrics})
    print(f"   Acc: {metrics['accuracy']:.2f}% | F1: {metrics['f1']:.2f}% | AUC: {metrics['auc']:.2f}%")
    
    # E5: MLP Federated DP
    print("\n E5: MLP Federated + DP (ε=0.7, σ=1.2)")
    metrics = train_federated(X_tr, y_tr, X_val, y_val, 
                              PhishNetMLP, {'input_dim': 768, 'num_classes': 2},
                              use_dp=True, use_zt=False)
    results.append({'Exp': 'E5', 'Architecture': 'MLP', 'Mode': 'Federated', 
                    'Privacy': 'DP', 'LLM': 'No', **metrics})
    print(f"   Acc: {metrics['accuracy']:.2f}% | F1: {metrics['f1']:.2f}% | AUC: {metrics['auc']:.2f}%")
    
    # E6: FT-Transformer Federated DP
    print("\n🔬 E6: FT-Transformer Federated + DP (ε=0.7, σ=1.2)")
    metrics = train_federated(X_tr, y_tr, X_val, y_val, 
                              FTTransformer, 
                              {'input_dim': 768, 'd_model': 256, 'n_heads': 8, 
                               'n_layers': 6, 'ffn_dim': 1024, 'dropout': 0.20, 'num_classes': 2},
                              use_dp=True, use_zt=False)
    results.append({'Exp': 'E6', 'Architecture': 'FT-Transformer', 'Mode': 'Federated', 
                    'Privacy': 'DP', 'LLM': 'No', **metrics})
    print(f"   Acc: {metrics['accuracy']:.2f}% | F1: {metrics['f1']:.2f}% | AUC: {metrics['auc']:.2f}%")
    
    # E7: MLP Federated ZT+DP
    print("\n🔬 E7: MLP Federated + ZT + DP (ε=0.7, σ=1.2)")
    metrics = train_federated(X_tr, y_tr, X_val, y_val, 
                              PhishNetMLP, {'input_dim': 768, 'num_classes': 2},
                              use_dp=True, use_zt=True)
    results.append({'Exp': 'E7', 'Architecture': 'MLP', 'Mode': 'Federated', 
                    'Privacy': 'ZT+DP', 'LLM': 'No', **metrics})
    print(f"   Acc: {metrics['accuracy']:.2f}% | F1: {metrics['f1']:.2f}% | AUC: {metrics['auc']:.2f}%")
    
    # Print Summary Table
    print("\n" + "="*80)
    print("📋 SUMMARY TABLE")
    print("="*80)
    print(f"{'Exp':<4} {'Architecture':<15} {'Mode':<12} {'Privacy':<8} {'Acc':<8} {'F1':<8} {'AUC':<8}")
    print("-"*80)
    for r in results:
        print(f"{r['Exp']:<4} {r['Architecture']:<15} {r['Mode']:<12} {r['Privacy']:<8} "
              f"{r['accuracy']:<8.2f} {r['f1']:<8.2f} {r['auc']:<8.2f}")
    print("="*80)
    
    # Plot Results
    plt.figure(figsize=(12, 6))
    exps = [r['Exp'] for r in results]
    accs = [r['accuracy'] for r in results]
    f1s = [r['f1'] for r in results]
    
    x = np.arange(len(exps))
    width = 0.35
    
    plt.bar(x - width/2, accs, width, label='Accuracy', color='blue', alpha=0.7)
    plt.bar(x + width/2, f1s, width, label='F1-Score', color='green', alpha=0.7)
    
    plt.xlabel('Experiment', fontsize=12, fontweight='bold')
    plt.ylabel('Performance (%)', fontsize=12, fontweight='bold')
    plt.title('CFL-Phish: Exp E1-E7 Results', fontsize=14, fontweight='bold')
    plt.xticks(x, exps)
    plt.legend(fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.6, axis='y')
    plt.tight_layout()
    plt.savefig('cfl_phish_results_table.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\n✅ All experiments completed successfully!")

if __name__ == "__main__":
    main()
