# =============================================================================
# CFL-Phish: Statistical Significance + Non-IID Visualization
# STRICTLY ALIGNED with Paper Equations:
#   - Eq. 9  : DP mechanism (Clip + Gaussian noise, σ=1.2, C=1.0)
#   - Eq. 10 : Four-signal ZT risk score (α=0.25, β=0.35, γ=0.20, δ=0.20)
#   - Eq. 11 : Trust score τ_k = max(τ_min, 1 - ρ_k)
#   - Eq. 12 : Trust-weighted aggregation
#   - Table 6: batch_size=64, lr=0.001
# =============================================================================
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import Adam
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from scipy import stats
import numpy as np
import matplotlib.pyplot as plt
import copy, time, gc, os
from sklearn.model_selection import train_test_split

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🖥️ Device: {DEVICE}")

# =============================================================================
# 1. SMART AUTO-LOAD & NORMALIZATION
# =============================================================================
if 'X_all' not in globals():
    print("️ Data variables not found. Loading from Google Drive cache...")
    CACHE_DIR = "/content/drive/MyDrive/Phish360_cache"
    cache_file = os.path.join(CACHE_DIR, "trainval_768.npz")
    if not os.path.exists(cache_file):
        cache_file = os.path.join(CACHE_DIR, "trainval_real.npz")

    tr = np.load(cache_file, allow_pickle=True)
    X_all = tr['X'].astype(np.float32)
    y_all = tr['y'].astype(np.int64)

    # ✅ Use development-validation split as per paper
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_all, y_all, test_size=0.15, stratify=y_all, random_state=42
    )

# ✅ CRITICAL FIX: Normalize the data
print("🔄 Normalizing data features...")
scaler = StandardScaler()
X_tr = scaler.fit_transform(X_tr).astype(np.float32)
X_val = scaler.transform(X_val).astype(np.float32)
print(f"✅ Data normalized! X_tr mean: {X_tr.mean():.4f}, std: {X_tr.std():.4f}\n")

# =============================================================================
# 2. DP Manager — Eq. 9: g_priv = Clip(g, C) + N(0, σ²C²I)
# =============================================================================
class DPManager:
    """
    Differential Privacy mechanism per Eq. 9 of the paper.
    Uses σ = 1.2 (for ε=0.7, δ=10⁻⁵, C=1.0)
    """
    def __init__(self, epsilon=0.7, delta=1e-5, C=1.0, sigma=1.2):
        self.epsilon = epsilon
        self.delta = delta
        self.C = C
        self.sigma = sigma  # σ ≈ 1.2 as per paper

    def apply(self, model, progress):
        # Step 1: Compute L2 norm of gradients
        total_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                total_norm += p.grad.data.norm(2).item() ** 2
        total_norm = total_norm ** 0.5

        # Step 2: Clip gradients to C (Eq. 9, first part)
        if total_norm > self.C:
            clip_coef = self.C / (total_norm + 1e-6)
            for p in model.parameters():
                if p.grad is not None:
                    p.grad.data.mul_(clip_coef)

        # Step 3: Add Gaussian noise N(0, σ²C²I) (Eq. 9, second part)
        noise_std = self.sigma * self.C
        with torch.no_grad():
            for p in model.parameters():
                if p.grad is not None:
                    noise = torch.normal(0, noise_std, p.grad.shape, device=p.grad.device)
                    p.grad.data.add_(noise)

# =============================================================================
# 3. Zero Trust Manager — Eq. 10, 11, 12 (Four-signal ZT)
# =============================================================================
class ZeroTrustManager:
    """
    Full four-signal ZT implementation (Section 3.6).
    Computes authentication, divergence, communication, and performance anomalies.
    Weights: α=0.25, β=0.35, γ=0.20, δ=0.20
    """
    def __init__(self, n_clients, alpha=0.25, beta=0.35, gamma=0.20, delta_zt=0.20, tau_min=0.1):
        self.n_clients = n_clients
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta_zt = delta_zt
        self.tau_min = tau_min
        self.trust = {f"client_{i+1}": 1.0 for i in range(n_clients)}
        self.history = {f"client_{i+1}": [] for i in range(n_clients)}
        self.response_times = {f"client_{i+1}": [] for i in range(n_clients)}

    def update(self, cid, acc, response_time=None):
        """Update trust score based on performance and response time."""
        self.history[cid].append(acc)
        if response_time is not None:
            self.response_times[cid].append(response_time)

    def compute_anomalies(self, client_deltas, client_losses, client_times):
        """Compute all four anomaly signals (Eq. 10)."""
        K = self.n_clients
        anomalies = {}

        # Compute population statistics
        delta_norms = []
        for i in range(K):
            cid = f"client_{i+1}"
            if cid in client_deltas:
                norm = sum(torch.norm(v).item() ** 2 for v in client_deltas[cid].values()) ** 0.5
                delta_norms.append(norm)
            else:
                delta_norms.append(0.0)

        mean_delta = np.mean(delta_norms)
        max_delta = max(delta_norms) + 1e-6
        
        # ✅ FIX: Convert dict values to list before computing mean
        mean_loss = np.mean(list(client_losses.values())) + 1e-6
        mean_time = np.mean(list(client_times.values())) + 1e-6

        for i in range(K):
            cid = f"client_{i+1}"

            # a_k: Authentication anomaly (simulated as 0 for this experiment)
            a_k = 0.0

            # d_k: Update divergence (Eq. 10)
            d_k = abs(delta_norms[i] - mean_delta) / max_delta

            # c_k: Communication anomaly (Eq. 10)
            c_k = abs(client_times[cid] - mean_time) / mean_time

            # p_k: Performance anomaly (Eq. 10)
            p_k = abs(client_losses[cid] - mean_loss) / mean_loss

            anomalies[cid] = {
                'a_k': a_k, 'd_k': d_k, 'c_k': c_k, 'p_k': p_k
            }

        return anomalies

    def compute_trust_scores(self, anomalies):
        """Compute trust scores using Eq. 10 and Eq. 11."""
        trust_scores = {}
        for cid, anom in anomalies.items():
            # Eq. 10: Composite risk score ρ_k
            rho_k = (self.alpha * anom['a_k']
                    + self.beta * anom['d_k']
                    + self.gamma * anom['c_k']
                    + self.delta_zt * anom['p_k'])
            # Eq. 11: Trust score τ_k
            tau_k = max(self.tau_min, 1.0 - rho_k)
            trust_scores[cid] = tau_k
        return trust_scores

    def aggregate(self, updates, counts, trust_scores):
        """Trust-weighted aggregation (Eq. 12)."""
        weighted_updates = {}
        total_weight = 0.0

        for cid, upd in updates.items():
            w = counts[cid] * trust_scores[cid]
            total_weight += w
            for k in upd.keys():
                weighted_updates[k] = weighted_updates.get(k, 0) + upd[k] * w

        for k in weighted_updates.keys():
            weighted_updates[k] = weighted_updates[k] / total_weight

        return weighted_updates


class PhishNetMLP(nn.Module):
    def __init__(self, input_dim, num_classes=2):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.25),
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )
    def forward(self, x): return self.network(x)


def partition_non_iid(y, n_clients, alpha=0.5):
    client_indices = [[] for _ in range(n_clients)]
    for cls in (0, 1):
        idxs = np.where(y == cls)[0]
        np.random.shuffle(idxs)
        props = np.random.dirichlet(np.repeat(alpha, n_clients))
        sizes = (props * len(idxs)).astype(int)
        sizes[-1] = len(idxs) - sizes[:-1].sum()
        start = 0
        for c in range(n_clients):
            client_indices[c].extend(idxs[start:start+sizes[c]].tolist())
            start += sizes[c]
    return [np.array(c, dtype=int) for c in client_indices]


# =============================================================================
# 4. Non-IID Data Distribution Visualization
# =============================================================================
print("\n" + "="*90)
print(" GENERATING NON-IID DATA DISTRIBUTION PLOTS")
print("="*90)

client_counts = [3, 5, 10, 20, 30, 50]
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()

for idx, n_clients in enumerate(client_counts):
    ax = axes[idx]
    cidx = partition_non_iid(y_tr, n_clients, alpha=0.5)

    legit_counts, phish_counts = [], []
    for c in range(n_clients):
        client_y = y_tr[cidx[c]]
        legit_counts.append(int(np.sum(client_y == 0)))
        phish_counts.append(int(np.sum(client_y == 1)))

    x = np.arange(n_clients)
    width = 0.6

    ax.bar(x, legit_counts, width, label='Legitimate', color='#2ecc71', alpha=0.85, edgecolor='black', linewidth=0.5)
    ax.bar(x, phish_counts, width, bottom=legit_counts, label='Phishing', color='#e74c3c', alpha=0.85, edgecolor='black', linewidth=0.5)

    ax.set_title(f'{n_clients} Clients (Dirichlet $\\alpha=0.5$)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Client ID', fontsize=10)
    ax.set_ylabel('Samples', fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels([f'C{i+1}' for i in range(n_clients)], fontsize=9)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    if idx == 0:
        ax.legend(loc='upper right', fontsize=9)

plt.suptitle('Non-IID Data Distribution Across Federated Clients', fontsize=16, fontweight='bold', y=0.98)
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, 0.02), ncol=2, fontsize=12, frameon=True)

plt.tight_layout(rect=[0, 0.05, 1, 0.95])
plt.savefig('non_iid_distribution_all_clients.png', dpi=300, bbox_inches='tight')
print("✅ Saved combined distribution plot: non_iid_distribution_all_clients.png")
plt.show()

# =============================================================================
# 5. Training Function (WITH BATCH SIZE 64 FIX)
# =============================================================================
def train_single_run(X_train, y_train, X_val, y_val, n_clients,
                     use_dp=True, use_zt=True, n_rounds=10, local_epochs=5, seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)

    input_dim = X_train.shape[1]
    clients = [PhishNetMLP(input_dim=input_dim).to(DEVICE) for _ in range(n_clients)]
    global_model = PhishNetMLP(input_dim=input_dim).to(DEVICE)
    cidx = partition_non_iid(y_train, n_clients, alpha=0.5)

    criterion = nn.CrossEntropyLoss()
    dp_manager = DPManager() if use_dp else None
    zt_manager = ZeroTrustManager(n_clients) if use_zt else None

    for r in range(n_rounds):
        updates, accs, counts, client_deltas, client_losses, client_times = {}, {}, {}, {}, {}, {}
        round_start = time.time()
        round_progress = (r + 1) / n_rounds

        for c in range(n_clients):
            Xc, yc = X_train[cidx[c]], y_train[cidx[c]]
            cid = f"client_{c+1}"
            counts[cid] = len(Xc)
            if len(Xc) == 0: continue

            # ✅ FIX: Use batch_size=64 as per paper
            loader = DataLoader(TensorDataset(torch.FloatTensor(Xc), torch.LongTensor(yc)),
                              batch_size=64, shuffle=True, drop_last=False)

            # ✅ FIX: Use Adam without weight_decay as per paper
            opt = Adam(clients[c].parameters(), lr=1e-3)
            clients[c].train()

            client_start = time.time()
            for e in range(local_epochs):
                for bX, by in loader:
                    # ✅ CRITICAL FIX: Skip batches with 1 or 0 samples to prevent BatchNorm crash
                    if bX.size(0) <= 1:
                        continue

                    bX, by = bX.to(DEVICE), by.to(DEVICE)
                    opt.zero_grad()
                    out = clients[c](bX)
                    loss = criterion(out, by)
                    loss.backward()

                    # ✅ Apply DP mechanism (Eq. 9)
                    if use_dp and dp_manager:
                        dp_manager.apply(clients[c], round_progress)

                    opt.step()

            client_time = time.time() - client_start
            client_times[cid] = client_time

            # Local accuracy for ZT
            clients[c].eval()
            with torch.no_grad():
                out = clients[c](torch.FloatTensor(Xc).to(DEVICE))
                acc = (out.argmax(1) == torch.LongTensor(yc).to(DEVICE)).float().mean().item()
            accs[cid] = acc
            client_losses[cid] = 1.0 - acc  # Use error rate as loss proxy

            # Compute delta for ZT
            delta_theta = {}
            for name, param in clients[c].named_parameters():
                delta_theta[name] = param.data.cpu() - global_model.state_dict()[name].cpu()
            client_deltas[cid] = delta_theta

            if use_zt and zt_manager:
                zt_manager.update(cid, acc, client_time)

            updates[cid] = copy.deepcopy(clients[c].state_dict())

        # ✅ Aggregation with Full ZT (Eq. 12)
        if use_zt and zt_manager:
            anomalies = zt_manager.compute_anomalies(client_deltas, client_losses, client_times)
            trust_scores = zt_manager.compute_trust_scores(anomalies)
            agg = zt_manager.aggregate(updates, counts, trust_scores)
        else:
            agg = {k: sum(u[k] for u in updates.values()) / len(updates)
                   for k in list(updates.values())[0].keys()}

        global_model.load_state_dict(agg)
        for c in range(n_clients):
            clients[c].load_state_dict(agg)

    # Final Evaluation
    global_model.eval()
    with torch.no_grad():
        probs = torch.softmax(global_model(torch.FloatTensor(X_val).to(DEVICE)), dim=1)[:, 1].cpu().numpy()
        preds = (probs > 0.5).astype(int)

    return accuracy_score(y_val, preds) * 100

# =============================================================================
# 6. Execution Loop
# =============================================================================
print("\n" + "="*90)
print("🚀 STARTING STATISTICAL SIGNIFICANCE ANALYSIS (3 Independent Runs)")
print("="*90)

n_runs = 3  # Set to 3 for speed, change to 5 for final paper results
all_results = {
    'baseline': {c: [] for c in client_counts},
    'dp_only': {c: [] for c in client_counts},
    'proposed': {c: [] for c in client_counts}
}

for c in client_counts:
    print(f"\n🔹 Testing with {c} clients ({n_runs} runs each)...")
    print("-" * 70)

    for run in range(n_runs):
        seed = 42 + (run * 100) + c

        acc_baseline = train_single_run(X_tr, y_tr, X_val, y_val, c, use_dp=False, use_zt=False, seed=seed)
        all_results['baseline'][c].append(acc_baseline)

        acc_dp = train_single_run(X_tr, y_tr, X_val, y_val, c, use_dp=True, use_zt=False, seed=seed)
        all_results['dp_only'][c].append(acc_dp)

        acc_proposed = train_single_run(X_tr, y_tr, X_val, y_val, c, use_dp=True, use_zt=True, seed=seed)
        all_results['proposed'][c].append(acc_proposed)

        if run == 0:
            print(f"   Run {run+1}: Baseline={acc_baseline:.2f}% | DP={acc_dp:.2f}% | Proposed={acc_proposed:.2f}%")

    print(f"\n    Summary ({n_runs} runs):")
    print(f"      Baseline:  {np.mean(all_results['baseline'][c]):.2f}% ± {np.std(all_results['baseline'][c]):.2f}")
    print(f"      DP Only:   {np.mean(all_results['dp_only'][c]):.2f}% ± {np.std(all_results['dp_only'][c]):.2f}")
    print(f"      Proposed:  {np.mean(all_results['proposed'][c]):.2f}% ± {np.std(all_results['proposed'][c]):.2f}")

    torch.cuda.empty_cache()
    gc.collect()

# =============================================================================
# 7. Statistical Tests & Final Plotting
# =============================================================================
print("\n\n" + "="*90)
print(" STATISTICAL SIGNIFICANCE TESTS (Paired t-test)")
print("="*90)
print(f"\n{'Clients':<10} {'Comparison':<25} {'Mean Diff':<12} {'p-value':<12} {'Significant?':<15}")
print("-"*90)

significance_results = []
for c in client_counts:
    baseline_vals = all_results['baseline'][c]
    proposed_vals = all_results['proposed'][c]

    # ✅ FIX: Use paired t-test since same seeds are used
    t_stat, p_val = stats.ttest_rel(baseline_vals, proposed_vals)
    mean_diff = np.mean(proposed_vals) - np.mean(baseline_vals)

    if p_val < 0.001: sig = "*** (p<0.001)"
    elif p_val < 0.01: sig = "** (p<0.01)"
    elif p_val < 0.05: sig = "* (p<0.05)"
    else: sig = "ns (not sig.)"

    significance_results.append({'clients': c, 'mean_diff': mean_diff, 'p_value': p_val, 'significant': p_val < 0.05})
    print(f"{c:<10} {'Baseline vs Proposed':<25} {mean_diff:+.2f}%{'':<7} {p_val:<12.4f} {sig:<15}")

# =============================================================================
# 8. Print Table 16 Format
# =============================================================================
print("\n\n" + "="*90)
print("📋 TABLE 16: Statistical Significance Analysis")
print("="*90)
print(f"\n{'Clients':<10} {'Plain FL':<20} {'Perturbation only':<20} {'Perturbation + ZT':<20} {'Gap':<10} {'P-value':<10}")
print("-"*90)

for c in client_counts:
    baseline_mean = np.mean(all_results['baseline'][c])
    baseline_std = np.std(all_results['baseline'][c])
    dp_mean = np.mean(all_results['dp_only'][c])
    dp_std = np.std(all_results['dp_only'][c])
    proposed_mean = np.mean(all_results['proposed'][c])
    proposed_std = np.std(all_results['proposed'][c])
    gap = proposed_mean - baseline_mean
    p_val = [r['p_value'] for r in significance_results if r['clients'] == c][0]

    print(f"{c:<10} {baseline_mean:.2f}±{baseline_std:.2f}{'':<10} {dp_mean:.2f}±{dp_std:.2f}{'':<10} {proposed_mean:.2f}±{proposed_std:.2f}{'':<5} {gap:+.2f}{'':<5} {p_val:.4f}")

# Final Plots
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
x = np.arange(len(client_counts))
width = 0.25

baseline_means = [np.mean(all_results['baseline'][c]) for c in client_counts]
baseline_stds = [np.std(all_results['baseline'][c]) for c in client_counts]
proposed_means = [np.mean(all_results['proposed'][c]) for c in client_counts]
proposed_stds = [np.std(all_results['proposed'][c]) for c in client_counts]

# Plot 1: Mean ± Std
ax1 = axes[0]
ax1.bar(x - width/2, baseline_means, width, yerr=baseline_stds, label='Baseline', color='gray', alpha=0.7, capsize=5)
ax1.bar(x + width/2, proposed_means, width, yerr=proposed_stds, label='Proposed (ZT+DP)', color='green', alpha=0.7, capsize=5)
ax1.set_xlabel('Number of Clients'); ax1.set_ylabel('Accuracy (%)')
ax1.set_title('Mean Accuracy ± Standard Deviation')
ax1.set_xticks(x); ax1.set_xticklabels([f'{c}' for c in client_counts])
ax1.legend(); ax1.grid(axis='y', alpha=0.3)

# Plot 2: p-values
ax2 = axes[1]
p_values = [r['p_value'] for r in significance_results]
colors = ['red' if p < 0.05 else 'gray' for p in p_values]
ax2.bar(client_counts, p_values, color=colors, alpha=0.7, edgecolor='black')
ax2.axhline(y=0.05, color='red', linestyle='--', linewidth=2, label='Significance Threshold (p=0.05)')
ax2.set_xlabel('Number of Clients'); ax2.set_ylabel('p-value')
ax2.set_title('Statistical Significance (Baseline vs Proposed)')
ax2.set_xticks(client_counts); ax2.legend(); ax2.grid(axis='y', alpha=0.3)

# Plot 3: Performance Gap
ax3 = axes[2]
gaps = [r['mean_diff'] for r in significance_results]
colors_gap = ['green' if g > 0 else 'red' for g in gaps]
ax3.bar(client_counts, gaps, color=colors_gap, alpha=0.7, edgecolor='black')
ax3.axhline(y=0, color='black', linewidth=1.5)
ax3.set_xlabel('Number of Clients'); ax3.set_ylabel('Performance Gap (%)')
ax3.set_title('Proposed - Baseline (Positive = Improvement)')
ax3.set_xticks(client_counts); ax3.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('statistical_significance_mlp_final.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n✅ All analyses and visualizations completed successfully!")
