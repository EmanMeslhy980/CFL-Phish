# ============================================================
# CFL-Phish: Strictly Paper-Aligned Continual Federated Learning
#
# ============================================================

import os
import copy
import json
import random
import warnings
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

warnings.filterwarnings("ignore")


# ============================================================
# 1. GLOBAL CONFIGURATION (Strictly from Paper Table 6)
# ============================================================

INPUT_DIM = 768
NUM_CLASSES = 2

NUM_CLIENTS = 3
NUM_TASKS = 3

ROUNDS_PER_TASK = 10
LOCAL_EPOCHS = 5

BATCH_SIZE = 64
LEARNING_RATE = 0.001

REPLAY_SIZE = 3000
FISHER_SAMPLES = 200
EWC_LAMBDA = 8000.0

SEEDS = [42, 123, 456, 789, 101112]

OUTPUT_DIR = "./cfl_phish_reproducibility"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# 2. REPRODUCIBILITY
# ============================================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================================================
# 3. DEVICE
# ============================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("=" * 80)
print("CFL-Phish — Strictly Paper-Aligned Implementation")
print("=" * 80)
print(f"Device: {DEVICE}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
print()


# ============================================================
# 4. DATASET PATH & LOADING
# ============================================================

DATA_PATHS = [
    "/content/drive/MyDrive/Phish360_cache/full_768.npz",
    "/content/Phish360_data/full_768.npz",
    "./full_768.npz"
]

def locate_dataset():
    for path in DATA_PATHS:
        if os.path.exists(path):
            return path
    raise FileNotFoundError("\nfull_768.npz was not found.\nExpected one of:\n" + "\n".join(DATA_PATHS))

DATA_PATH = locate_dataset()
print(f"Dataset:\n{DATA_PATH}\n")

def load_npz_dataset(path):
    data = np.load(path, allow_pickle=True)
    print("=" * 80)
    print("NPZ CONTENT")
    print("=" * 80)
    print("Keys:", list(data.keys()))
    
    feature_candidates = ["X", "x", "features", "embeddings", "X_full", "X768", "features_768"]
    X, feature_key = None, None
    for key in feature_candidates:
        if key in data:
            X, feature_key = data[key], key
            break
            
    if X is None:
        for key in data.keys():
            arr = data[key]
            if isinstance(arr, np.ndarray) and arr.ndim == 2 and arr.shape[1] == INPUT_DIM:
                X, feature_key = arr, key
                break
                
    if X is None:
        raise ValueError("No 768-dimensional feature matrix was found.")
    print(f"Feature key: {feature_key}")

    label_candidates = ["y", "Y", "labels", "label", "targets", "target"]
    y, label_key = None, None
    for key in label_candidates:
        if key in data:
            y, label_key = data[key], key
            break
            
    if y is None:
        for key in data.keys():
            arr = data[key]
            if isinstance(arr, np.ndarray) and arr.ndim == 1 and len(arr) == len(X):
                y, label_key = arr, key
                break
                
    if y is None:
        raise ValueError("No label vector was found.")
    print(f"Label key: {label_key}")

    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y).reshape(-1)
    print(f"Raw X shape: {X.shape}\nRaw y shape: {y.shape}\n")
    return X, y

X, y_raw = load_npz_dataset(DATA_PATH)


# ============================================================
# 5. LABEL NORMALIZATION & VALIDATION
# ============================================================

def normalize_labels(y):
    y = np.asarray(y)
    if np.issubdtype(y.dtype, np.number):
        unique = np.unique(y)
        if set(unique.tolist()).issubset({0, 1}):
            return y.astype(np.int64)
        if set(unique.tolist()).issubset({-1, 1}):
            return (y == 1).astype(np.int64)
        raise ValueError(f"Unsupported numeric labels: {unique}")

    y_string = np.array([str(v).strip().lower() for v in y])
    legitimate_labels = {"0", "legitimate", "legit", "benign", "safe", "normal", "clean", "good"}
    phishing_labels = {"1", "phishing", "phish", "malicious", "malware", "unsafe", "attack", "bad"}

    normalized = np.full(len(y_string), -1, dtype=np.int64)
    for i, value in enumerate(y_string):
        if value in legitimate_labels:
            normalized[i] = 0
        elif value in phishing_labels:
            normalized[i] = 1

    unknown = np.where(normalized == -1)[0]
    if len(unknown) > 0:
        raise ValueError("Unknown label values:\n" + str(np.unique(y_string[unknown])[:50]))
    return normalized

y = normalize_labels(y_raw)

if X.ndim != 2:
    raise ValueError(f"Expected 2D X, got {X.shape}")
if X.shape[1] != INPUT_DIM:
    raise ValueError(f"Expected {INPUT_DIM} features, got {X.shape[1]}")
if len(X) != len(y):
    raise ValueError(f"X/y sample count mismatch. X: {len(X)}, y: {len(y)}")

valid_mask = np.isfinite(X).all(axis=1)
invalid_count = int((~valid_mask).sum())
if invalid_count > 0:
    print(f"WARNING: Removing {invalid_count} invalid rows.")
    X, y = X[valid_mask], y[valid_mask]


# ============================================================
# 6. DATASET AUDIT & PROPORTIONAL TABLE 4
# ============================================================

observed_legitimate = int((y == 0).sum())
observed_phishing = int((y == 1).sum())
observed_total = len(y)

PAPER_LEGITIMATE = 6416
PAPER_PHISHING = 4332
PAPER_TOTAL = 10748

print("=" * 80)
print("DATASET LABEL AUDIT")
print("=" * 80)
print(f"Observed total      : {observed_total:,}")
print(f"Observed legitimate : {observed_legitimate:,}")
print(f"Observed phishing   : {observed_phishing:,}")
print(f"Paper composition   : {PAPER_LEGITIMATE:,} legitimate / {PAPER_PHISHING:,} phishing")
print()

# Proportional scaling of Paper's Table 4 to match actual dataset (6331/4417)
# while preserving exact Task totals (4300, 3900, 2548) and approximate ratios (20%, 30%, 90%)
TABLE_4_DISTRIBUTION = {
    "Task 1": {
        "Client 1": {"total": 1800, "legitimate": 1420, "phishing": 380},
        "Client 2": {"total": 1300, "legitimate": 1026, "phishing": 274},
        "Client 3": {"total": 1200, "legitimate": 948, "phishing": 252}
    },
    "Task 2": {
        "Client 1": {"total": 1400, "legitimate": 967, "phishing": 433},
        "Client 2": {"total": 1300, "legitimate": 898, "phishing": 402},
        "Client 3": {"total": 1200, "legitimate": 829, "phishing": 371}
    },
    "Task 3": {
        "Client 1": {"total": 900, "legitimate": 89, "phishing": 811},
        "Client 2": {"total": 850, "legitimate": 84, "phishing": 766},
        "Client 3": {"total": 798, "legitimate": 70, "phishing": 728}
    }
}

def verify_table_4():
    total, legitimate, phishing = 0, 0, 0
    rows = []
    for task_name, clients in TABLE_4_DISTRIBUTION.items():
        task_total, task_legit, task_phish = 0, 0, 0
        for client_name, spec in clients.items():
            assert spec["legitimate"] + spec["phishing"] == spec["total"]
            task_total += spec["total"]
            task_legit += spec["legitimate"]
            task_phish += spec["phishing"]
            rows.append({
                "Task": task_name, "Client": client_name, "Samples": spec["total"],
                "Legitimate": spec["legitimate"], "Phishing": spec["phishing"],
                "Phishing (%)": 100 * spec["phishing"] / spec["total"]
            })
        total += task_total
        legitimate += task_legit
        phishing += task_phish
        
    assert total == 10748
    assert legitimate == observed_legitimate
    assert phishing == observed_phishing
    return pd.DataFrame(rows)

table4_df = verify_table_4()
table4_csv = os.path.join(OUTPUT_DIR, "table4_proportional_distribution.csv")
table4_df.to_csv(table4_csv, index=False)

print("=" * 80)
print("PROPORTIONAL TABLE 4 VERIFIED (Matches actual 6331/4417 dataset)")
print("=" * 80)
print(table4_df.to_string(index=False, formatters={"Phishing (%)": "{:.2f}".format}))
print(f"\nSaved:\n{table4_csv}\n")


# ============================================================
# 7. BUILD TASK DATA
# ============================================================

def build_task_data(X, y, seed=42):
    rng = np.random.default_rng(seed)
    legit_indices = np.where(y == 0)[0]
    phish_indices = np.where(y == 1)[0]
    rng.shuffle(legit_indices)
    rng.shuffle(phish_indices)

    tasks = {}
    leg_ptr, phish_ptr = 0, 0

    for task_name, clients in TABLE_4_DISTRIBUTION.items():
        tasks[task_name] = {}
        for client_name, spec in clients.items():
            n_legit = spec["legitimate"]
            n_phish = spec["phishing"]

            sel_legit = legit_indices[leg_ptr : leg_ptr + n_legit]
            sel_phish = phish_indices[phish_ptr : phish_ptr + n_phish]

            if len(sel_legit) != n_legit or len(sel_phish) != n_phish:
                raise RuntimeError("Data allocation failed. Check dataset size.")

            leg_ptr += n_legit
            phish_ptr += n_phish

            selected_indices = np.concatenate([sel_legit, sel_phish])
            rng.shuffle(selected_indices)

            tasks[task_name][client_name] = {
                "X": X[selected_indices].copy(),
                "y": y[selected_indices].copy(),
                "indices": selected_indices.copy()
            }
            
    # Verify no overlap
    all_indices = [idx for task in tasks.values() for client in task.values() for idx in client["indices"]]
    assert len(all_indices) == 10748 and len(set(all_indices)) == 10748
    return tasks

tasks = build_task_data(X, y, seed=42)
print("=" * 80)
print("TASK ALLOCATION VERIFIED: No sample overlap. 10,748 samples allocated.")
print("=" * 80)


# ============================================================
# 8. MODEL
# ============================================================

class CFLModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(768, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.30),
            nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.25),
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.20),
            nn.Linear(128, 2)
        )
        self.initialize_weights()

    def initialize_weights(self):
        for layer in self.network:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

    def forward(self, x):
        return self.network(x)


# ============================================================
# 9. STANDARD REPLAY BUFFER (FIFO, No Artificial Balancing)
# ============================================================

class ReplayBuffer:
    def __init__(self, max_size=3000):
        self.max_size = max_size
        self.X = np.empty((0, INPUT_DIM), dtype=np.float32)
        self.y = np.empty((0,), dtype=np.int64)

    def __len__(self):
        return len(self.y)

    def add(self, X_new, y_new):
        if len(X_new) == 0:
            return
        X_new, y_new = np.asarray(X_new, dtype=np.float32), np.asarray(y_new, dtype=np.int64)
        self.X = np.concatenate([self.X, X_new], axis=0)
        self.y = np.concatenate([self.y, y_new], axis=0)
        if len(self.X) > self.max_size:
            self.X = self.X[-self.max_size:]
            self.y = self.y[-self.max_size:]

    def sample(self):
        if len(self.y) == 0:
            return None, None
        return self.X.copy(), self.y.copy()


# ============================================================
# 10. ELASTIC WEIGHT CONSOLIDATION (EWC)
# ============================================================

class EWC:
    def __init__(self, fisher_samples=200, lambda_ewc=8000.0):
        self.fisher_samples = fisher_samples
        self.lambda_ewc = lambda_ewc
        self.fisher = {}
        self.optimal_params = {}

    def has_reference(self):
        return len(self.fisher) > 0

    def compute_fisher(self, model, X, y, seed=42):
        if len(X) == 0:
            return
        model.eval()
        rng = np.random.default_rng(seed)
        n = min(self.fisher_samples, len(X))
        selected = rng.choice(len(X), size=n, replace=False)
        
        fisher = {name: torch.zeros_like(parameter, device=DEVICE) for name, parameter in model.named_parameters()}
        
        for index in selected:
            model.zero_grad(set_to_none=True)
            X_i = torch.tensor(X[index], dtype=torch.float32, device=DEVICE).unsqueeze(0)
            y_i = torch.tensor([int(y[index])], dtype=torch.long, device=DEVICE)
            
            logits = model(X_i)
            loss = F.cross_entropy(logits, y_i)
            loss.backward()
            
            for name, parameter in model.named_parameters():
                if parameter.grad is not None:
                    fisher[name] += parameter.grad.detach() ** 2
                    
        for name in fisher:
            fisher[name] /= float(n)
            
        self.fisher = {name: value.detach().clone() for name, value in fisher.items()}
        self.optimal_params = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
        model.zero_grad(set_to_none=True)
        model.train()

    def penalty(self, model):
        if not self.has_reference():
            return torch.tensor(0.0, device=DEVICE)
        penalty = torch.tensor(0.0, device=DEVICE)
        for name, parameter in model.named_parameters():
            if name not in self.fisher:
                continue
            penalty += torch.sum(self.fisher[name] * (parameter - self.optimal_params[name]) ** 2)
        return 0.5 * self.lambda_ewc * penalty


# ============================================================
# 11. FEDERATED CLIENT
# ============================================================

class FederatedClient:
    def __init__(self, client_id, seed=42):
        self.client_id = client_id
        self.seed = seed
        self.model = CFLModel().to(DEVICE)
        self.replay = ReplayBuffer(max_size=REPLAY_SIZE)
        self.ewc = EWC(fisher_samples=FISHER_SAMPLES, lambda_ewc=EWC_LAMBDA)

    def set_parameters(self, state_dict):
        self.model.load_state_dict(copy.deepcopy(state_dict))

    def get_parameters(self):
        return copy.deepcopy(self.model.state_dict())

    def train_local(self, X_current, y_current):
        self.model.train()
        X_replay, y_replay = self.replay.sample()
        
        if X_replay is None:
            X_replay = np.empty((0, INPUT_DIM), dtype=np.float32)
            y_replay = np.empty((0,), dtype=np.int64)
            
        current_dataset = TensorDataset(torch.tensor(X_current, dtype=torch.float32), torch.tensor(y_current, dtype=torch.long))
        current_loader = DataLoader(current_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
        
        if len(X_replay) > 0:
            replay_dataset = TensorDataset(torch.tensor(X_replay, dtype=torch.float32), torch.tensor(y_replay, dtype=torch.long))
            replay_loader = DataLoader(replay_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
        else:
            replay_loader = None
            
        # Standard Adam without weight_decay as per strict paper alignment
        optimizer = optim.Adam(self.model.parameters(), lr=LEARNING_RATE)
        
        for epoch in range(LOCAL_EPOCHS):
            replay_iterator = iter(replay_loader) if replay_loader is not None else None
            
            for X_batch, y_batch in current_loader:
                X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
                optimizer.zero_grad(set_to_none=True)
                
                current_logits = self.model(X_batch)
                loss_cls = F.cross_entropy(current_logits, y_batch)
                
                if replay_iterator is not None:
                    try:
                        X_replay_batch, y_replay_batch = next(replay_iterator)
                    except StopIteration:
                        replay_iterator = iter(replay_loader)
                        X_replay_batch, y_replay_batch = next(replay_iterator)
                        
                    X_replay_batch, y_replay_batch = X_replay_batch.to(DEVICE), y_replay_batch.to(DEVICE)
                    replay_logits = self.model(X_replay_batch)
                    loss_replay = F.cross_entropy(replay_logits, y_replay_batch)
                else:
                    loss_replay = torch.tensor(0.0, device=DEVICE)
                    
                loss_ewc = self.ewc.penalty(self.model)
                
                # Standard loss combination (Weight = 1.0, no arbitrary 0.5 factor)
                loss = loss_cls + loss_replay + loss_ewc
                
                loss.backward()
                optimizer.step()
                
        return {"state_dict": copy.deepcopy(self.model.state_dict()), "num_samples": len(X_current)}

    def consolidate_task(self, X_task, y_task, seed):
        self.ewc.compute_fisher(self.model, X_task, y_task, seed=seed)
        self.replay.add(X_task, y_task)


# ============================================================
# 12. FEDERATED SERVER (Sample-Weighted FedAvg)
# ============================================================

class FederatedServer:
    def __init__(self):
        self.model = CFLModel().to(DEVICE)

    def get_parameters(self):
        return copy.deepcopy(self.model.state_dict())

    def set_parameters(self, state_dict):
        self.model.load_state_dict(copy.deepcopy(state_dict))

    def fedavg(self, client_updates):
        total_samples = sum(update["num_samples"] for update in client_updates)
        global_state = self.get_parameters()

        for key in global_state:
            value = global_state[key]
            if torch.is_floating_point(value):
                aggregated = torch.zeros_like(value)
                for update in client_updates:
                    weight = update["num_samples"] / total_samples
                    aggregated += update["state_dict"][key] * weight
                global_state[key] = aggregated
            else:
                largest_client = max(client_updates, key=lambda update: update["num_samples"])
                global_state[key] = largest_client["state_dict"][key].clone()

        self.set_parameters(global_state)
        return global_state


# ============================================================
# 13. EVALUATION
# ============================================================

@torch.no_grad()
def evaluate_model(model, X, y):
    model.eval()
    dataset = TensorDataset(torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.long))
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    predictions, targets = [], []
    for X_batch, y_batch in loader:
        X_batch = X_batch.to(DEVICE)
        logits = model(X_batch)
        predicted = torch.argmax(logits, dim=1).cpu().numpy()
        predictions.extend(predicted.tolist())
        targets.extend(y_batch.numpy().tolist())

    predictions, targets = np.asarray(predictions), np.asarray(targets)
    return {
        "accuracy": accuracy_score(targets, predictions),
        "precision": precision_score(targets, predictions, zero_division=0),
        "recall": recall_score(targets, predictions, zero_division=0),
        "f1": f1_score(targets, predictions, zero_division=0),
        "confusion_matrix": confusion_matrix(targets, predictions)
    }


# ============================================================
# 14. CFL TRAINING LOOP
# ============================================================

def run_cfl(tasks, seed):
    set_seed(seed)
    server = FederatedServer()
    clients = [FederatedClient(client_id=i + 1, seed=seed) for i in range(NUM_CLIENTS)]

    task_end_metrics = {}

    for task_index, task_name in enumerate(tasks.keys(), start=1):
        print()
        print("=" * 80)
        print(f"SEED {seed} | {task_name}")
        print("=" * 80)

        task_clients = tasks[task_name]
        X_task, y_task = get_task_data(task_clients)
        print(f"Task samples: {len(y_task):,}")

        for round_id in range(1, ROUNDS_PER_TASK + 1):
            print(f"\nRound {round_id}/{ROUNDS_PER_TASK}")
            global_state = server.get_parameters()
            client_updates = []

            for client_id, client in enumerate(clients, start=1):
                client.set_parameters(global_state)
                client_data = task_clients[f"Client {client_id}"]
                update = client.train_local(X_current=client_data["X"], y_current=client_data["y"])
                client_updates.append(update)
                print(f"  Client {client_id}: {update['num_samples']:,}")

            server.fedavg(client_updates)

        # Consolidate task (EWC + Replay) AFTER training is complete
        final_state = server.get_parameters()
        for client_id, client in enumerate(clients, start=1):
            client.set_parameters(final_state)
            client_data = task_clients[f"Client {client_id}"]
            client.consolidate_task(X_task=client_data["X"], y_task=client_data["y"], seed=seed + task_index + client_id)

        # Evaluate retention on ALL seen tasks (Standard CL Protocol)
        task_end_metrics[task_name] = {}
        for previous_task_name in list(tasks.keys())[:task_index]:
            X_prev, y_prev = get_task_data(tasks[previous_task_name])
            metrics = evaluate_model(server.model, X_prev, y_prev)
            task_end_metrics[task_name][previous_task_name] = metrics
            
        print(f"\n  Retention after {task_name}:")
        for prev_name in list(tasks.keys())[:task_index]:
            print(f"    {prev_name} Accuracy: {task_end_metrics[task_name][prev_name]['accuracy'] * 100:.2f}%")

    # Calculate Forgetting
    forgetting_rows = []
    task_names = list(tasks.keys())

    for task_position, task_name in enumerate(task_names):
        own_accuracy = task_end_metrics[task_name][task_name]["accuracy"]
        later_accuracies = []

        for later_task in task_names[task_position + 1:]:
            if task_name in task_end_metrics[later_task]:
                later_accuracies.append(task_end_metrics[later_task][task_name]["accuracy"])

        if len(later_accuracies) > 0:
            final_accuracy = later_accuracies[-1]
            forgetting = max(0.0, own_accuracy - final_accuracy)
        else:
            final_accuracy = own_accuracy
            forgetting = 0.0

        forgetting_rows.append({
            "seed": seed, "task": task_name, "initial_accuracy": own_accuracy,
            "final_accuracy": final_accuracy, "forgetting": forgetting
        })

    return server, clients, pd.DataFrame(forgetting_rows)


def get_task_data(task):
    X_list = [task[client_name]["X"] for client_name in task]
    y_list = [task[client_name]["y"] for client_name in task]
    return np.concatenate(X_list, axis=0), np.concatenate(y_list, axis=0)


# ============================================================
# 15. EXECUTION & SAVING
# ============================================================

print()
print("=" * 80)
print("STARTING MULTI-SEED CFL EXPERIMENT (5 Seeds)")
print("=" * 80)

all_forgetting = []
for seed in SEEDS:
    print()
    print("#" * 80)
    print(f"RUNNING SEED = {seed}")
    print("#" * 80)
    server, clients, forgetting_df = run_cfl(tasks, seed=seed)
    all_forgetting.append(forgetting_df)

forgetting_all_df = pd.concat(all_forgetting, ignore_index=True)
forgetting_csv = os.path.join(OUTPUT_DIR, "cfl_forgetting_all_seeds.csv")
forgetting_all_df.to_csv(forgetting_csv, index=False)

forgetting_summary_df = (
    forgetting_all_df.groupby("task")
    .agg(
        initial_accuracy_mean=("initial_accuracy", "mean"),
        initial_accuracy_std=("initial_accuracy", "std"),
        final_accuracy_mean=("final_accuracy", "mean"),
        final_accuracy_std=("final_accuracy", "std"),
        forgetting_mean=("forgetting", "mean"),
        forgetting_std=("forgetting", "std")
    )
    .reset_index()
)

forgetting_summary_csv = os.path.join(OUTPUT_DIR, "cfl_forgetting_summary.csv")
forgetting_summary_df.to_csv(forgetting_summary_csv, index=False)

# Reproducibility Config
config = {
    "experiment": "CFL-Phish Strictly Paper-Aligned Continual Learning",
    "dataset": os.path.abspath(DATA_PATH),
    "input_dimension": INPUT_DIM,
    "num_classes": NUM_CLASSES,
    "clients": NUM_CLIENTS,
    "tasks": NUM_TASKS,
    "rounds_per_task": ROUNDS_PER_TASK,
    "local_epochs": LOCAL_EPOCHS,
    "batch_size": BATCH_SIZE,
    "learning_rate": LEARNING_RATE,
    "optimizer": "Adam (No weight decay)",
    "replay_size": REPLAY_SIZE,
    "replay_strategy": "Standard FIFO (No artificial balancing)",
    "fisher_samples": FISHER_SAMPLES,
    "ewc_lambda": EWC_LAMBDA,
    "seeds": SEEDS,
    "dataset_composition": {
        "legitimate": observed_legitimate,
        "phishing": observed_phishing,
        "total": observed_total
    },
}    "note": ""


config_path = os.path.join(OUTPUT_DIR, "reproducibility_config.json")
with open(config_path, "w", encoding="utf-8") as f:
    json.dump(config, f, indent=2)

print()
print("=" * 80)
print("FINAL REPRODUCIBILITY REPORT")
print("=" * 80)
print(f"Total Samples: {observed_total:,} (Legitimate: {observed_legitimate:,}, Phishing: {observed_phishing:,})")
print(f"Labels modified: NO")
print(f"Artificial relabeling: NO")
print()
print("Output files saved to:")
print(f"  - {table4_csv}")
print(f"  - {forgetting_csv}")
print(f"  - {forgetting_summary_csv}")
print(f"  - {config_path}")
print("=" * 80)
print("✅ CFL-Phish reproducibility experiment completed successfully.")
print("=" * 80)
