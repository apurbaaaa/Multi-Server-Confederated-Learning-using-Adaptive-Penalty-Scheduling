import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler

# ----------------------------
# Config
# ----------------------------
CSV_PATH = "paysim.csv"
TEST_FRAC = 0.1
N_CLIENTS = 10
MIN_SIZE = 500        # change as needed
SEED = 42
SAVE_PREFIX = "client"  # will produce client1.csv .. client10.csv
GLOBAL_TEST_CSV = "global_test.csv"
GLOBAL_TRAIN_CSV = "global_train.csv"

# ----------------------------
# Load and preprocess (same "ethos" as your original code)
# ----------------------------
df = pd.read_csv(CSV_PATH)

# Drop irrelevant columns if present
drop_cols = ["nameOrig", "nameDest", "isFlaggedFraud"]
df = df.drop(columns=[c for c in drop_cols if c in df.columns])

# Encode categorical column 'type' (if present)
if "type" in df.columns:
    le = LabelEncoder()
    df["type"] = le.fit_transform(df["type"].astype(str))

# Scale numeric features (keep the same feature list you used)
num_cols = ["amount", "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest"]
num_cols_present = [c for c in num_cols if c in df.columns]
if num_cols_present:
    scaler = StandardScaler()
    df[num_cols_present] = scaler.fit_transform(df[num_cols_present])

# Shuffle full dataset (reproducible)
df = df.sample(frac=1, random_state=SEED).reset_index(drop=True)

# ----------------------------
# Create global test set (first TEST_FRAC fraction)
# ----------------------------
test_size = int(len(df) * TEST_FRAC)
df_test = df.iloc[:test_size].reset_index(drop=True)
df_train = df.iloc[test_size:].reset_index(drop=True)

df_test.to_csv(GLOBAL_TEST_CSV, index=False)
df_train.to_csv(GLOBAL_TRAIN_CSV, index=False)

print("Global test shape:", df_test.shape)
print("Global train shape:", df_train.shape)

# ----------------------------
# Partition df_train into random parts with a minimum per client
# ----------------------------
def random_partition_with_min(df_in, n_clients=10, min_size=100, seed=42, save_prefix="client"):
    n = len(df_in)
    if min_size * n_clients > n:
        raise ValueError(
            f"Impossible: min_size * n_clients = {min_size * n_clients} > available rows {n}"
        )

    rng = np.random.default_rng(seed)
    indices = np.arange(n)
    rng.shuffle(indices)

    # allocate min_size indices to each client
    assigned = [list() for _ in range(n_clients)]
    pos = 0
    for i in range(n_clients):
        block = indices[pos: pos + min_size]
        assigned[i].extend(int(x) for x in block)
        pos += min_size

    leftovers = indices[pos:].tolist()
    if leftovers:
        # assign each leftover index uniformly at random to a client
        choices = rng.integers(low=0, high=n_clients, size=len(leftovers))
        for idx, c in zip(leftovers, choices):
            assigned[int(c)].append(int(idx))

    # Build DataFrames and save
    clients = []
    for i, idxs in enumerate(assigned):
        # keep original df_train ordering for each client's rows (optional)
        idxs_sorted = sorted(idxs)
        client_df = df_in.iloc[idxs_sorted].reset_index(drop=True)
        clients.append(client_df)
        client_df.to_csv(f"{save_prefix}{i+1}.csv", index=False)
        print(f"Saved {save_prefix}{i+1}.csv with {len(client_df)} rows")

    # verification: all rows assigned exactly once
    all_assigned = sum((len(lst) for lst in assigned))
    if all_assigned != n:
        raise RuntimeError(f"Assignment error: assigned {all_assigned} rows but df has {n}")

    return clients

# run partitioning on df_train
clients = random_partition_with_min(df_train, n_clients=N_CLIENTS, min_size=MIN_SIZE, seed=SEED, save_prefix=SAVE_PREFIX)
