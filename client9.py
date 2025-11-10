# admm_flower_client.py
from collections import OrderedDict
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score
from flwr.client import NumPyClient, ClientApp

# -------------------------
# Data loader (per-client)
# -------------------------
def load_data(client_file="client9.csv", test_split_ratio=0.5):
    df = pd.read_csv(client_file)
    y = df["isFraud"].astype(int).values
    X = df.drop(columns=["isFraud"])

    scaler = StandardScaler()
    X = scaler.fit_transform(X).astype("float32")
    y = y.astype("float32")

    split_idx = int(test_split_ratio * len(X))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

    return X_train, y_train, X_test, y_test

# -------------------------
# Model
# -------------------------
class FraudNN(nn.Module):
    def __init__(self, input_dim):
        super(FraudNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)  # logits

# -------------------------
# Utilities: convert between list-of-arrays and state_dict
# -------------------------
def params_list_to_state_dict(model, params_list):
    """
    params_list: list of numpy arrays in the same order as model.state_dict().keys()
    Returns an OrderedDict suitable for model.load_state_dict
    """
    keys = list(model.state_dict().keys())
    if len(keys) != len(params_list):
        raise ValueError(f"Mismatch: {len(keys)} state keys vs {len(params_list)} arrays")
    state_dict = OrderedDict()
    for k, arr in zip(keys, params_list):
        state_dict[k] = torch.tensor(arr).to(torch.get_default_dtype())
    return state_dict

def state_dict_to_params_list(state_dict):
    # returns list of numpy arrays in state_dict key order
    return [v.cpu().numpy() for _, v in state_dict.items()]

def flatten_params_list(params_list):
    """Concatenate a list of numpy arrays into a 1D numpy array (float32)."""
    if params_list is None:
        return np.array([], dtype=np.float32)
    flat = np.concatenate([p.ravel().astype(np.float32) for p in params_list])
    return flat

def flatten_state_dict_parameters(model):
    """Return a 1D torch tensor of current model parameters (in same ordering)."""
    params = []
    for _, p in model.state_dict().items():
        params.append(p.view(-1))
    if len(params) == 0:
        return torch.tensor([], dtype=torch.float32)
    return torch.cat(params)

# -------------------------
# Training and evaluation
# -------------------------
def train_with_proximal(model, trainloader, prox_target_flat, sigma1, epochs=1, lr=1e-4, device="cpu"):
    model.to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # convert prox_target to torch tensor on device
    prox_target = torch.tensor(prox_target_flat, dtype=torch.float32, device=device)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        total_samples = 0
        for inputs, labels in trainloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            logits = model(inputs)
            bce = criterion(logits, labels)

            # build flattened parameter vector (torch)
            flat_params = flatten_state_dict_parameters(model).to(device)

            # proximal loss: (sigma1/2) * ||theta - prox_target||^2
            prox_loss = (sigma1 / 2.0) * torch.sum((flat_params - prox_target) ** 2)

            loss = bce + prox_loss
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            total_samples += inputs.size(0)

        avg_loss = running_loss / max(total_samples, 1)
        # optional: print per epoch
        print(f"[Local] Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}")

def test_model(model, testloader, device="cpu"):
    model.to(device)
    model.eval()
    criterion = nn.BCEWithLogitsLoss()
    total_loss = 0.0
    total_acc = 0.0
    total_auc = 0.0
    total_samples = 0

    with torch.no_grad():
        for inputs, labels in testloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            logits = model(inputs)
            loss = criterion(logits, labels).item()
            probs = torch.sigmoid(logits).cpu().numpy()
            preds = (probs > 0.5).astype(np.float32)
            y_true = labels.cpu().numpy()

            # safe metrics
            try:
                acc = accuracy_score(y_true, preds)
            except Exception:
                acc = 0.0
            try:
                auc = roc_auc_score(y_true, probs)
            except Exception:
                auc = 0.0

            n = labels.size(0)
            total_loss += loss * n
            total_acc += acc * n
            total_auc += auc * n
            total_samples += n

    if total_samples == 0:
        return 0.0, 0.0, 0.0
    return total_loss / total_samples, total_acc / total_samples, total_auc / total_samples

# -------------------------
# Flower client
# -------------------------
class FlowerADMMClient(NumPyClient):
    def __init__(self, client_csv_path, device="cpu"):
        self.device = device
        X_train, y_train, X_test, y_test = load_data(client_csv_path)
        self.trainloader = DataLoader(TensorDataset(X_train, y_train), batch_size=64, shuffle=True)
        self.testloader = DataLoader(TensorDataset(X_test, y_test), batch_size=64, shuffle=False)
        self.net = FraudNN(X_train.shape[1])
        self.criterion = nn.BCEWithLogitsLoss()

    # returns list of numpy arrays (same format as in original client)
    def get_parameters(self, config):
        return state_dict_to_params_list(self.net.state_dict())

    # set by passing list of numpy arrays in same ordering as state_dict keys
    def set_parameters(self, parameters):
        state_dict = params_list_to_state_dict(self.net, parameters)
        self.net.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        """
        parameters: list of numpy arrays (server-sent model init) - optional
        config: dict that should contain 'y' (list), 'lambda' (list), 'sigma1', 'local_epochs', optional 'lr','batch_size'
        """
        # If server passed a model in 'parameters', set it
        if parameters is not None:
            try:
                self.set_parameters(parameters)
            except Exception as e:
                # if shape mismatch, continue with client's current parameters
                print("[Client] Warning: cannot load server parameters:", e)

        # parse config
        # config may come as dictionary of strings (Flower sometimes serializes); handle robustly
        sigma1 = float(config.get("sigma1", 1.0))
        local_epochs = int(config.get("local_epochs", 1))
        lr = float(config.get("lr", 1e-4))
        # optional override for batch size
        batch_size = int(config.get("batch_size", 64))

        # get y and lambda from config
        y_list = config.get("y", None)
        lambda_list = config.get("lambda", None)

        if y_list is None or lambda_list is None:
            # Nothing to do ADMM-wise; run plain training for compatibility
            print("[Client] Warning: 'y' or 'lambda' not found in config. Running plain local training.")
            train_with_proximal(self.net, self.trainloader, prox_target_flat=np.zeros(0), sigma1=0.0, epochs=local_epochs, lr=lr, device=self.device)
        else:
            # flatten y and lambda lists in the same ordering as model.state_dict
            y_flat = flatten_params_list(y_list)
            lambda_flat = flatten_params_list(lambda_list)

            if y_flat.shape[0] != lambda_flat.shape[0]:
                raise ValueError("Server-sent y and lambda flattened lengths do not match")

            # construct prox_target = y + lambda / sigma1
            prox_target = y_flat.astype(np.float32) + (lambda_flat.astype(np.float32) / float(sigma1))

            # call training with proximal term
            # optionally override trainloader batch_size
            if batch_size != self.trainloader.batch_size:
                self.trainloader = DataLoader(self.trainloader.dataset, batch_size=batch_size, shuffle=True)

            train_with_proximal(self.net, self.trainloader, prox_target_flat=prox_target, sigma1=sigma1, epochs=local_epochs, lr=lr, device=self.device)

        # after training return parameters as list
        return state_dict_to_params_list(self.net.state_dict()), len(self.trainloader.dataset), {}

    def evaluate(self, parameters, config):
        # set model from server parameters if provided
        if parameters is not None:
            try:
                self.set_parameters(parameters)
            except Exception as e:
                print("[Client] Warning: eval cannot set params:", e)

        loss, acc, auc = test_model(self.net, self.testloader, device=self.device)
        print(f"[Client Eval] Loss: {loss:.6f}, Acc: {acc:.4f}, AUC: {auc:.4f}")
        return float(loss), len(self.testloader.dataset), {"accuracy": float(acc), "auc": float(auc)}

# -------------------------
# Helper to start the client
# -------------------------
def client_fn(cid: str, csv_path="client9.csv"):
    return FlowerADMMClient(client_csv_path=csv_path).to_client()

if __name__ == "__main__":
    # Example CLI usage for a single client process:
    # python admm_flower_client.py client9.csv
    import sys
    from flwr.client import start_client

    client_csv = "client9.csv" if len(sys.argv) < 2 else sys.argv[1]
    client = FlowerADMMClient(client_csv_path=client_csv)
    start_client(server_address="127.0.0.1:5006", client=client.to_client())
