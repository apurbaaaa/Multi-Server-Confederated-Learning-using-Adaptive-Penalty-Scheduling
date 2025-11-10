# # admm_flower_server_merged.py
# import flwr as fl
# from flwr.server import ServerConfig
# from flwr.common import FitIns, Parameters
# from flwr.server.client_manager import ClientManager
# from flwr.server.strategy import FedAvg
# from typing import List, Tuple

# import numpy as np
# from collections import defaultdict, OrderedDict
# import math
# import logging
# import torch
# import pandas as pd
# from sklearn.preprocessing import StandardScaler
# from sklearn.metrics import accuracy_score, roc_auc_score

# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# # ---------------------------
# # Your evaluation helpers (unchanged)
# # ---------------------------
# def load_global_test():
#     df = pd.read_csv("global_test.csv")
#     y = df["isFraud"].astype(int).values
#     X = df.drop(columns=["isFraud"])
#     scaler = StandardScaler()
#     X = scaler.fit_transform(X)
#     X = torch.tensor(X, dtype=torch.float32)
#     y = torch.tensor(y, dtype=torch.float32).view(-1, 1)
#     return X, y

# X_test, y_test = load_global_test()

# class FraudNN(torch.nn.Module):
#     def __init__(self, input_dim):
#         super(FraudNN, self).__init__()
#         self.fc1 = torch.nn.Linear(input_dim, 64)
#         self.fc2 = torch.nn.Linear(64, 32)
#         self.fc3 = torch.nn.Linear(32, 1)
#         self._init_weights()

#     def _init_weights(self):
#         for m in self.modules():
#             if isinstance(m, torch.nn.Linear):
#                 torch.nn.init.xavier_uniform_(m.weight)
#                 torch.nn.init.zeros_(m.bias)

#     def forward(self, x):
#         x = torch.relu(self.fc1(x))
#         x = torch.relu(self.fc2(x))
#         return self.fc3(x)

# def evaluate(server_round, parameters, config):
#     model = FraudNN(X_test.shape[1])
#     # build state_dict from parameters list-of-arrays
#     keys = list(model.state_dict().keys())
#     state_dict = OrderedDict()
#     for k, arr in zip(keys, parameters):
#         state_dict[k] = torch.tensor(arr)
#     model.load_state_dict(state_dict, strict=True)

#     model.eval()
#     criterion = torch.nn.BCEWithLogitsLoss()
#     with torch.no_grad():
#         logits = model(X_test)
#         loss = criterion(logits, y_test).item()
#         probs = torch.sigmoid(logits)
#         preds = (probs > 0.5).float()
#         acc = accuracy_score(y_test.numpy(), preds.numpy())
#         auc = roc_auc_score(y_test.numpy(), probs.numpy())
#     print(f"[Server Eval] Round {server_round} -> Loss: {loss:.4f}, Acc: {acc:.4f}, AUC: {auc:.4f}")
#     return loss, {"accuracy": acc, "auc": auc}

# def weighted_average(metrics: List[Tuple[int, dict]]) -> dict:
#     accs = [num_examples * m["accuracy"] for num_examples, m in metrics if "accuracy" in m]
#     aucs = [num_examples * m["auc"] for num_examples, m in metrics if "auc" in m]
#     examples = [num_examples for num_examples, _ in metrics]
#     results = {}
#     if accs:
#         results["accuracy"] = sum(accs) / sum(examples)
#     if aucs:
#         results["auc"] = sum(aucs) / sum(examples)
#     return results

# # ---------------------------
# # Parameter utilities (flatten/split)
# # ---------------------------
# def params_to_numpy_list(parameters: Parameters):
#     """Convert Flower Parameters to list of numpy arrays (works with common wrappers)."""
#     try:
#         from flwr.common import parameters_to_ndarrays
#         nd = parameters_to_ndarrays(parameters)
#         return [np.asarray(x.astype(np.float32)) for x in nd]
#     except Exception:
#         # fallback: assume it's already list-like
#         return [np.asarray(x) for x in parameters]

# def params_list_to_flat(params_list):
#     if params_list is None or len(params_list) == 0:
#         return np.array([], dtype=np.float32)
#     return np.concatenate([p.ravel().astype(np.float32) for p in params_list])

# def flat_to_params_list(flat, params_template):
#     if params_template is None or len(params_template) == 0:
#         return []
#     out = []
#     idx = 0
#     for arr in params_template:
#         size = int(np.prod(arr.shape))
#         slice_ = flat[idx: idx + size]
#         out.append(slice_.reshape(arr.shape).astype(arr.dtype))
#         idx += size
#     return out

# def ndarrays_to_parameters(ndarrays):
#     from flwr.common import ndarrays_to_parameters as _wrap
#     return _wrap(ndarrays)

# # ---------------------------
# # ADMM Strategy (merged, minimal changes)
# # ---------------------------
# class ADMMStrategy(FedAvg):
#     def __init__(self, num_groups=4, alpha=0.3, sigma1=1.0, sigma2=1.0, rho=1.0, local_epochs=1, lr=1e-4, batch_size=64, min_available_clients=3, **kwargs):
#         super().__init__(min_available_clients=min_available_clients, **kwargs)
#         self.num_groups = num_groups
#         self.alpha = float(alpha)
#         self.sigma1 = float(sigma1)
#         self.sigma2 = float(sigma2)
#         self.rho = float(rho)
#         self.local_epochs = int(local_epochs)
#         self.lr = float(lr)
#         self.batch_size = int(batch_size)

#         self._initialized = False
#         self.client_to_group = dict()
#         self.group_to_clients = defaultdict(list)
#         self.params_template = None
#         self.x_client = dict()
#         self.lambda_client_flat = dict()
#         self.y_group_flat = dict()
#         self.beta_edges = dict()
#         self.neighbors = defaultdict(list)

#         # default chain neighbors
#         for g in range(num_groups):
#             if g - 1 >= 0:
#                 self.neighbors[g].append(g - 1)
#             if g + 1 < num_groups:
#                 self.neighbors[g].append(g + 1)

#     def _lazy_init(self, client_manager: ClientManager, parameters: Parameters):
#         if self._initialized:
#             return
#         available = client_manager.list_all()
#         client_ids = [c.cid for c in available]
#         client_ids.sort()
#         for idx, cid in enumerate(client_ids):
#             g = idx % self.num_groups
#             self.client_to_group[cid] = g
#             self.group_to_clients[g].append(cid)
#         ndarrays = params_to_numpy_list(parameters)
#         self.params_template = [arr.copy() for arr in ndarrays]
#         flat0 = params_list_to_flat(self.params_template)
#         for g in range(self.num_groups):
#             self.y_group_flat[g] = flat0.copy()
#         for cid in client_ids:
#             self.x_client[cid] = flat0.copy()
#             self.lambda_client_flat[cid] = np.zeros_like(flat0)
#         for p in range(self.num_groups):
#             for q in self.neighbors[p]:
#                 if p < q:
#                     self.beta_edges[(p, q)] = np.zeros_like(flat0)
#         self._initialized = True
#         logger.info("ADMM initialized: %d clients, param vector length %d", len(client_ids), flat0.size)

#     def configure_fit(self, rnd, parameters, client_manager: ClientManager):
#         self._lazy_init(client_manager, parameters)
#         num_available = len(client_manager.list_all())
#         num_to_sample = max(1, int(math.ceil(self.alpha * num_available)))
#         clients = client_manager.sample(num_to_sample, max(1, self.min_available_clients))
#         fit_ins_list = []
#         for client_proxy in clients:
#             cid = client_proxy.cid
#             g = self.client_to_group.get(cid, 0)
#             y_flat = self.y_group_flat[g]
#             lam_flat = self.lambda_client_flat[cid]
#             y_list = flat_to_params_list(y_flat, self.params_template)
#             lam_list = flat_to_params_list(lam_flat, self.params_template)
#             parameters_to_send = ndarrays_to_parameters(y_list)
#             config = {
#                 "sigma1": str(self.sigma1),
#                 "local_epochs": str(self.local_epochs),
#                 "lr": str(self.lr),
#                 "batch_size": str(self.batch_size),
#                 "y": [arr.tolist() for arr in y_list],
#                 "lambda": [arr.tolist() for arr in lam_list],
#             }
#             fit_ins_list.append((client_proxy, FitIns(parameters_to_send, config)))
#         return fit_ins_list

#     def aggregate_fit(self, rnd, results, failures):
#         if not results:
#             logger.warning("No fit results in round %d", rnd)
#             return None, {}
#         for client_proxy, fit_res in results:
#             cid = client_proxy.cid
#             try:
#                 client_params_list = params_to_numpy_list(fit_res.parameters)
#             except Exception:
#                 client_params_list = fit_res.parameters
#             flat = params_list_to_flat(client_params_list)
#             self.x_client[cid] = flat.copy()
#         group_x_sum = {}
#         group_counts = {}
#         for g in range(self.num_groups):
#             cids = self.group_to_clients.get(g, [])
#             if len(cids) == 0:
#                 group_x_sum[g] = np.zeros_like(next(iter(self.y_group_flat.values())))
#                 group_counts[g] = 0
#             else:
#                 xs = [self.x_client[cid] for cid in cids]
#                 group_x_sum[g] = np.sum(xs, axis=0)
#                 group_counts[g] = len(cids)
#         def compute_A_T_beta_for_group(g):
#             total = np.zeros_like(next(iter(self.y_group_flat.values())))
#             for (p, q), beta in self.beta_edges.items():
#                 if g == p:
#                     total = total + beta
#                 elif g == q:
#                     total = total - beta
#             return total
#         for g in range(self.num_groups):
#             num_clients = group_counts.get(g, 0)
#             if num_clients == 0:
#                 continue
#             x_sum = group_x_sum[g]
#             A_T_beta_g = compute_A_T_beta_for_group(g)
#             neighbor_sum = np.zeros_like(self.y_group_flat[g])
#             for h in self.neighbors[g]:
#                 neighbor_sum = neighbor_sum + self.y_group_flat[h]
#             numerator = (self.alpha * self.sigma1 * x_sum) - A_T_beta_g + (self.sigma2 * neighbor_sum) + (self.rho * self.y_group_flat[g])
#             denom = (self.alpha * self.sigma1 * num_clients) + (self.sigma2 * len(self.neighbors[g])) + self.rho
#             if denom == 0:
#                 continue
#             y_new = numerator / float(denom)
#             self.y_group_flat[g] = y_new.astype(np.float32)
#         for (p, q), beta in list(self.beta_edges.items()):
#             y_p = self.y_group_flat[p]
#             y_q = self.y_group_flat[q]
#             self.beta_edges[(p, q)] = beta + self.sigma2 * (y_p - y_q)
#         for cid, x_flat in self.x_client.items():
#             g = self.client_to_group[cid]
#             y_g = self.y_group_flat[g]
#             lam = self.lambda_client_flat[cid]
#             lam_bar = lam + self.sigma1 * (x_flat - y_g)
#             self.lambda_client_flat[cid] = lam + self.alpha * (lam_bar - lam)
#         all_y = np.stack([self.y_group_flat[g] for g in range(self.num_groups)], axis=0)
#         y_avg = np.mean(all_y, axis=0)
#         params_list = flat_to_params_list(y_avg, self.params_template)
#         parameters = ndarrays_to_parameters(params_list)
#         return parameters, {}

# # ---------------------------
# # Run server
# # ---------------------------
# if __name__ == "__main__":
#     strategy = ADMMStrategy(
#         num_groups=4,
#         alpha=0.3,
#         sigma1=1.0,
#         sigma2=1.0,
#         rho=1.0,
#         local_epochs=1,
#         lr=1e-4,
#         batch_size=64,
#         min_available_clients=3,
#         evaluate_fn=evaluate,
#         evaluate_metrics_aggregation_fn=weighted_average,
#     )

#     config = ServerConfig(num_rounds=20)

#     fl.server.start_server(
#         server_address="0.0.0.0:5007",
#         config=config,
#         strategy=strategy,
#     )
