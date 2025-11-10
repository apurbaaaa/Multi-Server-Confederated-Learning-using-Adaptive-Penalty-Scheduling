# import flwr as fl
# from flwr.server import ServerConfig
# from flwr.server.strategy import FedAvg
# from flwr.common import Metrics
# from typing import List, Tuple

# import torch
# import torch.nn as nn
# import pandas as pd
# from sklearn.preprocessing import StandardScaler
# from sklearn.metrics import accuracy_score, roc_auc_score



# def load_global_test():
#     df = pd.read_csv("global_test.csv")

#     y = df["isFraud"].astype(int).values
#     X = df.drop(columns=["isFraud"])

#     # Scale numeric features
#     scaler = StandardScaler()
#     X = scaler.fit_transform(X)

#     X = torch.tensor(X, dtype=torch.float32)
#     y = torch.tensor(y, dtype=torch.float32).view(-1, 1)

#     return X, y


# X_test, y_test = load_global_test()



# class FraudNN(nn.Module):
#     def __init__(self, input_dim):
#         super(FraudNN, self).__init__()
#         self.fc1 = nn.Linear(input_dim, 64)
#         self.fc2 = nn.Linear(64, 32)
#         self.fc3 = nn.Linear(32, 1)

#         self._init_weights()

#     def _init_weights(self):
#         for m in self.modules():
#             if isinstance(m, nn.Linear):
#                 nn.init.xavier_uniform_(m.weight)
#                 nn.init.zeros_(m.bias)

#     def forward(self, x):
#         x = torch.relu(self.fc1(x))
#         x = torch.relu(self.fc2(x))
#         return self.fc3(x)



# def evaluate(server_round, parameters, config):
#     # Rebuild model with same architecture
#     model = FraudNN(X_test.shape[1])
#     state_dict = {k: torch.tensor(v) for k, v in zip(model.state_dict().keys(), parameters)}
#     model.load_state_dict(state_dict, strict=True)

#     model.eval()
#     criterion = nn.BCEWithLogitsLoss()

#     with torch.no_grad():
#         logits = model(X_test)
#         loss = criterion(logits, y_test).item()
#         probs = torch.sigmoid(logits)
#         preds = (probs > 0.5).float()

#         acc = accuracy_score(y_test.numpy(), preds.numpy())
#         auc = roc_auc_score(y_test.numpy(), probs.numpy())

#     print(f"[Server Eval] Round {server_round} -> Loss: {loss:.4f}, Acc: {acc:.4f}, AUC: {auc:.4f}")
#     return loss, {"accuracy": acc, "auc": auc}



# def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
#     accs = [num_examples * m["accuracy"] for num_examples, m in metrics if "accuracy" in m]
#     aucs = [num_examples * m["auc"] for num_examples, m in metrics if "auc" in m]
#     examples = [num_examples for num_examples, _ in metrics]

#     results = {}
#     if accs:
#         results["accuracy"] = sum(accs) / sum(examples)
#     if aucs:
#         results["auc"] = sum(aucs) / sum(examples)

#     return results



# strategy = FedAvg(
#     evaluate_fn=evaluate,  # centralized evaluation
#     evaluate_metrics_aggregation_fn=weighted_average,
#     min_available_clients=3,
# )

# config = ServerConfig(num_rounds=20)


# if __name__ == "__main__":
#     fl.server.start_server(
#         server_address="0.0.0.0:5008",
#         config=config,
#         strategy=strategy,
#     )
