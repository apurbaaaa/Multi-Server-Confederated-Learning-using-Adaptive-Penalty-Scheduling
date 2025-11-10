# admm_flower_server_merged.py
import flwr as fl
from flwr.server import ServerConfig
from flwr.common import FitIns, Parameters
from flwr.server.client_manager import ClientManager
from flwr.server.strategy import FedAvg
from typing import List, Tuple

import numpy as np
from collections import defaultdict, OrderedDict
import math
import logging
import torch
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score
import traceback
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------
# Your evaluation helpers (unchanged)
# ---------------------------
def load_global_test():
    df = pd.read_csv("global_test.csv")
    y = df["isFraud"].astype(int).values
    X = df.drop(columns=["isFraud"])
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32).view(-1, 1)
    return X, y

X_test, y_test = load_global_test()

class FraudNN(torch.nn.Module):
    def __init__(self, input_dim):
        super(FraudNN, self).__init__()
        self.fc1 = torch.nn.Linear(input_dim, 64)
        self.fc2 = torch.nn.Linear(64, 32)
        self.fc3 = torch.nn.Linear(32, 1)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                torch.nn.init.zeros_(m.bias)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

def evaluate(server_round, parameters, config):
    model = FraudNN(X_test.shape[1])
    # build state_dict from parameters list-of-arrays
    keys = list(model.state_dict().keys())
    state_dict = OrderedDict()
    for k, arr in zip(keys, parameters):
        state_dict[k] = torch.tensor(arr)
    model.load_state_dict(state_dict, strict=True)

    model.eval()
    criterion = torch.nn.BCEWithLogitsLoss()
    with torch.no_grad():
        logits = model(X_test)
        loss = criterion(logits, y_test).item()
        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).float()
        acc = accuracy_score(y_test.numpy(), preds.numpy())
        auc = roc_auc_score(y_test.numpy(), probs.numpy())
    print(f"[Server Eval] Round {server_round} -> Loss: {loss:.4f}, Acc: {acc:.4f}, AUC: {auc:.4f}")
    return loss, {"accuracy": acc, "auc": auc}

def weighted_average(metrics: List[Tuple[int, dict]]) -> dict:
    accs = [num_examples * m["accuracy"] for num_examples, m in metrics if "accuracy" in m]
    aucs = [num_examples * m["auc"] for num_examples, m in metrics if "auc" in m]
    examples = [num_examples for num_examples, _ in metrics]
    results = {}
    if accs:
        results["accuracy"] = sum(accs) / sum(examples)
    if aucs:
        results["auc"] = sum(aucs) / sum(examples)
    return results

# ---------------------------
# Parameter utilities (flatten/split)
# ---------------------------
def params_to_numpy_list(parameters: Parameters):
    """Convert Flower Parameters to list of numpy arrays (works with common wrappers)."""
    try:
        from flwr.common import parameters_to_ndarrays
        nd = parameters_to_ndarrays(parameters)
        return [np.asarray(x.astype(np.float32)) for x in nd]
    except Exception:
        # fallback: assume it's already list-like
        return [np.asarray(x) for x in parameters]

def params_list_to_flat(params_list):
    if params_list is None or len(params_list) == 0:
        return np.array([], dtype=np.float32)
    return np.concatenate([p.ravel().astype(np.float32) for p in params_list])

def flat_to_params_list(flat, params_template):
    if params_template is None or len(params_template) == 0:
        return []
    out = []
    idx = 0
    for arr in params_template:
        size = int(np.prod(arr.shape))
        slice_ = flat[idx: idx + size]
        out.append(slice_.reshape(arr.shape).astype(arr.dtype))
        idx += size
    return out

def ndarrays_to_parameters(ndarrays):
    from flwr.common import ndarrays_to_parameters as _wrap
    return _wrap(ndarrays)

# ---------------------------
# ADMM Strategy (merged, minimal changes)
# ---------------------------
class ADMMStrategy(FedAvg):
    def __init__(self, num_groups=4, alpha=0.3, sigma1=1.0, sigma2=1.0, rho=1.0, local_epochs=1, lr=1e-4, batch_size=64, min_available_clients=3, **kwargs):
        super().__init__(min_available_clients=min_available_clients, **kwargs)
        self.num_groups = num_groups
        self.alpha = float(alpha)
        self.sigma1 = float(sigma1)
        self.sigma2 = float(sigma2)
        self.rho = float(rho)
        self.local_epochs = int(local_epochs)
        self.lr = float(lr)
        self.batch_size = int(batch_size)

        self._initialized = False
        self.client_to_group = dict()
        self.group_to_clients = defaultdict(list)
        self.params_template = None
        self.x_client = dict()
        self.lambda_client_flat = dict()
        self.y_group_flat = dict()
        self.beta_edges = dict()
        self.neighbors = defaultdict(list)

        # default chain neighbors
        for g in range(num_groups):
            if g - 1 >= 0:
                self.neighbors[g].append(g - 1)
            if g + 1 < num_groups:
                self.neighbors[g].append(g + 1)

    def _lazy_init(self, client_manager: ClientManager, parameters: Parameters):
        """
        Backwards/forwards compatible initialization of ADMM state.
        - Enumerates connected clients in a way that works across Flower versions.
        - Builds round-robin client->group mapping.
        - Extracts param template from the provided `parameters`.
        - Initializes y_group_flat, x_client, lambda_client_flat, beta_edges.
        """
        if self._initialized:
            return

        # --- Enumerate clients robustly ---
        available = None
        if hasattr(client_manager, "list_all"):
            available = client_manager.list_all()
        elif hasattr(client_manager, "all"):
            available = client_manager.all()
        elif hasattr(client_manager, "all_clients"):
            available = client_manager.all_clients()
        elif hasattr(client_manager, "clients"):
            available = client_manager.clients()
        else:
            # Last resort: try common attributes, then error
            attrs = dir(client_manager)
            raise RuntimeError(
                "Cannot enumerate clients from client_manager. "
                f"Detected attributes: {attrs}. Please adapt _lazy_init() for this Flower version."
            )

        # Convert to list of proxies/representations
        available_list = list(available) if available is not None else []
        client_ids = []
        for c in available_list:
            # Prefer .cid if present
            if hasattr(c, "cid"):
                client_ids.append(c.cid)
            elif isinstance(c, (tuple, list)) and len(c) >= 1:
                client_ids.append(str(c[0]))
            else:
                client_ids.append(str(c))

        client_ids.sort()

        if len(client_ids) < self.min_available_clients:
            logger.warning(
                "Detected %d clients but min_available_clients=%d",
                len(client_ids),
                self.min_available_clients,
            )

        # --- Build mapping client -> group (round-robin) ---
        for idx, cid in enumerate(client_ids):
            g = idx % self.num_groups
            self.client_to_group[cid] = g
            self.group_to_clients[g].append(cid)

        logger.info(
            "Assigned %d clients into %d groups (example mapping: %s)",
            len(client_ids),
            self.num_groups,
            dict(list(self.client_to_group.items())[:10]),
        )

        # --- Extract parameter template from initial parameters ---
        try:
            ndarrays = params_to_numpy_list(parameters)
        except Exception as e:
            raise RuntimeError(f"Unable to extract parameter template from server parameters: {e}")

        # Save template shapes and dtype
        self.params_template = [arr.copy() for arr in ndarrays]

        # Flatten initial vector
        flat0 = params_list_to_flat(self.params_template)

        # --- Initialize group y, client x and lambdas, beta on edges ---
        for g in range(self.num_groups):
            self.y_group_flat[g] = flat0.copy().astype(np.float32)

        for cid in client_ids:
            self.x_client[cid] = flat0.copy().astype(np.float32)
            self.lambda_client_flat[cid] = np.zeros_like(flat0, dtype=np.float32)

        # initialize beta for each undirected edge (store only p<q)
        for p in range(self.num_groups):
            for q in self.neighbors[p]:
                if p < q:
                    self.beta_edges[(p, q)] = np.zeros_like(flat0, dtype=np.float32)

        self._initialized = True
        logger.info("ADMM state initialized: param vector length %d", flat0.size)


    # Replace your existing configure_fit signature with this function
    def configure_fit(self, *args, **kwargs):
        """
        Backwards/forwards compatible wrapper for Flower's configure_fit call.
        Supports both:
        - configure_fit(rnd, parameters, client_manager)
        - configure_fit(server_round=..., parameters=..., client_manager=...)
        """
        # Unpack possible calling styles
        if "server_round" in kwargs or ("parameters" in kwargs and "client_manager" in kwargs):
            rnd = kwargs.get("server_round", kwargs.get("rnd", None))
            parameters = kwargs.get("parameters")
            client_manager = kwargs.get("client_manager")
        else:
            # positional style: (rnd, parameters, client_manager)
            if len(args) >= 3:
                rnd, parameters, client_manager = args[0], args[1], args[2]
            else:
                raise TypeError("configure_fit called with unexpected arguments")

        # Lazy init on first round (will enumerate clients inside _lazy_init if needed)
        # but we also need a robust way to count available clients here for sampling.
        # Try to obtain an iterable/list of client proxies in a backwards-compatible way.
        if hasattr(client_manager, "list_all"):
            available_iter = client_manager.list_all()
        elif hasattr(client_manager, "all"):
            available_iter = client_manager.all()
        elif hasattr(client_manager, "all_clients"):
            available_iter = client_manager.all_clients()
        elif hasattr(client_manager, "clients"):
            available_iter = client_manager.clients()
        else:
            # fallback: try sample(1,1) to at least get one proxy, then build available_iter
            try:
                fallback_clients = client_manager.sample(1, 1)
                available_iter = fallback_clients
            except Exception:
                raise RuntimeError(
                    "Unable to enumerate clients from client_manager. "
                    "Update configure_fit() to match your Flower version."
                )

        available_list = list(available_iter) if available_iter is not None else []
        num_available = len(available_list)

        # ensure ADMM internal state is initialized (this extracts parameter template etc.)
        # _lazy_init expects a client_manager and parameters; it will probe the manager as needed
        try:
            self._lazy_init(client_manager, parameters)
        except Exception as e:
            # If lazy init failed but we still have initial params template, continue.
            logger.debug("Warning: _lazy_init raised: %s", e)

        # determine how many clients to sample
        num_to_sample = max(1, int(math.ceil(self.alpha * max(num_available, 1))))

        # Prefer using client_manager.sample when available
        if hasattr(client_manager, "sample"):
            # Some Flower versions expect (num, min_num)
            try:
                clients = client_manager.sample(num_to_sample, max(1, self.min_available_clients))
            except TypeError:
                # Alternative signature: sample(num)
                clients = client_manager.sample(num_to_sample)
        else:
            # fallback: take first num_to_sample from available_list
            clients = available_list[:num_to_sample]

        fit_ins_list = []
        for client_proxy in clients:
            # client_proxy might be a ClientProxy or simple tuple; try to extract cid robustly
            try:
                cid = client_proxy.cid
            except Exception:
                # If it's a tuple like (cid, proxy)
                if isinstance(client_proxy, (tuple, list)) and len(client_proxy) >= 1:
                    cid = str(client_proxy[0])
                else:
                    cid = str(client_proxy)

            # derive group
            g = self.client_to_group.get(cid, 0)

            # prepare y and lambda as lists shaped like params_template
            y_flat = self.y_group_flat.get(g)
            if y_flat is None:
                # fallback to global template zero
                y_flat = params_list_to_flat(self.params_template) if self.params_template is not None else np.array([], dtype=np.float32)
            lam_flat = self.lambda_client_flat.get(cid, np.zeros_like(y_flat))

            y_list = flat_to_params_list(y_flat, self.params_template)
            lam_list = flat_to_params_list(lam_flat, self.params_template)

            # parameters we send as the "current model": use y_list (server's group model)
            parameters_to_send = ndarrays_to_parameters(y_list)

            # config to instruct client on ADMM local solve
            # NOTE: Flower's config may not accept nested list types reliably for some transports.
            # We avoid putting large lists directly into the config. `y` is already sent
            # as the FitIns.parameters (ndarrays). We serialize `lambda` as a JSON string
            # so it is safe to transmit in the config and parse on the client.
            try:
                lambda_json = json.dumps([arr.tolist() for arr in lam_list])
            except Exception:
                # Fallback: send empty string if serialization fails
                lambda_json = ""

            config = {
                "sigma1": str(self.sigma1),
                "local_epochs": str(self.local_epochs),
                "lr": str(self.lr),
                "batch_size": str(self.batch_size),
                # send lambda as JSON string; clients will json.loads it
                "lambda": lambda_json,
            }

            fit_ins = FitIns(parameters_to_send, config)
            fit_ins_list.append((client_proxy, fit_ins))

        return fit_ins_list



    def aggregate_fit(self, rnd, results, failures):
        # Diagnostic logging: number of successful results and failures
        try:
            num_results = len(results) if results is not None else 0
        except Exception:
            num_results = 0
        try:
            num_failures = len(failures) if failures is not None else 0
        except Exception:
            num_failures = 0

        logger.info("aggregate_fit called for round %d: %d results, %d failures", rnd, num_results, num_failures)

        # Log failure details (if any)
        if failures:
            try:
                for f in failures:
                    # Flower may provide either (client_proxy, exception) tuples OR raw exceptions
                    if isinstance(f, (list, tuple)) and len(f) == 2:
                        cp, exc = f
                        cid_f = getattr(cp, "cid", str(cp))
                        logger.error("Fit failure from client %s: %s", cid_f, repr(exc))
                        # If exc is an exception instance, include its traceback if possible
                        try:
                            logger.error("Traceback for failure from client %s:\n%s", cid_f, ''.join(traceback.format_exception(type(exc), exc, exc.__traceback__)))
                        except Exception:
                            logger.debug("Could not format traceback for failure from client %s", cid_f)
                    else:
                        # raw exception object or other payload
                        logger.error("Fit failure entry (raw): %s", repr(f))
                        try:
                            if isinstance(f, BaseException):
                                logger.error("Traceback: %s", ''.join(traceback.format_exception(type(f), f, f.__traceback__)))
                        except Exception:
                            logger.debug("Could not format traceback for raw failure entry")
            except Exception:
                logger.exception("Error while logging failures in aggregate_fit")

        if not results:
            logger.warning("No fit results in round %d", rnd)
            return None, {}

        # Log expected template size to help debug shape mismatches
        try:
            template_flat = params_list_to_flat(self.params_template) if self.params_template is not None else np.array([], dtype=np.float32)
            logger.info("Expected parameter vector length (template): %d", template_flat.size)
        except Exception:
            logger.exception("Unable to compute params_template flat length")

        for client_proxy, fit_res in results:
            # Robust extraction of client id
            try:
                cid = client_proxy.cid
            except Exception:
                if isinstance(client_proxy, (tuple, list)) and len(client_proxy) >= 1:
                    cid = str(client_proxy[0])
                else:
                    cid = str(client_proxy)

            # If a new client connected after _lazy_init, assign it to a group and initialize its ADMM state
            if cid not in self.client_to_group:
                # simple round-robin assignment based on current count
                new_idx = len(self.client_to_group)
                g = new_idx % self.num_groups
                self.client_to_group[cid] = g
                self.group_to_clients[g].append(cid)
                # initialize x and lambda for the new client using template
                try:
                    template_flat = params_list_to_flat(self.params_template) if self.params_template is not None else np.array([], dtype=np.float32)
                except Exception:
                    template_flat = np.array([], dtype=np.float32)
                self.x_client[cid] = template_flat.copy()
                self.lambda_client_flat[cid] = np.zeros_like(template_flat, dtype=np.float32)
                logger.info("Assigned new client %s to group %d (initialized x/lambda)", cid, g)

            # Try to convert returned parameters to numpy list and log shapes
            try:
                client_params_list = params_to_numpy_list(fit_res.parameters)
                shapes = [p.shape for p in client_params_list]
                flat_len = params_list_to_flat(client_params_list).size
                logger.info("Received fit result from client %s: %d param arrays, shapes=%s, flat_len=%d", cid, len(client_params_list), shapes, flat_len)
            except Exception as ex:
                logger.exception("Failed to parse parameters from client %s: %s", cid, ex)
                # fallback to raw parameters
                try:
                    client_params_list = fit_res.parameters
                    logger.info("Fallback: stored raw parameters type for client %s: %s", cid, type(client_params_list))
                except Exception:
                    logger.exception("Unable to access raw fit_res.parameters for client %s", cid)
                    continue

            try:
                flat = params_list_to_flat(client_params_list)
                self.x_client[cid] = flat.copy()
            except Exception:
                logger.exception("Failed to flatten/store parameters for client %s", cid)
        group_x_sum = {}
        group_counts = {}
        for g in range(self.num_groups):
            cids = self.group_to_clients.get(g, [])
            if len(cids) == 0:
                group_x_sum[g] = np.zeros_like(next(iter(self.y_group_flat.values())))
                group_counts[g] = 0
            else:
                xs = [self.x_client[cid] for cid in cids]
                group_x_sum[g] = np.sum(xs, axis=0)
                group_counts[g] = len(cids)
        def compute_A_T_beta_for_group(g):
            total = np.zeros_like(next(iter(self.y_group_flat.values())))
            for (p, q), beta in self.beta_edges.items():
                if g == p:
                    total = total + beta
                elif g == q:
                    total = total - beta
            return total
        for g in range(self.num_groups):
            num_clients = group_counts.get(g, 0)
            if num_clients == 0:
                continue
            x_sum = group_x_sum[g]
            A_T_beta_g = compute_A_T_beta_for_group(g)
            neighbor_sum = np.zeros_like(self.y_group_flat[g])
            for h in self.neighbors[g]:
                neighbor_sum = neighbor_sum + self.y_group_flat[h]
            numerator = (self.alpha * self.sigma1 * x_sum) - A_T_beta_g + (self.sigma2 * neighbor_sum) + (self.rho * self.y_group_flat[g])
            denom = (self.alpha * self.sigma1 * num_clients) + (self.sigma2 * len(self.neighbors[g])) + self.rho
            if denom == 0:
                continue
            y_new = numerator / float(denom)
            self.y_group_flat[g] = y_new.astype(np.float32)
        for (p, q), beta in list(self.beta_edges.items()):
            y_p = self.y_group_flat[p]
            y_q = self.y_group_flat[q]
            self.beta_edges[(p, q)] = beta + self.sigma2 * (y_p - y_q)
        for cid, x_flat in self.x_client.items():
            g = self.client_to_group[cid]
            y_g = self.y_group_flat[g]
            lam = self.lambda_client_flat[cid]
            lam_bar = lam + self.sigma1 * (x_flat - y_g)
            self.lambda_client_flat[cid] = lam + self.alpha * (lam_bar - lam)
        all_y = np.stack([self.y_group_flat[g] for g in range(self.num_groups)], axis=0)
        y_avg = np.mean(all_y, axis=0)
        params_list = flat_to_params_list(y_avg, self.params_template)
        parameters = ndarrays_to_parameters(params_list)
        return parameters, {}

# ---------------------------
# Run server
# ---------------------------
if __name__ == "__main__":
    strategy = ADMMStrategy(
        num_groups=4,
        alpha=0.3,
        sigma1=1.0,
        sigma2=1.0,
        rho=1.0,
        local_epochs=1,
        lr=1e-4,
        batch_size=64,
        min_available_clients=3,
        evaluate_fn=evaluate,
        evaluate_metrics_aggregation_fn=weighted_average,
    )

    config = ServerConfig(num_rounds=20)

    fl.server.start_server(
        server_address="0.0.0.0:5006",
        config=config,
        strategy=strategy,
    )
