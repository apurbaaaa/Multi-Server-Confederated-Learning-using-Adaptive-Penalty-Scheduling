# Multi-Server Federated Learning with Adaptive Penalty Scheduling

This project demonstrates a multi-server-style, adaptive-penalty variant of federated learning built on top of [Flower](https://flower.dev/). It implements an ADMM-inspired strategy that groups clients, adapts dual penalties during training, and synchronises models across multiple client clusters while tracking federation-wide performance on a held-out global test set.

## Repository Map

- `server1.py` – Flower server running the adaptive ADMM aggregation logic, residual tracking, and global evaluation pipeline.
- `client1.py` … `client10.py` – Flower `NumPyClient` implementations (one per dataset shard). Each script loads its own `clientX.csv` by default but accepts any CSV path via CLI.
- `global_test.py` – Data preparation routine that ingests `paysim.csv`, produces per-client CSVs, and builds `global_test.csv` / `global_train.csv` splits.
- `data_preprocessing.ipynb` – Scratch notebook for exploratory feature engineering (not required for running the federation).

## Prerequisites

- Python 3.9 or newer.
- A working PyTorch installation compatible with your hardware (CPU-only is fine).
- Dependencies (install in your virtual environment):

  ```bash
  pip install flwr torch pandas scikit-learn numpy
  ```

  Optional: `jupyter` for the preprocessing notebook.

## Data Preparation Workflow

1. Download the PaySim dataset (or any dataset with compatible columns) as `paysim.csv` into the repository root.
2. Run the splitter to generate per-client data and evaluation folds:

   ```bash
  python global_test.py
  ```

   This will:

   - Shuffle and scale core numeric columns.
   - Reserve 10 % for `global_test.csv` (used by the server for central evaluation).
   - Store the remaining 90 % as `global_train.csv`.
   - Partition `global_train.csv` into 10 balanced client CSVs (`client1.csv` … `client10.csv`) with a minimum row threshold per client.

3. Inspect the generated CSVs if you need to adjust class balance or feature engineering.

## Running the Federated System

1. **Start the server** (defaults to port `5006` and 20 rounds):

   ```bash
  python server1.py
  ```

   - Uses the adaptive ADMM strategy (`ADMMStrategy`).
   - Requires at least 10 available clients (`min_available_clients=10`).
   - Evaluates each global round on `global_test.csv`.
   - Saves per-round residual norms to `residual_history.csv` when training finishes.

2. **Launch clients** in separate terminals (or processes). Provide the desired CSV path when starting:

   ```bash
  python client1.py client1.csv
  python client2.py client2.csv
  ...
  python client10.py client10.csv
  ```

   - Each client trains a `FraudNN` (3-layer feedforward network) using proximal updates derived from the server-provided ADMM state (`y`, `lambda`, `sigma1`).
   - Training uses mini-batching, BCE-with-logits loss, and an adaptive proximal penalty.

3. **Monitor training** through Flower’s logs. The server prints global metrics (`loss`, `accuracy`, `AUC`) each round. Clients emit local batch losses and evaluation summaries.

## Key Implementation Details

- **Adaptive Penalty Scheduling:** The server tracks primal/dual residual norms each aggregation round and scales `sigma1`/`sigma2` using heuristics from Boyd et al. (ADMM). Dual variables (`lambda` per client, `beta` per group edge) are rescaled to keep them consistent with penalty changes.
- **Group-Based Aggregation:** Clients are assigned to groups in a round-robin fashion. Group models are averaged with their neighbours before being merged into a global model snapshot.
- **Robust Flower Integration:** `configure_fit` and `aggregate_fit` handle multiple Flower API versions, ensuring compatibility with different releases. Errors trigger detailed logging rather than crashing the federation.
- **Resilience:** Client training/evaluation wrappers catch and report exceptions, returning safe fallbacks so the server can continue aggregating even with partial failures.

## Customisation Tips

- Adjust hyperparameters (rounds, learning rate, batch size, `alpha`, `sigma1`, `sigma2`, group count, etc.) via the `ADMMStrategy` constructor in `server1.py`.
- Modify the model architecture inside `FraudNN` in `client*.py` and `server1.py` if you need different capacity.
- Change `global_test.py` to adapt the preprocessing pipeline, column selection, or partitioning strategy for new datasets.

## Troubleshooting & Logging

- Server logs print residual norms and penalty scaling actions. Inspect `residual_history.csv` after training to analyse convergence behaviour.
- If a client fails to parse the `lambda` payload or encounters malformed data, it logs a warning and falls back to plain local training for that round.
- Flower requires that clients reach the server; verify network settings or firewall rules when distributing across machines.

## Notebook Usage

To explore preprocessing interactively:

```bash
jupyter notebook data_preprocessing.ipynb
```

This notebook isn’t part of the automated pipeline but can help with feature inspection or alternative scaling strategies.

## Next Steps

- Integrate automatic reporting or dashboards for global metrics.
- Extend `ADMMStrategy` to support asynchronous federated averaging or heterogeneous client capability weighting.
- Package dependencies into a `requirements.txt` or `pyproject.toml` for reproducible environments.


