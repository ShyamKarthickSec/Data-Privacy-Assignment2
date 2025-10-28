# Exploration and Reflection on Privacy Anomalies inSmart Digital Ecosystems - Data Privacy - CSEC5614
**Notebook:** `Activity_Code_02_Group_3.ipynb`  
**Scope:** This notebook demonstrates practical, *runnable* implementations for three privacy/security anomalies using Differential Privacy (DP), Homomorphic Encryption (HE), and Federated Learning (FL) with DP-SGD.

---

## Table of Contents
- [Overview](#overview)
- [Environment Setup](#environment-setup)
- [How to Run](#how-to-run)
- [Dataset Notes](#dataset-notes)
- [Anomaly 1 — Public Transport & Contactless Payment Data (DP counts)](#anomaly-1--public-transport--contactless-payment-data-dp-counts)
- [Anomaly 2 — Geo-Fencing Location Privacy with Paillier HE](#anomaly-2--geo-fencing-location-privacy-with-paillier-he)
- [Anomaly 3 — Federated Learning on Wearable Health Data (DP-SGD)](#anomaly-3--federated-learning-on-wearable-health-data-dp-sgd)
- [Parameters & Tuning](#parameters--tuning)
- [Troubleshooting](#troubleshooting)
- [References](#references)

---

## Overview
The notebook is organized into three sections (as indicated by its headings):
- # **Implementation of All Three Anomalies Proposed Solutions Executed in This Notebook.**
- # **Anomaly 1: Public Transport & Contactless Payment Data**
- # **Anomaly 2: Location Data Exploitation in GeoFencing Apps**
- # **Anomaly 3: ealth & Wearable Data Leakage in Federated Learning**

The core ideas explored:
1. **Differential Privacy for aggregate counts** (Laplace mechanism) with an *entropy-aware epsilon* policy to reduce noise when user anonymity is lower.
2. **Privacy-preserving geo-fence membership check** using **Paillier homomorphic encryption**: the server evaluates distance from a fence center on encrypted coordinates; the client decrypts and checks within-threshold locally.
3. **Federated Learning with DP-SGD-style training** over synthetic wearable signals: client-side gradient clipping + Gaussian noise, server-side **FedAvg** to update a global model.

---

## Environment Setup

Tested with Python 3.10+ (CPU). If using GPU, install a matching TensorFlow build.

```bash
# Recommended: create a clean virtual environment
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

# Core libs used by the notebook
pip install numpy pandas diffprivlib phe tensorflow
```

> **Notes**
> - `diffprivlib` provides the Laplace mechanism used in Anomaly 1.
> - `phe` provides Paillier HE used in Anomaly 2.
> - `tensorflow` is used to build and train models in Anomaly 3.
> - If `tensorflow` installation fails on Windows, try `pip install tensorflow-cpu`.

---

## How to Run

Option A — JupyterLab:
1. Start Jupyter: `jupyter lab` (or `jupyter notebook`).
2. Open `Activity_Code_02_Group_3.ipynb`.
3. **Run cells top-to-bottom**. Each anomaly section is self-contained; install cells (e.g., `!pip install ...`) appear before usage.

Option B — Programmatic:
- Convert to a Python script and run:
  ```bash
  jupyter nbconvert --to script Activity_Code_02_Group_3.ipynb
  python Activity_Code_02_Group_3.py
  ```

---

## Dataset Notes

- **Transit synthetic file**: the Anomaly 1 section expects a CSV like `transit_synthetic_30.csv` containing columns at minimum:
  - `UserID`, `Station`, `Time` (e.g., `"HH:MM"` or full timestamp)
- If you **don’t have this file**, either:
  - Supply your own small synthetic dataset with those columns, or
  - Modify the `csv_path` variable to point to an existing CSV.

No external datasets are required for Anomaly 2 and Anomaly 3 — they generate synthetic examples internally.

---

## Anomaly 1 — Public Transport & Contactless Payment Data (DP counts)

**Goal:** Release per-station, per-hour user counts with **differential privacy** to mitigate re-identification risks in sparse/low-entropy buckets.

**Key Ideas**
- One contribution per user per (Station, Hour): deduplicate before aggregation.
- Compute **Shannon entropy** over unique users in each bucket.
- Use **entropy-aware epsilon**:
  - If entropy is below a threshold `H_MIN`, scale epsilon down via `LOW_ENTROPY_FACTOR` (more noise, higher privacy).
  - Else, use `BASE_EPSILON` directly.
- Add **Laplace noise** with `sensitivity=1.0` to each count.

**Notable Functions**
- `shannon_entropy_from_userlist(user_list)` — computes entropy of a user bucket.
- `laplace_mechanism(value, epsilon, sensitivity)` — wraps `diffprivlib.mechanisms.Laplace`.

**Typical Output Columns**
- `Station`, `Hour`, `true_count`, `entropy`, `epsilon_used`, `noisy_count`.

**Where to edit**
- `BASE_EPSILON`, `LOW_ENTROPY_FACTOR`, `H_MIN`, `SENSITIVITY`, `ROUND_RESULTS` near the start of the Anomaly 1 code block.
- `csv_path` for your dataset location.

---

## Anomaly 2 — Geo-Fencing Location Privacy with Paillier HE

**Goal:** Server determines whether a client is **inside a circular geo-fence** without learning the client’s raw coordinates.

**Protocol (simplified)**
1. **Key generation:** Server (or secure device) creates Paillier public/private keypair.
2. **Client encodes & encrypts:** Client scales (`scale`) and encrypts \(x, y, x^2, y^2\) using the **public key**.
3. **Server computes on ciphertexts:** Using HE properties,
   - computes encrypted \( (x - a)^2 + (y - b)^2 \) given server’s fence center \((a, b)\) and pre-chosen radius \(R\) (all consistently scaled).
4. **Client decrypts & checks:** Client decrypts the result and locally checks if distance² \(\le R^2\). The server never sees the plaintext location.

**Notable Functions**
- `encode_coord`, `encode_square`, `decode_square_int` — fixed-point encoding helpers.
- `generate_keys(n_length=2048)` — Paillier keypair.
- `client_encrypt_location(pubkey, x, y, scale)` — client-side encryption of location.
- `server_compute_encrypted_distance_squared(pubkey, enc_bundle, a, b, R, scale)` — server-side HE computation.
- `client_decrypt_and_check(privkey, enc_dist2, R, scale)` — client-side decision.
- `demo()` — runnable end-to-end example.

**Tuning & Precision**
- Choose `scale = 10**k` based on desired decimal precision (e.g., `10**6` for ~micro-degree).
- Larger `n_length` (e.g., 2048) is more secure but slower.

---

## Anomaly 3 — Federated Learning on Wearable Health Data (DP-SGD)

**Goal:** Train a global model on **synthetic wearable sequences** while preserving client privacy via **per-step clipping** and **Gaussian noise** added to gradients.

**Data & Model**
- Synthetic time-series with shape `(SEQ_LEN, NUM_FEATURES)` per sample.
- CNN-based classifier: `Conv1D → MaxPool1D → Conv1D → GAP → Dense → Dense(softmax)`.

**Federated Training Loop**
- Each round selects `CLIENTS_PER_ROUND` out of `NUM_CLIENTS`.
- For each selected client:
  1. Initialize model with current global weights.
  2. Train locally using `dp_train_step` (clip gradients to `CLIPPING_NORM`, add noise with `NOISE_MULTIPLIER * CLIPPING_NORM`).
  3. Return updated weights.
- **FedAvg** aggregates client weights to update the global model.

**Notable Functions**
- `generate_synthetic_client_data(num_clients, samples_per_client)` — builds per-client datasets.
- `create_model()` — constructs the CNN.
- `dp_train_step(model, optimizer, x_batch, y_batch, clipping_norm, noise_stddev)` — DP gradient step.
- `average_weights(weights_list)` / `set_model_weights(model, weights)` — FedAvg utilities.

**Hyperparameters (editable)**
- `NUM_CLIENTS`, `CLIENTS_PER_ROUND`, `ROUNDS`, `LOCAL_EPOCHS`, `BATCH_SIZE`
- `CLIPPING_NORM`, `NOISE_MULTIPLIER`, `LEARNING_RATE`
- `SEQ_LEN`, `NUM_FEATURES`, `NUM_CLASSES`

---

## Parameters & Tuning

### Differential Privacy (Anomaly 1)
- **Epsilon** (`BASE_EPSILON`): ↑epsilon → ↓noise (weaker privacy). Typical small values in (0.1–2.0).
- **Entropy threshold** (`H_MIN`): buckets with entropy below this get **stricter** noise (via `LOW_ENTROPY_FACTOR`).
- **Sensitivity**: with 1 contribution/user, `sensitivity=1.0`.

### Homomorphic Encryption (Anomaly 2)
- **Key size** (`n_length`): 1024 for demo, 2048+ for stronger security.
- **Scale**: determines fixed-point precision; ensure **all parties** use the same `scale`.
- **Radius/Center**: keep consistent units and scale with coordinates.

### Federated Learning + DP (Anomaly 3)
- **Clipping norm** (`CLIPPING_NORM`): tight bounds improve privacy but can slow learning.
- **Noise multiplier** (`NOISE_MULTIPLIER`): larger noise → stronger privacy, lower utility.
- **Client sampling**: fewer clients/round speeds up but may reduce stability.

---

## Troubleshooting

- **`diffprivlib` import errors:** Ensure `pip install diffprivlib` succeeded. If NumPy version conflicts, upgrade/downgrade NumPy within your environment.
- **`phe`/Paillier slow or failing on Windows:** Try smaller `n_length` for demos (e.g., 1024). Make sure your Python build is 64-bit.
- **`tensorflow` install issues:**
  - CPU-only: `pip install tensorflow-cpu`
  - Verify Python version compatibility (3.9–3.11 are broadly supported).
- **`transit_synthetic_30.csv` not found:** Provide a CSV at that path or edit `csv_path` to your file.
- **Out-of-memory during FL:** Reduce `BATCH_SIZE`, `SEQ_LEN`, or `NUM_CLIENTS`.

---

## References
- **Differential Privacy:** N. Holohan et al., IBM `diffprivlib` — Laplace mechanism.
- **Homomorphic Encryption:** Paillier cryptosystem (`phe` library).
- **Federated Learning:** McMahan et al., “Communication-Efficient Learning of Deep Networks from Decentralized Data” (FedAvg). DP-SGD concept per Abadi et al.

---

## Appendix — Detected Functions (from the notebook)
average_weights , client_decrypt_and_check , client_encrypt_location , create_model , decode_square_int , demo , dp_train_step , encode_coord , encode_square , generate_keys , generate_synthetic_client_data , laplace_mechanism , server_compute_encrypted_distance_squared , set_model_weights , shannon_entropy_from_userlist


