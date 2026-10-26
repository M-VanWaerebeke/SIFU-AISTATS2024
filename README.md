# SIFU — Sequential Informed Federated Unlearning

Research implementation for **SIFU: Sequential Informed Federated Unlearning for Efficient and Provable Client Unlearning in Federated Optimization**, AISTATS 2024.

[Paper and citation](https://proceedings.mlr.press/v238/fraboni24a.html) · [Original collaborative repository](https://github.com/Accenture/Labs-Federated-Learning/tree/SIFU) · [Experiment parameters](docs/experiment-parameters.md)

SIFU removes a client's contribution from a federated model by choosing a suitable earlier model, adding calibrated noise, and resuming optimization with the remaining clients. The implementation includes SIFU and comparison methods such as retraining, fine-tuning, FedEraser and FedAccum.

Authors: Yann Fraboni, Martin Van Waerebeke, Kevin Scaman, Richard Vidal, Laetitia Kameni and Marco Lorenzi. Yann Fraboni and Martin Van Waerebeke contributed equally to the paper.

## Quick start: an offline CPU example

The example trains a small linear classifier across four synthetic clients, then runs the actual SIFU code to forget client 0. No image datasets, GPU, account or checkpoint download is needed.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.txt
python examples/cpu_demo.py --output-dir demo-output
python -m pytest -q
```

Run these commands from the repository root. The example performs four training rounds followed by four unlearning rounds, writes model checkpoints and histories under `demo-output/saved_exp_info/`, and produces `demo-output/summary.json`. Its expected summary contains `clients_before: 4`, `forgotten_clients: [0]` and `clients_after: 3`. It checks execution, rather than reproducing the paper's performance or verifying its theoretical guarantees.

The supported CPU checks were run with Python 3.9.6, PyTorch 2.8.0, torchvision 0.23.0 and NumPy 2.0.2. The paper originally used Python 3.9 and PyTorch 1.11.0. The requirements describe the currently checked execution environment; equivalence to the original full benchmark has not been established.

## Paper experiments

`FU.py` provides the experiment command line. It first needs a `train` run with the same model and training parameters before a `SIFU` run can load the saved model. See the [full parameter reference](docs/experiment-parameters.md).

```bash
bash SIFU_example.sh
```

This is the historical CIFAR-10 Figure 2 experiment recipe, using 100 clients, ten seeds and up to 10,000 rounds. It downloads CIFAR-10 and launches multiple processes; inspect the script and resource requirements before using it on a shared machine. The offline example above is the small supported starting point. Full paper experiments have not been rerun during this maintenance update.

The historical experiment reports training-client metrics; these should not be described as held-out test accuracy. Dataset acquisition and partitioning are implemented in `FL/read_db.py`. CelebA additionally requires its own data access/download setup.

## Code map

| Path | Purpose |
| --- | --- |
| `FU.py` | Train and unlearn entrypoint |
| `FL/server.py`, `FL/branch.py` | Aggregation, unlearning metric and model rewind |
| `FL/client.py` | Client SGD updates |
| `FL/read_db.py`, `FL/create_model.py` | Data partitions and models |
| `FedAccum/` | FedAccum/FedEraser comparison implementation |
| `policy.py` | Client-removal request sequences |
| `tests/` | Supported offline regression and integration checks |
| `test/` | Historical tests targeting earlier interfaces; retained for provenance |

## Contribution and provenance

This repository remains a fork of Accenture's collaborative federated-learning code. Its history records earlier implementation by Yann Fraboni and contributions under his other commit identity. Martin Van Waerebeke's historical GitHub identity, `Mart1vw`, authored the [AISTATS 2024 code update](https://github.com/M-VanWaerebeke/SIFU-AISTATS2024/commit/e940485948791cda9bc05fb4f9e199d95d3f68e4), which updated client/server routines, models, data handling, experiment recording and the reproduction script, followed by cleanup commits. These records demonstrate specific implementation contributions; the work is collaborative.

The release maintenance adds an offline example, regression checks, explicit dependency versions and fixes for output creation and the experiment script. This is research software, with the checked scope described above. The repository has no explicit software license; this maintenance update does not add one or change the upstream licensing status. Please contact the authors about reuse terms.

If using the research, cite the published paper; machine-readable metadata is in [CITATION.cff](CITATION.cff).
