"""Small checks of the supported release path; no dataset downloads."""
import pickle
from types import SimpleNamespace

import numpy as np
import torch

from examples.cpu_demo import run_demo
from FL.branch import Branch
from FL.experiment import Experiment


def test_cpu_train_then_unlearn(tmp_path):
    summary = run_demo(tmp_path)
    checkpoint = torch.load(tmp_path / summary["checkpoint"], weights_only=True)
    assert checkpoint
    assert all(torch.isfinite(tensor).all() for tensor in checkpoint.values())
    with next((tmp_path / "saved_exp_info/metric").glob("*SIFU*.pkl")).open("rb") as f:
        metric = pickle.load(f)
    assert metric.shape == (2, 5, 4)
    assert np.isfinite(metric).all()
    # The forgotten client receives no new contribution after the request.
    assert np.all(metric[1, :, 0] == 0)


def test_branch_rewinds_before_budget_is_exceeded():
    metric = np.zeros((1, 5, 3))
    metric[0, :, 0] = [0, 0.1, 0.3, 0.8, 1.2]
    branch = Branch()
    branch.update(metric, 1, [0], 0.5)
    assert branch() == [[0, 0], [0, 2]]


def test_zero_accuracy_history_can_be_saved(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    exp = Experiment.__new__(Experiment)
    exp.limit_train_iter = 0
    exp.file_name = "zero-accuracy"
    server = SimpleNamespace(t=2, acc=np.zeros((1, 3, 2)),
                             loss=np.ones((1, 3, 2)))
    exp.hists_save(server, "acc", "loss")
    with open("saved_exp_info/acc/zero-accuracy.pkl", "rb") as f:
        assert pickle.load(f).shape == (1, 3, 2)
