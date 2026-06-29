"""Region-ordering and NaN-handling tests for evaluation metrics (F05, F22).

The data pipeline stacks channels as ``torch.stack([tc, wt, et])`` (see
train_all.py / evaluate_checkpoint.py), so per-region arrays are ordered
[TC, WT, ET]. Any metric/analysis layer that labels region index i must use
that same order, or every per-region number is mislabelled.

The pure-ordering checks below need no torch/monai. The SegmentationMetrics
end-to-end check is monai-gated and skips when monai is unavailable.
"""

import numpy as np
import pytest

# Canonical data channel order, fixed by the dataset pipeline.
DATA_ORDER = ["TC", "WT", "ET"]


def test_segmentation_metrics_region_names_match_data_order():
    """SegmentationMetrics.REGION_NAMES must equal the data channel order.

    Regression for F05: it was ["ET","TC","WT"], permuting every per-region
    Dice/HD95 label. We read the class attribute without instantiating (which
    would require monai)."""
    import ast
    import pathlib

    src = pathlib.Path("src/evaluation/metrics.py").read_text()
    tree = ast.parse(src)
    region_names = None
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for tgt in node.targets:
                if isinstance(tgt, ast.Name) and tgt.id == "REGION_NAMES":
                    region_names = ast.literal_eval(node.value)
    assert region_names == DATA_ORDER, (
        f"REGION_NAMES={region_names} must match data channel order {DATA_ORDER}"
    )


def test_result_analyzer_region_order_matches_data_order():
    """ResultAnalyzer.REGIONS must equal the data channel order (F05)."""
    from src.analysis.result_analyzer import ResultAnalyzer

    assert ResultAnalyzer.REGIONS == DATA_ORDER


def test_trainer_dice_key_mapping_matches_data_order():
    """trainer.py must map dice_scores[0]->TC, [1]->WT, [2]->ET (F05).

    Parsed statically to avoid importing torch/monai. We check the source maps
    index 0 to dice_tc, not dice_et."""
    import pathlib
    import re

    src = pathlib.Path("src/training/trainer.py").read_text()
    # find "val/dice_xx": dice_scores[N]
    pairs = dict(
        (int(idx), name)
        for name, idx in re.findall(r'"val/dice_(\w+)":\s*dice_scores\[(\d)\]', src)
    )
    assert pairs.get(0) == "tc", f"index 0 must be TC, got {pairs.get(0)}"
    assert pairs.get(1) == "wt", f"index 1 must be WT, got {pairs.get(1)}"
    assert pairs.get(2) == "et", f"index 2 must be ET, got {pairs.get(2)}"


@pytest.mark.parametrize("region", DATA_ORDER)
def test_committed_eval_json_uses_data_order_labels(region):
    """The committed eval JSON keys are already in data order; this guards
    against a future relabel drifting away from [TC, WT, ET]."""
    import json
    import pathlib

    data = json.loads(
        pathlib.Path("experiments/local_results/oncoseg_eval.json").read_text()
        .replace("NaN", "null")
    )
    assert f"eval_dice_{region.lower()}" in data


def test_segmentation_metrics_end_to_end_region_labels():
    """End-to-end: a batch where TC/WT/ET have distinct, known Dice must come
    back labelled correctly. monai-gated."""
    pytest.importorskip("monai")
    import torch

    from src.evaluation.metrics import SegmentationMetrics

    m = SegmentationMetrics()
    # Build pred/target [B=1, C=3, ...] in order [TC, WT, ET].
    # TC: perfect, WT: half-overlap, ET: empty-but-predicted (Dice 0).
    shape = (1, 3, 8, 8, 8)
    target = torch.zeros(shape)
    pred = torch.zeros(shape)
    # TC channel 0: identical block -> Dice 1.0
    target[0, 0, :4] = 1
    pred[0, 0, :4] = 1
    # WT channel 1: target full, pred half -> Dice between 0 and 1
    target[0, 1] = 1
    pred[0, 1, :4] = 1
    # ET channel 2: target empty, pred has voxels -> Dice 0
    pred[0, 2, :2] = 1

    m.update(pred, target)
    res = m.compute()
    assert res["dice_TC"] == pytest.approx(1.0, abs=1e-3)
    assert 0.0 < res["dice_WT"] < 1.0
    assert res["dice_ET"] == pytest.approx(0.0, abs=1e-3)
