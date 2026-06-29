"""Unit tests for loss functions."""

import pytest
import torch

# src.training.losses imports monai.losses at module load; skip cleanly if absent.
pytest.importorskip("monai")


class TestDiceCELoss:
    """Test DiceCE combined loss."""

    @pytest.fixture
    def loss_fn(self):
        from src.training.losses import DiceCELoss

        return DiceCELoss(dice_weight=0.5, ce_weight=0.5)

    def test_output_is_scalar(self, loss_fn):
        pred = torch.randn(2, 4, 16, 16, 16)
        target = torch.zeros(2, 4, 16, 16, 16)
        target[:, 0] = 1.0  # All background
        loss = loss_fn(pred, target)
        assert loss.dim() == 0  # Scalar

    def test_loss_is_positive(self, loss_fn):
        pred = torch.randn(2, 4, 16, 16, 16)
        target = torch.zeros(2, 4, 16, 16, 16)
        target[:, 0] = 1.0
        loss = loss_fn(pred, target)
        assert loss.item() > 0

    def test_perfect_prediction_low_loss(self, loss_fn):
        """A near-perfect prediction should have lower loss than random."""
        target = torch.zeros(1, 4, 16, 16, 16)
        target[:, 0] = 1.0  # All class 0

        # Near-perfect: high logit for class 0
        good_pred = torch.zeros(1, 4, 16, 16, 16)
        good_pred[:, 0] = 10.0

        # Random prediction
        bad_pred = torch.randn(1, 4, 16, 16, 16)

        good_loss = loss_fn(good_pred, target)
        bad_loss = loss_fn(bad_pred, target)

        assert good_loss.item() < bad_loss.item()

    def test_gradient_flows(self, loss_fn):
        pred = torch.randn(1, 4, 16, 16, 16, requires_grad=True)
        target = torch.zeros(1, 4, 16, 16, 16)
        target[:, 0] = 1.0
        loss = loss_fn(pred, target)
        loss.backward()
        assert pred.grad is not None
        assert pred.grad.shape == pred.shape


class TestDeepSupervisionLoss:
    """Test deep supervision weighted loss."""

    def test_weighted_sum(self):
        from src.training.losses import DeepSupervisionLoss, DiceCELoss

        base_loss = DiceCELoss()
        ds_loss = DeepSupervisionLoss(base_loss)

        target = torch.zeros(1, 4, 16, 16, 16)
        target[:, 0] = 1.0

        predictions = [
            torch.randn(1, 4, 16, 16, 16),
            torch.randn(1, 4, 16, 16, 16),
            torch.randn(1, 4, 16, 16, 16),
        ]

        loss = ds_loss(predictions, target)
        assert loss.dim() == 0
        assert loss.item() > 0

    def test_single_prediction(self):
        from src.training.losses import DeepSupervisionLoss, DiceCELoss

        base_loss = DiceCELoss()
        ds_loss = DeepSupervisionLoss(base_loss)

        target = torch.zeros(1, 4, 16, 16, 16)
        target[:, 0] = 1.0
        predictions = [torch.randn(1, 4, 16, 16, 16)]

        loss = ds_loss(predictions, target)
        assert loss.dim() == 0

    def test_default_weights_are_decreasing_and_normalised(self):
        """F25: the default scheme must be strictly decreasing per scale and
        normalised to sum to 1 (weights = (1/2**i), then divided by their sum).

        We probe the loss with identical predictions at every scale so the
        returned value equals (sum of weights) * base_loss = base_loss; we then
        probe with one scale's loss isolated to recover each effective weight.
        """
        from src.training.losses import DeepSupervisionLoss

        class _IndexLoss:
            """Base loss that returns the mean of `pred` so we can control the
            per-scale contribution precisely."""

            def __call__(self, pred, target):
                return pred.mean()

        n = 4
        ds_loss = DeepSupervisionLoss(_IndexLoss())
        # Recover effective weights: set scale k's pred to all-ones, others zero.
        target = torch.zeros(1, 3, 8, 8, 8)
        weights = []
        for k in range(n):
            preds = [torch.zeros(1, 3, 8, 8, 8) for _ in range(n)]
            preds[k] = torch.ones(1, 3, 8, 8, 8)
            weights.append(float(ds_loss(preds, target)))

        # Normalised: sum to 1.
        assert abs(sum(weights) - 1.0) < 1e-6
        # Strictly decreasing.
        for i in range(n - 1):
            assert weights[i] > weights[i + 1], f"weight[{i}] !> weight[{i+1}]"
        # Matches the documented (1/2**i)/Z scheme.
        raw = [1.0 / (2**i) for i in range(n)]
        Z = sum(raw)
        for w, r in zip(weights, raw):
            assert abs(w - r / Z) < 1e-6

    def test_multiscale_predictions_are_interpolated(self):
        """F09/F25: predictions at coarser scales than the target must not raise
        and must yield a finite scalar (the target is interpolated per scale)."""
        from src.training.losses import DeepSupervisionLoss, DiceCELoss

        ds_loss = DeepSupervisionLoss(DiceCELoss())
        target = torch.zeros(1, 3, 32, 32, 32)
        target[:, 0] = 1.0
        # Full-res, half-res, quarter-res logits (as a real DS head produces).
        predictions = [
            torch.randn(1, 3, 32, 32, 32),
            torch.randn(1, 3, 16, 16, 16),
            torch.randn(1, 3, 8, 8, 8),
        ]
        loss = ds_loss(predictions, target)
        assert loss.dim() == 0
        assert torch.isfinite(loss)
