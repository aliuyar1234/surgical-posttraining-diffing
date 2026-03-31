from __future__ import annotations

import pytest

from src.analysis.common import build_split_audit, ensure_no_test_leakage


def test_split_audit_accepts_locked_train_tune_discipline() -> None:
    audit = build_split_audit(
        feature_summary_splits=["select_train", "select_tune"],
        candidate_scoring_splits=["select_train"],
        selector_fit_splits=["select_train"],
        forward_selection_splits=["select_tune"],
    )

    ensure_no_test_leakage(audit)
    assert audit["no_test_leakage"] is True


def test_split_audit_rejects_test_split_touch() -> None:
    audit = build_split_audit(
        feature_summary_splits=["select_train", "select_tune", "test"],
        candidate_scoring_splits=["select_train", "test"],
        selector_fit_splits=["select_train"],
        forward_selection_splits=["select_tune"],
    )

    with pytest.raises(ValueError, match="Test split leakage"):
        ensure_no_test_leakage(audit)


def test_split_audit_rejects_wrong_forward_selection_split() -> None:
    audit = build_split_audit(
        feature_summary_splits=["select_train", "select_tune"],
        candidate_scoring_splits=["select_train"],
        selector_fit_splits=["select_train"],
        forward_selection_splits=["select_train"],
    )

    with pytest.raises(ValueError, match="Forward selection must use only select_tune"):
        ensure_no_test_leakage(audit)
