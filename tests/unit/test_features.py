"""tests/unit/test_features.py — Unit tests for the feature pipeline."""
from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd
import pytest

from src.features.pipeline import FeatureTransformer, FeaturePipeline
from src.utils.schema import (
    RawRecord, ModelMetrics, PredictRequest,
    DriftReport, 
)


# ── Schema tests ──────────────────────────────────────────────────────────────

class TestSchemas:
    def test_raw_record_defaults(self):
        r = RawRecord(id="1", source="test", timestamp=datetime.now(timezone.utc), payload={"x": 1})
        assert r.schema_version == "1.0"

    def test_model_metrics_passes_threshold(self):
        m = ModelMetrics(f1=0.90, loss=0.20, accuracy=0.88)
        assert m.passes_threshold({"f1": 0.85}) is True
        assert m.passes_threshold({"f1": 0.95}) is False

    def test_model_metrics_none_fails_threshold(self):
        m = ModelMetrics(loss=0.1)
        assert m.passes_threshold({"f1": 0.5}) is False  # f1 is None

    def test_predict_request_validation(self):
        req = PredictRequest(inputs=["hello", "world"])
        assert len(req.inputs) == 2
        assert req.model_version == "champion"

    def test_drift_report_fields(self):
        report = DriftReport(
            timestamp=datetime.now(timezone.utc),
            dataset_drift=True,
            drift_score=0.35,
            drifted_columns=["age", "income"],
            details={},
        )
        assert report.drift_score == 0.35
        assert "age" in report.drifted_columns


# ── FeatureTransformer tests ──────────────────────────────────────────────────

class TestFeatureTransformer:
    def _sample_df(self):
        return pd.DataFrame({
            "age": [25, 30, 35, 40, 45],
            "income": [50000.0, 60000.0, 70000.0, 80000.0, 90000.0],
            "category": ["A", "B", "A", "C", "B"],
        })

    def test_standard_scaler(self):
        df = self._sample_df()
        t = FeatureTransformer().add_scaler("age", method="standard")
        result = t.fit_transform(df)
        assert abs(result["age"].mean()) < 1e-9
        assert abs(result["age"].std() - 1.0) < 0.2

    def test_minmax_scaler(self):
        df = self._sample_df()
        t = FeatureTransformer().add_scaler("income", method="minmax")
        result = t.fit_transform(df)
        assert result["income"].min() >= 0.0
        assert result["income"].max() <= 1.0

    def test_label_encoder(self):
        df = self._sample_df()
        t = FeatureTransformer().add_encoder("category")
        result = t.fit_transform(df)
        assert result["category"].dtype in [int, "int64", "int32"]

    def test_custom_transform(self):
        df = self._sample_df()
        t = FeatureTransformer().add_custom("double_age", lambda d: d.assign(age_doubled=d["age"] * 2))
        result = t.fit_transform(df)
        assert "age_doubled" in result.columns
        assert result["age_doubled"].iloc[0] == 50

    def test_transform_before_fit_raises(self):
        df = self._sample_df()
        t = FeatureTransformer().add_scaler("age")
        # Should raise because scaler is not fitted
        with pytest.raises(Exception):
            t.transform(df)  # sklearn raises NotFittedError


# ── FeaturePipeline static methods ───────────────────────────────────────────

class TestFeaturePipelineStatics:
    def test_add_text_length(self):
        df = pd.DataFrame({"text": ["hello world", "foo"]})
        result = FeaturePipeline._add_text_length(df)
        assert "text_len" in result.columns
        assert "word_count" in result.columns
        assert result["text_len"].iloc[0] == 11
        assert result["word_count"].iloc[0] == 2

    def test_add_text_length_no_text_col(self):
        df = pd.DataFrame({"value": [1, 2, 3]})
        result = FeaturePipeline._add_text_length(df)
        assert "text_len" not in result.columns

    def test_add_time_features(self):
        df = pd.DataFrame({"created_at": pd.to_datetime(["2024-01-15 10:30:00", "2024-06-20 18:00:00"])})
        result = FeaturePipeline._add_time_features(df)
        assert "created_at_hour" in result.columns
        assert result["created_at_hour"].iloc[0] == 10
        assert "created_at_month" in result.columns
