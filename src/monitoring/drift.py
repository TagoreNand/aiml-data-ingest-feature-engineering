"""src/monitoring/drift.py — Data drift detection with Evidently + auto-retrain trigger."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.utils.config import load_config
from src.utils.logger import logger
from src.utils.schema import DriftReport


class DriftDetector:
    """Compares current production data distribution against a reference dataset."""

    def __init__(self, config: dict | None = None) -> None:
        self.cfg = config or load_config()
        self.mon_cfg = self.cfg.get("monitoring", {}).get("drift", {})
        self.ref_path = Path(self.mon_cfg.get("reference_path", "data/processed/reference_dataset.parquet"))
        self._reference: pd.DataFrame | None = None

    def _load_reference(self) -> pd.DataFrame:
        if self._reference is not None:
            return self._reference
        if not self.ref_path.exists():
            raise FileNotFoundError(f"Reference dataset not found: {self.ref_path}")
        self._reference = pd.read_parquet(self.ref_path)
        return self._reference

    # ── Evidently-based full report ───────────────────────────────────────────

    def run_evidently(self, current: pd.DataFrame) -> DriftReport:
        try:
            from evidently.report import Report
            from evidently.metric_preset import DataDriftPreset

            report = Report(metrics=[DataDriftPreset()])
            report.run(reference_data=self._load_reference(), current_data=current)
            result = report.as_dict()

            drift_result = result["metrics"][0]["result"]
            drifted_cols = [
                col for col, info in drift_result.get("drift_by_columns", {}).items()
                if info.get("drift_detected", False)
            ]
            drift_score = drift_result.get("share_of_drifted_columns", 0.0)
            dataset_drift = drift_result.get("dataset_drift", False)

        except Exception as exc:
            logger.warning(f"Evidently drift check failed ({exc}). Falling back to PSI.")
            return self._psi_fallback(current)

        return DriftReport(
            timestamp=datetime.now(timezone.utc),
            dataset_drift=dataset_drift,
            drift_score=drift_score,
            drifted_columns=drifted_cols,
            details=result,
        )

    # ── PSI fallback ──────────────────────────────────────────────────────────

    def _psi_fallback(self, current: pd.DataFrame) -> DriftReport:
        reference = self._load_reference()
        threshold = self.mon_cfg.get("psi_threshold", 0.2)
        drifted_cols, scores = [], {}

        for col in reference.select_dtypes(include=["number"]).columns:
            if col not in current.columns:
                continue
            psi = self._psi(reference[col].dropna(), current[col].dropna())
            scores[col] = psi
            if psi > threshold:
                drifted_cols.append(col)

        drift_score = sum(v > threshold for v in scores.values()) / max(len(scores), 1)
        return DriftReport(
            timestamp=datetime.now(timezone.utc),
            dataset_drift=drift_score > 0.3,
            drift_score=drift_score,
            drifted_columns=drifted_cols,
            details={"psi_scores": scores},
        )

    @staticmethod
    def _psi(expected: pd.Series, actual: pd.Series, buckets: int = 10) -> float:
        """Population Stability Index."""
        min_val = min(expected.min(), actual.min())
        max_val = max(expected.max(), actual.max())
        breakpoints = pd.cut(expected, bins=buckets, retbins=True)[1]
        breakpoints[0], breakpoints[-1] = min_val - 1e-9, max_val + 1e-9
        expected_pct = pd.cut(expected, bins=breakpoints).value_counts(normalize=True).sort_index()
        actual_pct = pd.cut(actual, bins=breakpoints).value_counts(normalize=True).sort_index()
        expected_pct = expected_pct.clip(lower=1e-6)
        actual_pct = actual_pct.reindex(expected_pct.index, fill_value=1e-6)
        return float(((expected_pct - actual_pct) * (expected_pct / actual_pct).apply(lambda x: x.__class__(x) and __import__('math').log(x))).sum())

    # ── Persist report ────────────────────────────────────────────────────────

    def save_report(self, report: DriftReport) -> Path:
        out_dir = Path("logs/drift_reports")
        out_dir.mkdir(parents=True, exist_ok=True)
        out = out_dir / f"drift_{report.timestamp.strftime('%Y%m%dT%H%M%S')}.json"
        out.write_text(report.model_dump_json(indent=2))
        return out


class RetrainingTrigger:
    """Decides whether to kick off a retraining job based on drift + feedback volume."""

    def __init__(self, config: dict | None = None) -> None:
        self.cfg = config or load_config()
        self.fb_cfg = self.cfg.get("monitoring", {}).get("feedback", {})
        self.feedback_path = Path(self.fb_cfg.get("collection_path", "data/raw/feedback"))

    def _count_feedback(self) -> int:
        if not self.feedback_path.exists():
            return 0
        total = 0
        for f in self.feedback_path.glob("*.jsonl"):
            with f.open() as fh:
                total += sum(1 for _ in fh)
        return total

    def should_retrain(self, drift_report: DriftReport) -> tuple[bool, str]:
        drift_threshold = self.fb_cfg.get("retrain_trigger_drift_score", 0.25)
        min_samples = self.fb_cfg.get("min_samples_for_retrain", 500)
        feedback_count = self._count_feedback()

        if drift_report.dataset_drift and drift_report.drift_score >= drift_threshold:
            return True, f"Dataset drift detected (score={drift_report.drift_score:.3f})"
        if feedback_count >= min_samples:
            return True, f"Feedback threshold reached ({feedback_count} samples)"
        return False, f"No trigger (drift={drift_report.drift_score:.3f}, feedback={feedback_count})"

    def trigger_retrain(self, reason: str) -> None:
        logger.warning(f"RETRAIN TRIGGERED: {reason}")
        # In production this would enqueue a job via Airflow, Argo, or a Celery task.
        trigger_path = Path("logs/retrain_triggers.jsonl")
        trigger_path.parent.mkdir(parents=True, exist_ok=True)
        with trigger_path.open("a") as fh:
            fh.write(json.dumps({"timestamp": datetime.now(timezone.utc).isoformat(), "reason": reason}) + "\n")
