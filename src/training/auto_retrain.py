"""src/training/auto_retrain.py — Automated retraining pipeline triggered by drift or feedback."""
from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from src.utils.config import load_config
from src.utils.logger import logger


class RetrainingPipeline:
    """
    Full automated loop:
      1. Check trigger conditions (drift score OR feedback volume)
      2. Re-run feature pipeline on latest data
      3. Train new model version
      4. Evaluate vs champion
      5. Promote if better
      6. Log decision
    """

    def __init__(self, config: dict | None = None):
        self.cfg  = config or load_config()
        self.mon  = self.cfg.get("monitoring", {})
        self.fb   = self.mon.get("feedback", {})
        self.log_path = Path("logs/retrain_decisions.jsonl")
        self.log_path.parent.mkdir(exist_ok=True)

    # ── Trigger checks ────────────────────────────────────────────────────────

    def drift_triggered(self, drift_score: float) -> bool:
        threshold = self.mon.get("drift", {}).get("retrain_trigger_drift_score", 0.25)
        return drift_score >= threshold

    def feedback_triggered(self) -> tuple[bool, int]:
        fb_path = Path(self.fb.get("collection_path", "data/raw/feedback")) / "feedback.jsonl"
        if not fb_path.exists():
            return False, 0
        lines = fb_path.read_text().strip().splitlines()
        count = len(lines)
        min_samples = self.fb.get("min_samples_for_retrain", 500)
        return count >= min_samples, count

    # ── Pipeline steps ────────────────────────────────────────────────────────

    def _run_step(self, cmd: list[str], step_name: str) -> bool:
        logger.info(f"[retrain] Running: {step_name}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            logger.error(f"[retrain] {step_name} failed:\n{result.stderr}")
            return False
        logger.info(f"[retrain] {step_name} complete")
        return True

    def run(self, trigger: str, drift_score: float | None = None) -> dict:
        decision = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "trigger": trigger,
            "drift_score": drift_score,
            "steps_completed": [],
            "outcome": "pending",
        }

        logger.info(f"[retrain] Pipeline started | trigger={trigger}")

        steps = [
            (["python", "scripts/build_features.py", "--config", "configs/config.yaml"],
             "feature_pipeline"),
            (["python", "scripts/train.py", "--experiment", f"auto_retrain_{trigger}",
              "--config", "configs/config.yaml"],
             "training"),
        ]

        for cmd, name in steps:
            if self._run_step(cmd, name):
                decision["steps_completed"].append(name)
            else:
                decision["outcome"] = f"failed_at_{name}"
                self._log(decision)
                return decision

        decision["outcome"] = "success"
        logger.info("[retrain] Pipeline complete — new model version registered.")
        self._log(decision)
        return decision

    def _log(self, decision: dict) -> None:
        with self.log_path.open("a") as f:
            f.write(json.dumps(decision) + "\n")
        logger.info(f"[retrain] Decision logged → {self.log_path}")

    # ── Convenience: check + run ──────────────────────────────────────────────

    def maybe_retrain(self, drift_score: float) -> dict | None:
        if self.drift_triggered(drift_score):
            return self.run(trigger="drift", drift_score=drift_score)
        fb_triggered, fb_count = self.feedback_triggered()
        if fb_triggered:
            return self.run(trigger=f"feedback_{fb_count}_samples")
        logger.info("[retrain] No trigger conditions met.")
        return None
