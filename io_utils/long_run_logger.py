"""Stream large training runs and keep a recoverable PPO checkpoint."""

import csv
import json
import shutil
import time
from pathlib import Path

from config.paths import PROJECT_ROOT, RESULTS_DIR, SERVER_INFO_PATH, TRAIN_TASK_PATH


class LongRunLogger:
    CHECKPOINT_INTERVAL = 100
    LOG_COLUMNS = (
        "Episode", "Episode Reward", "task_Avg_Delay",
        "RSR_Correlated", "RSR_Independent", "Mask_Fallback_Rate",
    )

    def __init__(self, params, model, run_id):
        self.params = params
        self.model = model
        self.model_name = str(params.model_summary).strip().lower()
        self.run_dir = Path(RESULTS_DIR) / f"{self.model_name}_{run_id}_{params.EXPERIMENT_TAG}"
        self.run_dir.mkdir(parents=True, exist_ok=False)
        self.task_results_csv_path = self.run_dir / "task_results.csv"
        self.task_writer = None
        self.last_completed_episode = 0
        self.checkpoint_episode = 0
        self.started = time.monotonic()
        self.task_output = None
        self.log_output = None

    def __enter__(self):
        settings = {
            key: value for key, value in vars(self.params).items()
            if not key.startswith("_")
        }
        settings["model_summary"] = self.model_name
        (self.run_dir / "parameters.json").write_text(
            json.dumps(settings, indent=2), encoding="utf-8",
        )
        for source in (
            Path(PROJECT_ROOT) / "config" / "configuration.py",
            Path(SERVER_INFO_PATH), Path(TRAIN_TASK_PATH),
        ):
            shutil.copy2(source, self.run_dir / source.name)
        self.task_output = self.task_results_csv_path.open("x", newline="", encoding="utf-8")
        self.log_output = (self.run_dir / "logs.csv").open("x", newline="", encoding="utf-8")
        self.log_writer = csv.writer(self.log_output)
        self.log_writer.writerow(self.LOG_COLUMNS)
        self._write_status("running")
        print(f"Streaming training results: {self.run_dir}", flush=True)
        return self

    def __call__(self, loop):
        # Called after env.run(), when both replicas' outcome data is complete.
        if loop.task_results:
            if self.task_writer is None:
                self.task_writer = csv.DictWriter(
                    self.task_output, fieldnames=list(loop.task_results[0]),
                )
                self.task_writer.writeheader()
            self.task_writer.writerows(loop.task_results)
            self.task_output.flush()
            loop.task_results.clear()
        self.log_writer.writerow(loop.log_data[-1])
        self.log_output.flush()
        self.last_completed_episode = loop.this_episode
        if str(self.params.model_summary).strip().lower() == "ppo" and (
            loop.this_episode % self.CHECKPOINT_INTERVAL == 0
            or loop.this_episode == loop.total_episodes
        ):
            temporary = self.run_dir / "checkpoint_latest.tmp"
            self.model.save_model(temporary)
            temporary.replace(self.run_dir / "checkpoint_latest.pt")
            self.checkpoint_episode = loop.this_episode
            print(f"Saved checkpoint at episode {loop.this_episode}", flush=True)
        self._write_status("running")

    def _write_status(self, status, error=None):
        temporary = self.run_dir / "status.tmp"
        temporary.write_text(json.dumps({
            "status": status,
            "last_completed_episode": self.last_completed_episode,
            "total_episodes": self.params.total_episodes,
            "checkpoint_episode": self.checkpoint_episode,
            "elapsed_seconds": time.monotonic() - self.started,
            "error": error,
        }, indent=2), encoding="utf-8")
        temporary.replace(self.run_dir / "status.json")

    def __exit__(self, exc_type, exc_value, traceback):
        for output in (self.task_output, self.log_output):
            if output is not None:
                output.close()
        self._write_status(
            "completed" if exc_type is None else "failed",
            None if exc_type is None else str(exc_value),
        )
        return False
