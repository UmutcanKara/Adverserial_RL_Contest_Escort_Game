import csv
import os
from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional
from datetime import datetime


class CSVLogger:
    def __init__(self, out_dir: str = "logs", run_name: str = "run"):
        id = datetime.now().strftime('%Y%m%d_%H%M%S')
        os.makedirs(out_dir, exist_ok=True)
        self.step_path = os.path.join(out_dir, f"{run_name}_{id}_steps.csv")
        self.ep_path   = os.path.join(out_dir, f"{run_name}_{id}_episodes.csv")
        self._step_file = None
        self._ep_file = None
        self._step_writer = None
        self._ep_writer = None

    def log_step(self, row: Dict[str, Any]) -> None:
        if self._step_file is None:
            self._step_file = open(self.step_path, "w", newline="")
            self._step_writer = csv.DictWriter(self._step_file, fieldnames=list(row.keys()))
            self._step_writer.writeheader()
        self._step_writer.writerow(row) # type: ignore

    def log_episode(self, row: Dict[str, Any]) -> None:
        if self._ep_file is None:
            self._ep_file = open(self.ep_path, "w", newline="")
            self._ep_writer = csv.DictWriter(self._ep_file, fieldnames=list(row.keys()))
            self._ep_writer.writeheader()
        self._ep_writer.writerow(row) # type: ignore

    def close(self) -> None:
        if self._step_file:
            self._step_file.close()
        if self._ep_file:
            self._ep_file.close()