import json
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path


class RunTimer:
    def __init__(self, *, run_id: str, target: str):
        self.run_id = run_id
        self.target = target
        self.stages: dict[str, float] = {}
        self._started = time.perf_counter()

    @contextmanager
    def stage(self, name: str):
        t0 = time.perf_counter()
        print(f"[timing] {name} ...")
        try:
            yield
        finally:
            elapsed = time.perf_counter() - t0
            self.stages[name] = round(elapsed, 2)
            print(f"[timing] {name}: {elapsed:.1f}s")

    def total_sec(self) -> float:
        return round(time.perf_counter() - self._started, 2)

    def summary(self) -> None:
        total = self.total_sec()
        print("[timing] --- summary ---")
        for name, sec in self.stages.items():
            pct = 100 * sec / total if total else 0
            print(f"[timing]   {name}: {sec:.1f}s ({pct:.0f}%)")
        print(f"[timing]   total: {total:.1f}s")

    def to_dict(self) -> dict:
        return {
            "run_id": self.run_id,
            "target": self.target,
            "stages_sec": self.stages,
            "total_sec": self.total_sec(),
            "recorded_at": datetime.now(timezone.utc).isoformat(),
        }

    def write(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")
