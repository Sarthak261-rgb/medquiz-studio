"""GPU time and energy, read from the driver's cumulative energy counter (NVML,
millijoules since the driver loaded). Nothing here is estimated: without an NVIDIA
GPU the energy fields are simply None.

A stage writes its running totals to run.json every few items, so an interrupted
run still has an accurate record of what it used up to that point."""

from __future__ import annotations

import time
from functools import lru_cache

from .runs import RunDir, utcnow


class GpuCounter:
    def __init__(self, device: int = 0):
        self.name: str | None = None
        self._handle = None
        try:
            import pynvml

            pynvml.nvmlInit()
            self._nvml = pynvml
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(device)
            self.name = pynvml.nvmlDeviceGetName(self._handle)
            self.millijoules()
        except Exception:
            self._handle = None

    def millijoules(self) -> int | None:
        if self._handle is None:
            return None
        try:
            return int(self._nvml.nvmlDeviceGetTotalEnergyConsumption(self._handle))
        except Exception:
            return None


@lru_cache(maxsize=1)
def gpu_counter() -> GpuCounter:
    return GpuCounter()


def joules_since(start_mj: int | None) -> float | None:
    now = gpu_counter().millijoules()
    if start_mj is None or now is None:
        return None
    return round((now - start_mj) / 1000.0, 3)


class StageTracker:
    """Accumulates wall time and GPU energy for one stage of one run. Totals add to
    whatever an earlier, interrupted invocation already recorded."""

    def __init__(self, run: RunDir, name: str, flush_every: int = 10):
        self.run = run
        self.name = name
        self.flush_every = flush_every
        self.prev = run.meta().get("stages", {}).get(name, {})
        self.counter = gpu_counter()
        self.start_mj = self.counter.millijoules()
        self.t0 = time.perf_counter()
        self.n = 0

    def tick(self, n: int = 1) -> None:
        self.n += n
        if self.n % self.flush_every == 0:
            self.flush()

    def flush(self) -> dict:
        joules = joules_since(self.start_mj)
        prev_j = self.prev.get("gpu_joules")
        total_j = None if joules is None else round((prev_j or 0) + joules, 1)
        stage = {
            "n": self.prev.get("n", 0) + self.n,
            "seconds": round(self.prev.get("seconds", 0) + time.perf_counter() - self.t0, 1),
            "gpu_joules": total_j if total_j is not None else prev_j,
            "gpu": self.counter.name or self.prev.get("gpu"),
            "updated_at": utcnow(),
        }
        self.run.update_meta(stages={self.name: stage})
        return stage

    def __enter__(self) -> StageTracker:
        return self

    def __exit__(self, *exc) -> None:
        self.flush()
