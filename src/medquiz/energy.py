"""GPU energy used by a stage, read from the driver's cumulative energy counter
(NVML, millijoules since the driver loaded). Falls back to None without a GPU."""

from __future__ import annotations

import time


class EnergyMeter:
    def __init__(self, device: int = 0):
        self.device = device
        self.joules: float | None = None
        self.seconds: float = 0.0
        self.gpu: str | None = None
        self._start_mj: int | None = None
        self._handle = None

    def __enter__(self) -> EnergyMeter:
        self._t0 = time.perf_counter()
        try:
            import pynvml

            pynvml.nvmlInit()
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(self.device)
            self.gpu = pynvml.nvmlDeviceGetName(self._handle)
            self._start_mj = pynvml.nvmlDeviceGetTotalEnergyConsumption(self._handle)
        except Exception:
            self._handle = None
        return self

    def __exit__(self, *exc) -> None:
        self.seconds = time.perf_counter() - self._t0
        if self._handle is None:
            return
        try:
            import pynvml

            end = pynvml.nvmlDeviceGetTotalEnergyConsumption(self._handle)
            self.joules = (end - self._start_mj) / 1000.0
            pynvml.nvmlShutdown()
        except Exception:
            self.joules = None

    def as_dict(self) -> dict:
        return {"gpu": self.gpu, "seconds": round(self.seconds, 2), "gpu_joules": self.joules}
