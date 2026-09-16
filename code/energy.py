"""Energy measurement via direct NVML hardware counter reads.

SLURM cgroup isolation makes the allocated GPU appear as NVML index 0.
Energy is read from nvmlDeviceGetTotalEnergyConsumption(), a monotonic
hardware counter (millijoule resolution), at start()/stop() boundaries.

CodeCarbon is retained for CO2 emissions estimation and CPU/RAM energy.

Idle baseline subtraction following Iyengar et al.: measure idle GPU
power over 10s, subtract P_idle * wall_time from each measurement to
isolate dynamic computation energy.
"""

import time
import pynvml
from codecarbon import EmissionsTracker


class EnergyMonitor:
    def __init__(self, gpu_index=0):
        self._tracker = EmissionsTracker(
            log_level="error",
            tracking_mode="process",
            allow_multiple_runs=True,
        )
        self._tracker.start()
        self._task_counter = 0

        pynvml.nvmlInit()
        self.gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        self.gpu_name = pynvml.nvmlDeviceGetName(self.gpu_handle)

        info = pynvml.nvmlDeviceGetMemoryInfo(self.gpu_handle)
        print(f"EnergyMonitor: GPU 0 "
              f"({self.gpu_name}, {info.used/1e9:.1f}/{info.total/1e9:.1f} GB VRAM)")

        self.idle_power_watts = None
        self.call_log = []

    def measure_idle(self, duration=10):
        """Measure idle GPU power. Call once before experiments."""
        e0 = pynvml.nvmlDeviceGetTotalEnergyConsumption(self.gpu_handle)
        time.sleep(duration)
        e1 = pynvml.nvmlDeviceGetTotalEnergyConsumption(self.gpu_handle)
        self.idle_power_watts = (e1 - e0) / 1000.0 / duration
        return self.idle_power_watts

    def start(self):
        """Begin energy measurement for one LLM call or tool execution."""
        self._task_counter += 1
        self._task_name = f"call_{self._task_counter}"
        self._time_start = time.monotonic()
        self._nvml_energy_start = pynvml.nvmlDeviceGetTotalEnergyConsumption(
            self.gpu_handle
        )
        self._tracker.start_task(self._task_name)

    def stop(self, metadata=None):
        """End energy measurement. Returns record dict."""
        nvml_energy_end = pynvml.nvmlDeviceGetTotalEnergyConsumption(
            self.gpu_handle
        )
        task_emissions = self._tracker.stop_task(self._task_name)
        t_end = time.monotonic()
        wall_s = t_end - self._time_start

        gpu_j = (nvml_energy_end - self._nvml_energy_start) / 1000.0

        KWH_TO_J = 3_600_000
        if task_emissions is not None:
            cpu_j = task_emissions.cpu_energy * KWH_TO_J
            ram_j = task_emissions.ram_energy * KWH_TO_J
            total_j = gpu_j + cpu_j + ram_j
            emissions_kg = task_emissions.emissions
        else:
            cpu_j = 0
            ram_j = 0
            total_j = gpu_j
            emissions_kg = 0

        idle_j = (self.idle_power_watts or 0) * wall_s
        gpu_dynamic_j = max(0, gpu_j - idle_j)

        record = {
            "gpu_energy_joules": gpu_j,
            "gpu_dynamic_energy_joules": gpu_dynamic_j,
            "gpu_idle_energy_joules": idle_j,
            "cpu_energy_joules": cpu_j,
            "ram_energy_joules": ram_j,
            "total_energy_joules": total_j,
            "wall_seconds": wall_s,
            "avg_gpu_power_watts": gpu_j / wall_s if wall_s > 0 else 0,
            "emissions_kg_co2": emissions_kg,
        }
        if metadata:
            record.update(metadata)

        self.call_log.append(record)
        return record

    def shutdown(self):
        self._tracker.stop()
        pynvml.nvmlShutdown()
