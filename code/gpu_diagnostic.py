"""GPU energy measurement diagnostic.

Run via: sbatch --array=0-6 scripts/gpu_diagnostic.sbatch
Each array task gets a different GPU. Prints which GPU NVML and
CodeCarbon see, and whether energy readings are correct.
"""

import os
import time
import pynvml
from codecarbon import EmissionsTracker

print("=" * 60)
print("GPU DIAGNOSTIC")
print("=" * 60)

# 1. Environment variables
print("\n--- Environment ---")
for var in ['CUDA_VISIBLE_DEVICES', 'SLURM_JOB_GPUS', 'SLURM_STEP_GPUS',
            'SLURM_GPUS_ON_NODE', 'GPU_DEVICE_ORDINAL', 'NVIDIA_VISIBLE_DEVICES',
            'SLURM_JOB_ID', 'SLURM_ARRAY_TASK_ID', 'SLURM_NODELIST']:
    val = os.environ.get(var)
    if val is not None:
        print(f"  {var}={val}")

# 2. NVML device enumeration
print("\n--- NVML devices ---")
pynvml.nvmlInit()
count = pynvml.nvmlDeviceGetCount()
print(f"  nvmlDeviceGetCount() = {count}")
for i in range(count):
    h = pynvml.nvmlDeviceGetHandleByIndex(i)
    name = pynvml.nvmlDeviceGetName(h)
    info = pynvml.nvmlDeviceGetMemoryInfo(h)
    power = pynvml.nvmlDeviceGetPowerUsage(h) / 1000.0  # mW -> W
    print(f"  GPU {i}: {name}, VRAM {info.used/1e9:.1f}/{info.total/1e9:.1f} GB, power {power:.0f}W")

# 3. NVML energy counter test on GPU 0
print("\n--- NVML energy counter (GPU 0, 5s idle) ---")
h0 = pynvml.nvmlDeviceGetHandleByIndex(0)
e0 = pynvml.nvmlDeviceGetTotalEnergyConsumption(h0)
time.sleep(5)
e1 = pynvml.nvmlDeviceGetTotalEnergyConsumption(h0)
idle_j = (e1 - e0) / 1000.0
idle_w = idle_j / 5.0
print(f"  Energy delta: {idle_j:.1f} J over 5s = {idle_w:.1f} W")
print(f"  (expect ~50-70W for idle Ada, ~20W for idle A5000)")

# 4. CodeCarbon GPU tracking
print("\n--- CodeCarbon ---")
tracker = EmissionsTracker(
    log_level="error",
    tracking_mode="process",
    allow_multiple_runs=True,
    measure_power_secs=0.1,
)
tracker.start()

# Check CodeCarbon internals
for attr in ['_gpu_ids', '_hardware']:
    val = getattr(tracker, attr, 'NOT FOUND')
    if attr == '_hardware' and val != 'NOT FOUND':
        gpu = getattr(val, 'gpu', None)
        if gpu:
            print(f"  _hardware.gpu = {gpu}")
            print(f"  _hardware.gpu.gpu_ids = {getattr(gpu, 'gpu_ids', 'NOT FOUND')}")
            print(f"  _hardware.gpu.gpu_model = {getattr(gpu, 'gpu_model', 'NOT FOUND')}")
    else:
        print(f"  {attr} = {val}")

# CodeCarbon energy test (5s idle)
tracker.start_task("test_idle")
time.sleep(5)
result = tracker.stop_task("test_idle")
if result is not None:
    KWH_TO_J = 3_600_000
    cc_gpu_j = result.gpu_energy * KWH_TO_J
    cc_gpu_w = cc_gpu_j / 5.0
    print(f"  CodeCarbon GPU energy: {cc_gpu_j:.1f} J over 5s = {cc_gpu_w:.1f} W")
else:
    print(f"  CodeCarbon returned None (no sample captured)")

tracker.stop()

# 5. Summary
print("\n--- Summary ---")
print(f"  NVML GPU 0 idle power: {idle_w:.1f} W")
if result is not None:
    print(f"  CodeCarbon idle power: {cc_gpu_w:.1f} W")
    if abs(idle_w - cc_gpu_w) < 20:
        print("  MATCH: NVML and CodeCarbon agree")
    else:
        print("  MISMATCH: NVML and CodeCarbon disagree!")
print()

pynvml.nvmlShutdown()
