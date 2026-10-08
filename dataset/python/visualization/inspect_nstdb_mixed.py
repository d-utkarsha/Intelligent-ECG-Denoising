
import wfdb
import matplotlib.pyplot as plt
from pathlib import Path

# NSTDB location
NSTDB_PATH = Path(
    r"C:\Users\dutka\OneDrive\Documents\DU\Intelligent-ECG-Denoising\dataset\NSTDB"
)

# 118 mixed-noise records
records_118 = [
    "118e24",
    "118e18",
    "118e12",
    "118e06",
    "118e00",
    "118e_6"
]

# Plot from 5 minutes onward
start_time = 300
duration = 20

fig, axes = plt.subplots(6, 1, figsize=(14, 14), sharex=True)

for ax, record in zip(axes, records_118):

    record_path = NSTDB_PATH / record

    signal, fields = wfdb.rdsamp(str(record_path))

    fs = fields["fs"]

    start_sample = int(start_time * fs)
    end_sample = start_sample + int(duration * fs)

    ecg = signal[start_sample:end_sample, 0]

    time = [
        start_time + i / fs
        for i in range(len(ecg))
    ]

    ax.plot(time, ecg)
    ax.set_ylabel(record)
    ax.grid(True, alpha=0.3)

axes[-1].set_xlabel("Time (seconds)")

fig.suptitle(
    "NSTDB Record 118 — ECG + Electrode-Motion Artifact",
    fontsize=16
)

plt.tight_layout()
plt.show()