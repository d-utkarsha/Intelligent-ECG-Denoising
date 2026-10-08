import wfdb
import matplotlib.pyplot as plt
from pathlib import Path

# --------------------------------------------------
# Find the NSTDB folder relative to this script
# --------------------------------------------------
NSTDB_PATH = Path(r"C:\Users\dutka\OneDrive\Documents\DU\Intelligent-ECG-Denoising\dataset\NSTDB")

# Noise records available in NSTDB
noise_records = ["bw", "ma", "em"]

# --------------------------------------------------
# Read and plot each noise signal
# --------------------------------------------------
for record_name in noise_records:

    record_path = NSTDB_PATH / record_name

    print(f"\nReading: {record_name}")
    print(f"Path: {record_path}")

    # Read WFDB record
    record = wfdb.rdrecord(str(record_path))

    signal = record.p_signal[:, 0]
    fs = record.fs

    print(f"Sampling frequency : {fs} Hz")
    print(f"Number of samples  : {len(signal)}")
    print(f"Duration           : {len(signal) / fs:.2f} seconds")

    # Show first 10 seconds
    seconds_to_plot = 10
    samples_to_plot = int(seconds_to_plot * fs)

    time = [i / fs for i in range(samples_to_plot)]

    plt.figure(figsize=(12, 4))
    plt.plot(time, signal[:samples_to_plot])

    plt.title(f"NSTDB Noise Signal — {record_name.upper()}")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Amplitude")
    plt.grid(True)

    plt.tight_layout()
    plt.show()