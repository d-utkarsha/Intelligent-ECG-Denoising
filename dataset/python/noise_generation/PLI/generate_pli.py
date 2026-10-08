import wfdb
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


# ============================================================
# PROJECT PATHS
# ============================================================

DATASET_PATH = Path(
    r"C:\Users\dutka\OneDrive\Documents\DU\Intelligent-ECG-Denoising\dataset"
)

# Original NSRDB dataset
NSRDB_PATH = DATASET_PATH / "NSRDB"

# PLI output folders
PLI_PATH = (
    DATASET_PATH
    / "python"
    / "noise_generation"
    / "PLI"
)

DATA_OUTPUT = PLI_PATH / "output" / "data"
GRAPH_OUTPUT = PLI_PATH / "output" / "graphs"

DATA_OUTPUT.mkdir(parents=True, exist_ok=True)
GRAPH_OUTPUT.mkdir(parents=True, exist_ok=True)


# ============================================================
# ECG SETTINGS
# ============================================================

RECORD = "16265"

# ECG1 = channel 0
CHANNEL = 0

# Use first 10 seconds for this initial dataset
DURATION = 3

# Power-line frequency in India
PLI_FREQUENCY = 50

# SNR levels to generate
SNR_LEVELS = [24, 18, 12, 6]


# ============================================================
# READ ORIGINAL ECG
# ============================================================

record_path = NSRDB_PATH / RECORD

signal, fields = wfdb.rdsamp(str(record_path))

fs = fields["fs"]

# Number of samples required
num_samples = int(DURATION * fs)

# Extract ECG1
ecg = signal[:num_samples, CHANNEL]

# Time axis
time = np.arange(num_samples) / fs


# ============================================================
# DISPLAY BASIC INFORMATION
# ============================================================

print()
print("=" * 65)
print("              PLI DATASET GENERATOR")
print("=" * 65)

print(f"Record              : {RECORD}")
print("Channel             : ECG1")
print(f"Sampling rate       : {fs} Hz")
print(f"Duration            : {DURATION} seconds")
print(f"Number of samples   : {len(ecg)}")
print(f"PLI frequency       : {PLI_FREQUENCY} Hz")
print(f"SNR levels          : {SNR_LEVELS}")

print("=" * 65)
print()


# ============================================================
# GENERATE DATA FOR EACH SNR
# ============================================================

for snr_db in SNR_LEVELS:

    print(f"Generating PLI at {snr_db} dB...")


    # --------------------------------------------------------
    # Generate 50 Hz sinusoidal interference
    # --------------------------------------------------------

    pli = np.sin(
        2 * np.pi * PLI_FREQUENCY * time
    )


    # --------------------------------------------------------
    # Calculate required noise power
    # --------------------------------------------------------

    signal_power = np.mean(ecg ** 2)

    noise_power = signal_power / (
        10 ** (snr_db / 10)
    )


    # --------------------------------------------------------
    # Scale noise to desired SNR
    # --------------------------------------------------------

    pli = pli * np.sqrt(
        noise_power / np.mean(pli ** 2)
    )


    # --------------------------------------------------------
    # Add noise to ECG
    # --------------------------------------------------------

    noisy_ecg = ecg + pli


    # --------------------------------------------------------
    # Save numerical data as NPZ
    # --------------------------------------------------------

    data_filename = (
        f"{RECORD}_ECG1_PLI_{snr_db}dB.npz"
    )

    data_path = DATA_OUTPUT / data_filename

    np.savez(
        data_path,

        # Time
        time=time,

        # Original ECG
        clean_ecg=ecg,

        # Generated PLI
        pli=pli,

        # Noisy ECG
        noisy_ecg=noisy_ecg,

        # Metadata
        sampling_rate=fs,
        pli_frequency=PLI_FREQUENCY,
        snr_db=snr_db,
        record=RECORD,
        channel="ECG1"
    )


    # --------------------------------------------------------
    # Create figure
    # --------------------------------------------------------

    fig, axes = plt.subplots(
        3,
        1,
        figsize=(14, 9),
        sharex=True
    )


    # --------------------------------------------------------
    # Plot 1 — Clean ECG
    # --------------------------------------------------------

    axes[0].plot(
        time,
        ecg
    )

    axes[0].set_title(
        f"{RECORD} ECG1 - Clean Reference"
    )

    axes[0].set_ylabel(
        "Amplitude (mV)"
    )

    axes[0].grid(True)


    # --------------------------------------------------------
    # Plot 2 — PLI
    # --------------------------------------------------------

    axes[1].plot(
        time,
        pli
    )

    axes[1].set_title(
        f"{PLI_FREQUENCY} Hz Power-Line Interference"
    )

    axes[1].set_ylabel(
        "Amplitude (mV)"
    )

    axes[1].grid(True)


    # --------------------------------------------------------
    # Plot 3 — Noisy ECG
    # --------------------------------------------------------

    axes[2].plot(
        time,
        noisy_ecg
    )

    axes[2].set_title(
        f"{RECORD} ECG1 + {PLI_FREQUENCY} Hz PLI "
        f"(SNR = {snr_db} dB)"
    )

    axes[2].set_xlabel(
        "Time (seconds)"
    )

    axes[2].set_ylabel(
        "Amplitude (mV)"
    )

    axes[2].grid(True)


    # --------------------------------------------------------
    # Save graph as PNG
    # --------------------------------------------------------

    plt.tight_layout()

    graph_filename = (
        f"{RECORD}_ECG1_PLI_{snr_db}dB.png"
    )

    graph_path = GRAPH_OUTPUT / graph_filename

    plt.savefig(
        graph_path,
        dpi=300,
        bbox_inches="tight"
    )


    # --------------------------------------------------------
    # Show graph
    # --------------------------------------------------------

    plt.show()

    plt.close()


    # --------------------------------------------------------
    # Confirmation
    # --------------------------------------------------------

    print(f"  Data  → {data_path}")
    print(f"  Graph → {graph_path}")
    print()


# ============================================================
# COMPLETION
# ============================================================

print("=" * 65)
print("PLI DATASET GENERATION COMPLETE")
print("=" * 65)

print()
print("Generated SNR levels:")

for snr_db in SNR_LEVELS:
    print(f"  ✓ {snr_db} dB")

print()
print("Data saved in:")
print(DATA_OUTPUT)

print()
print("Graphs saved in:")
print(GRAPH_OUTPUT)

print()
print("Done!")