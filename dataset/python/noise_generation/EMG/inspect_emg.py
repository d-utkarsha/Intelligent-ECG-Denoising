import wfdb
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.signal import resample_poly
from math import gcd


# ============================================================
# PATHS
# ============================================================

DATASET_PATH = Path(
    r"C:\Users\dutka\OneDrive\Documents\DU\Intelligent-ECG-Denoising\dataset"
)

NSRDB_PATH = DATASET_PATH / "NSRDB"
NSTDB_PATH = DATASET_PATH / "NSTDB"

EMG_PATH = NSTDB_PATH / "ma"

OUTPUT_PATH = (
    DATASET_PATH
    / "python"
    / "noise_generation"
    / "EMG"
    / "output"
)

DATA_OUTPUT = OUTPUT_PATH / "data"
GRAPH_OUTPUT = OUTPUT_PATH / "graphs"

DATA_OUTPUT.mkdir(parents=True, exist_ok=True)
GRAPH_OUTPUT.mkdir(parents=True, exist_ok=True)


# ============================================================
# SETTINGS
# ============================================================

ECG_RECORD = "16265"
ECG_CHANNEL = 0

DURATION = 3
SNR_LEVELS = [24, 18, 12, 6]


# ============================================================
# HEADER
# ============================================================

print("=" * 65)
print("                 EMG DATASET GENERATOR")
print("=" * 65)


# ============================================================
# READ CLEAN ECG
# ============================================================

print("\nReading clean ECG...")

ecg_signal, ecg_fields = wfdb.rdsamp(
    str(NSRDB_PATH / ECG_RECORD)
)

ecg_fs = ecg_fields["fs"]

clean_ecg = ecg_signal[
    :int(DURATION * ecg_fs),
    ECG_CHANNEL
]


# ============================================================
# READ NSTDB EMG
# ============================================================

print("Reading NSTDB muscle/EMG noise...")

emg_signal, emg_fields = wfdb.rdsamp(
    str(EMG_PATH)
)

emg_fs = emg_fields["fs"]

emg = emg_signal[:, 0]


# ============================================================
# DISPLAY INFORMATION
# ============================================================

print()
print(f"ECG sampling rate       : {ecg_fs} Hz")
print(f"EMG sampling rate       : {emg_fs} Hz")
print(f"ECG duration used       : {DURATION} seconds")


# ============================================================
# RESAMPLE EMG
# ============================================================

if emg_fs != ecg_fs:

    print("Resampling EMG to ECG sampling rate...")

    g = gcd(int(emg_fs), int(ecg_fs))

    up = int(ecg_fs // g)
    down = int(emg_fs // g)

    emg = resample_poly(
        emg,
        up,
        down
    )

    print(
        f"EMG samples after resampling: {len(emg)}"
    )

else:

    print("EMG already matches ECG sampling rate.")


# ============================================================
# MATCH LENGTH
# ============================================================

required_samples = len(clean_ecg)

if len(emg) < required_samples:
    raise ValueError(
        "EMG signal is shorter than the selected ECG segment."
    )

emg = emg[:required_samples]


# ============================================================
# TIME AXIS
# ============================================================

time = np.arange(
    len(clean_ecg)
) / ecg_fs


# ============================================================
# GENERATE EMG DATA
# ============================================================

print("\nGenerating EMG-contaminated ECG...")

for snr_db in SNR_LEVELS:

    # Signal powers
    ecg_power = np.mean(clean_ecg ** 2)
    emg_power = np.mean(emg ** 2)

    # Required EMG power for target SNR
    target_noise_power = (
        ecg_power / (10 ** (snr_db / 10))
    )

    # Scale EMG
    scale = np.sqrt(
        target_noise_power / emg_power
    )

    scaled_emg = emg * scale

    # Add EMG noise
    noisy_ecg = clean_ecg + scaled_emg


    # ========================================================
    # SAVE DATA
    # ========================================================

    data_file = (
        DATA_OUTPUT
        / f"16265_ECG1_EMG_{snr_db}dB.npz"
    )

    np.savez(
        data_file,
        time=time,
        clean_ecg=clean_ecg,
        emg_noise=scaled_emg,
        noisy_ecg=noisy_ecg,
        sampling_rate=ecg_fs,
        original_emg_sampling_rate=emg_fs,
        snr_db=snr_db,
        record=ECG_RECORD,
        channel=ECG_CHANNEL,
        noise_class="EMG"
    )


    # ========================================================
    # SAVE GRAPH
    # ========================================================

    graph_file = (
        GRAPH_OUTPUT
        / f"16265_ECG1_EMG_{snr_db}dB.png"
    )

    plt.figure(figsize=(14, 8))

    plt.subplot(3, 1, 1)
    plt.plot(time, clean_ecg)
    plt.title("Clean ECG — NSRDB 16265 ECG1")
    plt.ylabel("Amplitude")
    plt.grid(True, alpha=0.3)

    plt.subplot(3, 1, 2)
    plt.plot(time, scaled_emg)
    plt.title(
        f"EMG / Muscle Noise — {snr_db} dB"
    )
    plt.ylabel("Amplitude")
    plt.grid(True, alpha=0.3)

    plt.subplot(3, 1, 3)
    plt.plot(time, noisy_ecg)
    plt.title(
        f"ECG + EMG Noise — SNR {snr_db} dB"
    )
    plt.xlabel("Time (seconds)")
    plt.ylabel("Amplitude")
    plt.grid(True, alpha=0.3)

    plt.tight_layout()

    plt.savefig(
        graph_file,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(
        f"SNR {snr_db:>2} dB  →  "
        f"Data: {data_file.name}  |  "
        f"Graph: {graph_file.name}"
    )


# ============================================================
# FINAL
# ============================================================

print("\n" + "=" * 65)
print("EMG DATASET GENERATION COMPLETE")
print("=" * 65)

print(f"\nData saved to   : {DATA_OUTPUT}")
print(f"Graphs saved to : {GRAPH_OUTPUT}")