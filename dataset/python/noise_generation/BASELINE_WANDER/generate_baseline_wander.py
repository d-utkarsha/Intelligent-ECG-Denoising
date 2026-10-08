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

BW_PATH = NSTDB_PATH / "bw"

OUTPUT_PATH = (
    DATASET_PATH
    / "python"
    / "noise_generation"
    / "BASELINE_WANDER"
    / "output"
)

DATA_OUTPUT = OUTPUT_PATH / "data"
GRAPH_OUTPUT = OUTPUT_PATH / "graphs"

DATA_OUTPUT.mkdir(parents=True, exist_ok=True)
GRAPH_OUTPUT.mkdir(parents=True, exist_ok=True)


# ============================================================
# ECG SETTINGS
# ============================================================

ECG_RECORD = "16265"

# ECG1 = channel 0
ECG_CHANNEL = 0

# Initial dataset duration
DURATION = 3

# SNR levels
SNR_LEVELS = [24, 18, 12, 6]


# ============================================================
# READ CLEAN ECG
# ============================================================

print()
print("=" * 65)
print("          BASELINE WANDER DATASET GENERATOR")
print("=" * 65)

print()
print("Reading clean ECG...")

ecg_signal, ecg_fields = wfdb.rdsamp(
    str(NSRDB_PATH / ECG_RECORD)
)

ecg_fs = float(ecg_fields["fs"])

num_samples = int(
    DURATION * ecg_fs
)

ecg = ecg_signal[
    :num_samples,
    ECG_CHANNEL
]

time = np.arange(
    len(ecg)
) / ecg_fs


# ============================================================
# READ REAL NSTDB BASELINE-WANDER NOISE
# ============================================================

print("Reading NSTDB baseline-wander noise...")

bw_signal, bw_fields = wfdb.rdsamp(
    str(BW_PATH)
)

bw_fs = float(bw_fields["fs"])

# Use first channel of bw
bw = bw_signal[:, 0]


print()
print(f"ECG sampling rate       : {ecg_fs} Hz")
print(f"BW sampling rate        : {bw_fs} Hz")
print(f"ECG duration used       : {DURATION} seconds")


# ============================================================
# RESAMPLE BW TO ECG SAMPLING RATE
# ============================================================

if bw_fs != ecg_fs:

    print("Resampling BW to ECG sampling rate...")

    common = gcd(
        int(bw_fs),
        int(ecg_fs)
    )

    up = int(ecg_fs) // common
    down = int(bw_fs) // common

    bw_resampled = resample_poly(
        bw,
        up,
        down
    )

else:

    bw_resampled = bw


# ============================================================
# EXTRACT REQUIRED BW SEGMENT
# ============================================================

if len(bw_resampled) < len(ecg):

    raise ValueError(
        "NSTDB BW signal is shorter than the ECG segment."
    )

bw_resampled = bw_resampled[
    :len(ecg)
]


# ============================================================
# DISPLAY INFORMATION
# ============================================================

print(
    f"BW samples after resampling: "
    f"{len(bw_resampled)}"
)

print()
print("Generating baseline-wander versions...")
print()


# ============================================================
# GENERATE DIFFERENT SNR LEVELS
# ============================================================

for snr_db in SNR_LEVELS:

    # --------------------------------------------------------
    # Calculate ECG power
    # --------------------------------------------------------

    signal_power = np.mean(
        ecg ** 2
    )


    # --------------------------------------------------------
    # Calculate required BW noise power
    # --------------------------------------------------------

    noise_power = (
        signal_power
        / (10 ** (snr_db / 10))
    )


    # --------------------------------------------------------
    # Scale REAL NSTDB BW noise
    # --------------------------------------------------------

    bw_scaled = (
        bw_resampled
        * np.sqrt(
            noise_power
            / np.mean(bw_resampled ** 2)
        )
    )


    # --------------------------------------------------------
    # Create noisy ECG
    # --------------------------------------------------------

    noisy_ecg = ecg + bw_scaled


    # --------------------------------------------------------
    # SAVE DATA
    # --------------------------------------------------------

    data_filename = (
        f"{ECG_RECORD}_ECG1_BW_{snr_db}dB.npz"
    )

    data_path = (
        DATA_OUTPUT / data_filename
    )

    np.savez(
        data_path,

        time=time,

        clean_ecg=ecg,

        baseline_wander=bw_scaled,

        noisy_ecg=noisy_ecg,

        sampling_rate=ecg_fs,

        original_bw_sampling_rate=bw_fs,

        snr_db=snr_db,

        record=ECG_RECORD,

        channel="ECG1",

        noise_class="BASELINE_WANDER"
    )


    # ========================================================
    # CREATE GRAPH
    # ========================================================

    fig, axes = plt.subplots(
        3,
        1,
        figsize=(14, 9),
        sharex=True
    )


    # --------------------------------------------------------
    # Clean ECG
    # --------------------------------------------------------

    axes[0].plot(
        time,
        ecg
    )

    axes[0].set_title(
        f"{ECG_RECORD} ECG1 - Clean Reference"
    )

    axes[0].set_ylabel(
        "Amplitude (mV)"
    )

    axes[0].grid(True)


    # --------------------------------------------------------
    # Baseline Wander
    # --------------------------------------------------------

    axes[1].plot(
        time,
        bw_scaled
    )

    axes[1].set_title(
        f"NSTDB Real Baseline Wander - {snr_db} dB"
    )

    axes[1].set_ylabel(
        "Amplitude"
    )

    axes[1].grid(True)


    # --------------------------------------------------------
    # Noisy ECG
    # --------------------------------------------------------

    axes[2].plot(
        time,
        noisy_ecg
    )

    axes[2].set_title(
        f"{ECG_RECORD} ECG1 + Real NSTDB Baseline Wander "
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
    # Save graph
    # --------------------------------------------------------

    plt.tight_layout()

    graph_filename = (
        f"{ECG_RECORD}_ECG1_BW_{snr_db}dB.png"
    )

    graph_path = (
        GRAPH_OUTPUT / graph_filename
    )

    plt.savefig(
        graph_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()

    plt.close()


    # --------------------------------------------------------
    # Confirmation
    # --------------------------------------------------------

    print(
        f"✓ {snr_db} dB generated"
    )

    print(
        f"  Data  : {data_path}"
    )

    print(
        f"  Graph : {graph_path}"
    )

    print()


# ============================================================
# COMPLETE
# ============================================================

print("=" * 65)
print("BASELINE WANDER GENERATION COMPLETE")
print("=" * 65)

print()
print("Data saved to:")
print(DATA_OUTPUT)

print()
print("Graphs saved to:")
print(GRAPH_OUTPUT)

print()
print("Done!")