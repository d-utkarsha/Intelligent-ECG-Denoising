import wfdb
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.signal import resample_poly
from math import gcd
from itertools import combinations


# ============================================================
# PATHS
# ============================================================

DATASET_PATH = Path(
    r"C:\Users\dutka\OneDrive\Documents\DU\Intelligent-ECG-Denoising\dataset"
)

NSRDB_PATH = DATASET_PATH / "NSRDB"
NSTDB_PATH = DATASET_PATH / "NSTDB"

OUTPUT_PATH = (
    DATASET_PATH
    / "python"
    / "noise_generation"
    / "MIXED_NOISE"
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

PLI_FREQUENCY = 50


# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print("             MIXED NOISE DATASET GENERATOR")
print("=" * 70)


# ============================================================
# HELPER FUNCTION — RESAMPLING
# ============================================================

def resample_noise(noise, original_fs, target_fs):

    if original_fs == target_fs:
        return noise

    g = gcd(
        int(original_fs),
        int(target_fs)
    )

    up = int(target_fs // g)
    down = int(original_fs // g)

    return resample_poly(
        noise,
        up,
        down
    )


# ============================================================
# READ CLEAN ECG
# ============================================================

print("\nReading clean ECG...")

ecg_signal, ecg_fields = wfdb.rdsamp(
    str(NSRDB_PATH / ECG_RECORD)
)

ecg_fs = ecg_fields["fs"]

required_samples = int(
    DURATION * ecg_fs
)

clean_ecg = ecg_signal[
    :required_samples,
    ECG_CHANNEL
]

time = np.arange(
    required_samples
) / ecg_fs


# ============================================================
# READ REAL NOISE SOURCES
# ============================================================

print("Reading NSTDB noise sources...")

bw_signal, bw_fields = wfdb.rdsamp(
    str(NSTDB_PATH / "bw")
)

ma_signal, ma_fields = wfdb.rdsamp(
    str(NSTDB_PATH / "ma")
)

em_signal, em_fields = wfdb.rdsamp(
    str(NSTDB_PATH / "em")
)


# ============================================================
# EXTRACT NOISE CHANNELS
# ============================================================

bw = bw_signal[:, 0]
emg = ma_signal[:, 0]
motion = em_signal[:, 0]


# ============================================================
# SAMPLING INFORMATION
# ============================================================

print("\nSampling-rate information")
print("-" * 70)

print(f"ECG sampling rate       : {ecg_fs} Hz")
print(f"BW sampling rate        : {bw_fields['fs']} Hz")
print(f"EMG sampling rate       : {ma_fields['fs']} Hz")
print(f"Motion sampling rate    : {em_fields['fs']} Hz")
print(f"PLI frequency           : {PLI_FREQUENCY} Hz")


# ============================================================
# RESAMPLE REAL NOISE
# ============================================================

bw = resample_noise(
    bw,
    bw_fields["fs"],
    ecg_fs
)

emg = resample_noise(
    emg,
    ma_fields["fs"],
    ecg_fs
)

motion = resample_noise(
    motion,
    em_fields["fs"],
    ecg_fs
)


# ============================================================
# MATCH LENGTH
# ============================================================

if len(bw) < required_samples:
    raise ValueError("BW signal is too short.")

if len(emg) < required_samples:
    raise ValueError("EMG signal is too short.")

if len(motion) < required_samples:
    raise ValueError("Motion signal is too short.")

bw = bw[:required_samples]
emg = emg[:required_samples]
motion = motion[:required_samples]


# ============================================================
# GENERATE 50 Hz PLI
# ============================================================

pli = np.sin(
    2 * np.pi * PLI_FREQUENCY * time
)


# ============================================================
# STORE ALL NOISE SOURCES
# ============================================================

noise_sources = {
    "BW": bw,
    "PLI": pli,
    "EMG": emg,
    "MOTION": motion
}


# ============================================================
# GENERATE ALL COMBINATIONS
# ============================================================

noise_names = list(noise_sources.keys())

all_combinations = []

for r in range(2, len(noise_names) + 1):

    for combo in combinations(
        noise_names,
        r
    ):

        all_combinations.append(combo)


# ============================================================
# DISPLAY COMBINATIONS
# ============================================================

print("\nNoise combinations")
print("-" * 70)

two_noise = [
    c for c in all_combinations
    if len(c) == 2
]

three_noise = [
    c for c in all_combinations
    if len(c) == 3
]

four_noise = [
    c for c in all_combinations
    if len(c) == 4
]

print("\n2-NOISE COMBINATIONS:")
for combo in two_noise:
    print("  ✓ " + " + ".join(combo))

print("\n3-NOISE COMBINATIONS:")
for combo in three_noise:
    print("  ✓ " + " + ".join(combo))

print("\n4-NOISE COMBINATION:")
for combo in four_noise:
    print("  ✓ " + " + ".join(combo))


# ============================================================
# TOTAL DATASET COUNT
# ============================================================

total_combinations = len(all_combinations)

total_datasets = (
    total_combinations
    *
    len(SNR_LEVELS)
)

print("\n" + "-" * 70)

print(
    f"Total combinations     : {total_combinations}"
)

print(
    f"SNR levels             : {len(SNR_LEVELS)}"
)

print(
    f"Total datasets         : {total_datasets}"
)


# ============================================================
# GENERATE DATA
# ============================================================

print("\nGenerating mixed-noise ECG data...")
print("-" * 70)


for combo in all_combinations:

    # --------------------------------------------------------
    # Combine selected raw noise sources
    # --------------------------------------------------------

    combined_noise = np.zeros(
        required_samples
    )

    for noise_name in combo:

        combined_noise += (
            noise_sources[noise_name]
        )


    # Name for output files
    noise_name = "_".join(combo)


    # --------------------------------------------------------
    # Generate every SNR
    # --------------------------------------------------------

    for snr_db in SNR_LEVELS:

        # ECG power
        ecg_power = np.mean(
            clean_ecg ** 2
        )

        # Combined noise power
        noise_power = np.mean(
            combined_noise ** 2
        )

        # Target noise power
        target_noise_power = (
            ecg_power
            /
            (10 ** (snr_db / 10))
        )

        # Scaling factor
        scale = np.sqrt(
            target_noise_power
            /
            noise_power
        )

        # Scale TOTAL combined noise
        scaled_noise = (
            combined_noise
            *
            scale
        )

        # Create noisy ECG
        noisy_ecg = (
            clean_ecg
            +
            scaled_noise
        )


        # ----------------------------------------------------
        # VERIFY SNR
        # ----------------------------------------------------

        actual_snr = 10 * np.log10(
            np.mean(clean_ecg ** 2)
            /
            np.mean(scaled_noise ** 2)
        )


        # ----------------------------------------------------
        # SAVE DATA
        # ----------------------------------------------------

        data_file = (
            DATA_OUTPUT
            /
            f"16265_ECG1_MIXED_{noise_name}_{snr_db}dB.npz"
        )

        np.savez(
            data_file,

            time=time,

            clean_ecg=clean_ecg,

            mixed_noise=scaled_noise,

            noisy_ecg=noisy_ecg,

            sampling_rate=ecg_fs,

            snr_db=snr_db,

            record=ECG_RECORD,

            channel=ECG_CHANNEL,

            noise_class="MIXED",

            noise_count=len(combo),

            noise_types=",".join(combo)
        )


        # ----------------------------------------------------
        # SAVE GRAPH
        # ----------------------------------------------------

        graph_file = (
            GRAPH_OUTPUT
            /
            f"16265_ECG1_MIXED_{noise_name}_{snr_db}dB.png"
        )

        plt.figure(
            figsize=(14, 8)
        )


        # Clean ECG
        plt.subplot(3, 1, 1)

        plt.plot(
            time,
            clean_ecg
        )

        plt.title(
            "Clean ECG — NSRDB 16265 ECG1"
        )

        plt.ylabel("Amplitude")
        plt.grid(True, alpha=0.3)


        # Mixed noise
        plt.subplot(3, 1, 2)

        plt.plot(
            time,
            scaled_noise
        )

        plt.title(
            f"Mixed Noise: {' + '.join(combo)} — "
            f"{snr_db} dB"
        )

        plt.ylabel("Amplitude")
        plt.grid(True, alpha=0.3)


        # Noisy ECG
        plt.subplot(3, 1, 3)

        plt.plot(
            time,
            noisy_ecg
        )

        plt.title(
            f"ECG + {' + '.join(combo)} — "
            f"SNR {snr_db} dB"
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


        # ----------------------------------------------------
        # TERMINAL OUTPUT
        # ----------------------------------------------------

        print(
            f"{noise_name:<25} | "
            f"SNR {snr_db:>2} dB | "
            f"Actual {actual_snr:>5.2f} dB | "
            f"Saved"
        )


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("             MIXED NOISE GENERATION COMPLETE")
print("=" * 70)

print(f"\nECG record          : {ECG_RECORD}")
print(f"Channel             : ECG1")
print(f"Sampling rate       : {ecg_fs} Hz")
print(f"Duration             : {DURATION} seconds")

print("\nCombination summary:")
print(f"  2-noise           : {len(two_noise)}")
print(f"  3-noise           : {len(three_noise)}")
print(f"  4-noise           : {len(four_noise)}")

print(f"\nTotal combinations  : {total_combinations}")
print(f"SNR levels          : {len(SNR_LEVELS)}")
print(f"Total datasets      : {total_datasets}")

print(f"\nData saved to       : {DATA_OUTPUT}")
print(f"Graphs saved to     : {GRAPH_OUTPUT}")

print("\n" + "=" * 70)
print("                       ALL DONE ✓")
print("=" * 70)