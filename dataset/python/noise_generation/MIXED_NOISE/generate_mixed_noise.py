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
# HELPER — RESAMPLE NOISE
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
# RESAMPLE
# ============================================================

print("\nSampling-rate information")
print("-" * 70)

print(f"ECG sampling rate       : {ecg_fs} Hz")
print(f"BW sampling rate        : {bw_fields['fs']} Hz")
print(f"EMG sampling rate       : {ma_fields['fs']} Hz")
print(f"Motion sampling rate    : {em_fields['fs']} Hz")
print(f"PLI frequency           : {PLI_FREQUENCY} Hz")


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

bw = bw[:required_samples]
emg = emg[:required_samples]
motion = motion[:required_samples]


# ============================================================
# GENERATE SYNTHETIC PLI
# ============================================================

pli = np.sin(
    2 * np.pi * PLI_FREQUENCY * time
)


# ============================================================
# STORE AVAILABLE NOISES
# ============================================================

noise_sources = {
    "BW": bw,
    "PLI": pli,
    "EMG": emg,
    "MOTION": motion
}


# ============================================================
# MIXED-NOISE COMBINATIONS
# ============================================================

combinations = [
    ("BW", "PLI"),
    ("BW", "EMG"),
    ("BW", "MOTION"),
    ("PLI", "EMG"),
    ("PLI", "MOTION"),
    ("EMG", "MOTION")
]


print("\nMixed-noise combinations")
print("-" * 70)

for noise_a, noise_b in combinations:
    print(f"  {noise_a} + {noise_b}")


# ============================================================
# GENERATE MIXED DATA
# ============================================================

print("\nGenerating mixed-noise ECG data...")
print("-" * 70)

for noise_a, noise_b in combinations:

    # --------------------------------------------------------
    # Combine the two raw noise sources
    # --------------------------------------------------------

    combined_noise = (
        noise_sources[noise_a]
        +
        noise_sources[noise_b]
    )

    noise_name = f"{noise_a}_{noise_b}"


    for snr_db in SNR_LEVELS:

        # ----------------------------------------------------
        # Calculate signal and noise power
        # ----------------------------------------------------

        ecg_power = np.mean(
            clean_ecg ** 2
        )

        noise_power = np.mean(
            combined_noise ** 2
        )


        # ----------------------------------------------------
        # Scale TOTAL mixed noise to target SNR
        # ----------------------------------------------------

        target_noise_power = (
            ecg_power
            /
            (10 ** (snr_db / 10))
        )

        scale = np.sqrt(
            target_noise_power
            /
            noise_power
        )

        scaled_noise = (
            combined_noise * scale
        )


        # ----------------------------------------------------
        # Create noisy ECG
        # ----------------------------------------------------

        noisy_ecg = (
            clean_ecg
            +
            scaled_noise
        )


        # ----------------------------------------------------
        # Verify actual SNR
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

            noise_1=noise_a,

            noise_2=noise_b
        )


        # ----------------------------------------------------
        # SAVE GRAPH
        # ----------------------------------------------------

        graph_file = (
            GRAPH_OUTPUT
            /
            f"16265_ECG1_MIXED_{noise_name}_{snr_db}dB.png"
        )

        plt.figure(figsize=(14, 8))


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


        plt.subplot(3, 1, 2)

        plt.plot(
            time,
            scaled_noise
        )

        plt.title(
            f"Mixed Noise: {noise_a} + {noise_b} — "
            f"{snr_db} dB"
        )

        plt.ylabel("Amplitude")
        plt.grid(True, alpha=0.3)


        plt.subplot(3, 1, 3)

        plt.plot(
            time,
            noisy_ecg
        )

        plt.title(
            f"ECG + {noise_a} + {noise_b} — "
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
            f"{noise_name:<15} | "
            f"SNR {snr_db:>2} dB | "
            f"Actual {actual_snr:>5.2f} dB | "
            f"Saved"
        )


# ============================================================
# FINAL SUMMARY
# ============================================================

total_files = (
    len(combinations)
    *
    len(SNR_LEVELS)
)

print("\n" + "=" * 70)
print("             MIXED NOISE GENERATION COMPLETE")
print("=" * 70)

print(f"\nECG record          : {ECG_RECORD}")
print(f"Channel             : ECG1")
print(f"Sampling rate       : {ecg_fs} Hz")
print(f"Duration             : {DURATION} seconds")

print("\nNoise combinations:")
for noise_a, noise_b in combinations:
    print(f"  ✓ {noise_a} + {noise_b}")

print("\nSNR levels:")
for snr in SNR_LEVELS:
    print(f"  ✓ {snr} dB")

print(f"\nTotal datasets      : {total_files}")

print(f"\nData saved to       : {DATA_OUTPUT}")
print(f"Graphs saved to     : {GRAPH_OUTPUT}")

print("\n" + "=" * 70)
print("                       ALL DONE ✓")
print("=" * 70)