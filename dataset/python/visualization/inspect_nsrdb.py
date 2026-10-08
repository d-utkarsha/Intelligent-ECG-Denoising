import wfdb
import pandas as pd
from pathlib import Path

# --------------------------------------------------
# PATHS
# --------------------------------------------------

NSRDB_PATH = Path(
    r"C:\Users\dutka\OneDrive\Documents\DU\Intelligent-ECG-Denoising\dataset\NSRDB"
)

OUTPUT_PATH = Path(
    r"C:\Users\dutka\OneDrive\Documents\DU\Intelligent-ECG-Denoising\dataset\processed"
)

OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------
# RECORD
# --------------------------------------------------

record = "16265"

record_path = NSRDB_PATH / record

# --------------------------------------------------
# READ ECG
# --------------------------------------------------

signal, fields = wfdb.rdsamp(str(record_path))

fs = fields["fs"]

# ECG1 = channel 0
ecg1 = signal[:, 0]

# --------------------------------------------------
# CREATE TIME AXIS
# --------------------------------------------------

time = [i / fs for i in range(len(ecg1))]

# --------------------------------------------------
# CREATE DATAFRAME
# --------------------------------------------------

df = pd.DataFrame({
    "time_seconds": time,
    "ecg1_mV": ecg1
})

# --------------------------------------------------
# SAVE
# --------------------------------------------------

output_file = OUTPUT_PATH / "16265_ECG1_clean.csv"

df.to_csv(output_file, index=False)

# --------------------------------------------------
# INFORMATION
# --------------------------------------------------

print("----------------------------------------")
print("ECG EXTRACTION COMPLETE")
print("----------------------------------------")
print("Record          :", record)
print("Channel         : ECG1")
print("Sampling Rate   :", fs, "Hz")
print("Samples         :", len(ecg1))
print("Duration        :", len(ecg1) / fs, "seconds")
print("Output file     :", output_file)
print("----------------------------------------")