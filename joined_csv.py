#make the joined.csv directory for input into Multiple_{}.py scripts

import os
import pandas as pd

# === USER PARAMETERS ===
home_dir = "/Volumes/holtl02lab/holtl02labspace/Holt_Lab_Members/Nora_Holt/Histone_Experiments/Further_GEMs/230808_Uninfected_HSV1"  # Change this to your experiment folder

# === DIRECTORIES ===
results_dir = os.path.join(home_dir, "Results")
plot_dir = os.path.join(home_dir, "Further_Analyses")
os.makedirs(plot_dir, exist_ok=True)

# === READ INPUT FILES ===
all_data_path = os.path.join(results_dir, "all_data.txt")
input_files_path = os.path.join(results_dir, "input_files.csv")

all_data_df = pd.read_csv(all_data_path, sep="\t")
input_files_df = pd.read_csv(input_files_path)
input_files_df.columns = input_files_df.columns.astype(str)

# === CREATE UNIQUE IDENTIFIERS ===
input_files_df["filename_roi"] = input_files_df["file name"] + "-" + input_files_df["roi"].astype(str)
input_files_df.set_index("filename_roi", inplace=True)

all_data_df["filename_roi"] = all_data_df["file name"] + "-" + all_data_df["roi"].astype(str)

# === JOIN DATA ===
joined_df = all_data_df[["filename_roi", "Trajectory"]].join(
    input_files_df, on=["filename_roi"], how="right"
)

# === SAVE OUTPUT ===
joined_csv_path = os.path.join(results_dir, "joined.csv")
joined_df.to_csv(joined_csv_path, index=False)
print(f"joined.csv created at: {joined_csv_path}")