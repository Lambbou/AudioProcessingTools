import os
import csv
from glob import glob
from tqdm import tqdm

# Functions from audioprocessing.io_utils
from .io_utils import get_audio_duration, export_dict_to_csv

# WVMOS model for MOS calculation
from wvmos import get_wvmos # Make sure wvmos is in requirements.txt
# TODO: Make CUDA usage configurable, e.g., via a CLI option or environment variable
mos_model = get_wvmos(cuda=True) 

"""
Calculates MOS scores for audio files in a directory and saves results to a CSV.
(Refactored from process_csv in utils/compute_mos.py)
"""
def calculate_mos_for_directory(input_path: str, output_file: str, audio_type: str, model_instance):
    # Check if the input is an existing path
    if not os.path.isdir(input_path):
        print('Error: Input path must be an existing directory.')
        # Consider raising an exception for CLI to catch and report
        return

    # Check if the output file already exists
    if os.path.isfile(output_file):
        print('Error: Output file already exists.')
        # Consider raising an exception
        return

    results = {}
    for path in tqdm(sorted(glob(os.path.join(input_path, '**/*.' + audio_type), recursive=True))):
        duration = get_audio_duration(path) # Using imported function
        try:
            mos_score = model_instance.calculate_one(path)
        except Exception as e:
            print(f"Error calculating MOS for {path}: {e}")
            mos_score = "Error"
        results[path] = [duration, mos_score]

    # Perform operations on the CSV file and generate output
    print(f'Generating output file: {output_file}...\r')
    # Using imported function. Header will be ["Path", "Duration", "MOS"]
    export_dict_to_csv(results, output_file)
    print("MOS calculation done.")

"""
Filters rows from an input CSV based on a score and cumulative duration, then saves to a new CSV.
(Refactored from filter_and_save_csv in utils/select_best_samples.py)
"""
def filter_and_save_csv_by_score_and_duration(
    input_csv_path: str, 
    output_csv_path: str, 
    score_column_name: str = 'MOS', 
    duration_column_name: str = 'Duration',
    size_threshold_ms: int = 600000,
    sort_descending: bool = True
    ):
    
    selected_rows = []
    total_duration_ms = 0

    with open(input_csv_path, 'r') as csvfile:
        # Assuming tab delimiter as per original script
        reader = csv.DictReader(csvfile, delimiter="\t") 
        fieldnames = reader.fieldnames

        if not fieldnames:
            print(f"Warning: Input CSV '{input_csv_path}' is empty or has no header.")
            # Create an empty output file with header if possible, or just return
            if output_csv_path:
                 with open(output_csv_path, 'w', newline='') as out_csvfile:
                    # Try to use original fieldnames if available, else a default or error
                    writer = csv.DictWriter(out_csvfile, fieldnames=fieldnames if fieldnames else [], delimiter="\t")
                    writer.writeheader()
            return

        if score_column_name not in fieldnames:
            print(f"Error: Score column '{score_column_name}' not found in CSV header: {fieldnames}")
            return
        if duration_column_name not in fieldnames:
            print(f"Error: Duration column '{duration_column_name}' not found in CSV header: {fieldnames}")
            return
            
        try:
            # Ensure score and duration can be converted to float
            def row_filter_key(row):
                try:
                    return float(row[score_column_name])
                except ValueError:
                    # Handle cases where score might not be a float (e.g., "Error")
                    # Place these at the bottom if sorting descending, top if ascending
                    return float('-inf') if sort_descending else float('inf')

            sorted_rows = sorted(reader, key=row_filter_key, reverse=sort_descending)
        except Exception as e:
            print(f"Error sorting CSV: {e}")
            return

        for row in sorted_rows:
            try:
                current_duration = float(row[duration_column_name])
            except ValueError:
                print(f"Warning: Could not parse duration '{row[duration_column_name]}' for row: {row}. Skipping.")
                continue

            if total_duration_ms + current_duration <= size_threshold_ms:
                selected_rows.append(row)
                total_duration_ms += current_duration
            elif size_threshold_ms == 0: # Special case: if threshold is 0, select all
                 selected_rows.append(row)
                 total_duration_ms += current_duration


    with open(output_csv_path, 'w', newline='') as out_csvfile:
        writer = csv.DictWriter(out_csvfile, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(selected_rows)
    
    print(f"Selected rows saved to '{output_csv_path}'. Total duration: {total_duration_ms / 1000.0:.2f}s.")
