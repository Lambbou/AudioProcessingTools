import os
import csv
from glob import glob
from tqdm import tqdm

"""Audio sample selection and MOS (Mean Opinion Score) calculation utilities.

This module provides functionalities for:
1.  Calculating predicted MOS scores for audio files in a directory using the WVMOS model.
2.  Filtering and selecting samples from a CSV file based on specified criteria, such as
    a score column (e.g., MOS) and a cumulative duration threshold.

These tools are useful for curating datasets, particularly for speech and audio
machine learning tasks, by identifying high-quality samples or selecting a subset
of data that meets certain criteria.
"""
# Functions from audioprocessing.io_utils
from .io_utils import get_audio_duration, export_dict_to_csv

# WVMOS model for MOS calculation
from wvmos import get_wvmos # Ensure wvmos is listed in requirements.txt

# Initialize the WVMOS model globally.
# Currently, CUDA usage is hardcoded to True.
# TODO: Make CUDA usage configurable. This could be done via:
#   1. An environment variable checked at import time.
#   2. A global configuration function in the library (e.g., audioprocessing.config.set_cuda(True/False)).
#   3. Passing a 'device' or 'cuda' parameter down from CLI commands to functions like calculate_mos_for_directory,
#      which would then instantiate or reconfigure the model accordingly. This is the most flexible approach
#      but requires careful handling of the model instance.
# For now, if CUDA is not available, get_wvmos(cuda=True) might raise an error or fallback to CPU
# depending on its internal implementation. This should be tested.
mos_model = get_wvmos(cuda=True) 


def calculate_mos_for_directory(
    input_path: str, 
    output_file: str, 
    audio_type: str, 
    model_instance: any # Typically wvmos.WVMOS.WVMOS
    ) -> None:
    """Calculates MOS scores for audio files in a directory and saves results to a CSV.

    This function iterates through all audio files of the specified `audio_type`
    (case-insensitive) within the `input_path` directory (and its subdirectories).
    For each file, it calculates a Mean Opinion Score (MOS) using the provided
    `model_instance` (e.g., a WVMOS model). The results, including the file path,
    its duration in milliseconds, and the calculated MOS score, are written to
    a tab-separated CSV file specified by `output_file`. The CSV header is
    fixed as "Path", "Duration", "MOS".

    Args:
        input_path (str): The path to the source directory containing audio files.
        output_file (str): The path to the CSV file where results will be saved.
                           If the file exists, it will be overwritten.
        audio_type (str): The file extension (without the dot) of audio files
                          to process (e.g., "wav", "flac").
        model_instance (any): An instance of a MOS calculation model (e.g., WVMOS)
                              that has a `calculate_one(filepath: str) -> float` method.

    Returns:
        None. Results are written to `output_file`.

    Raises:
        FileNotFoundError: If `input_path` does not exist or is not a directory.
                           (This is checked explicitly, prior to `glob`).
        Exception: Catches and prints general exceptions during MOS calculation for
                   individual files, allowing the process to continue with other files.
                   The specific exceptions depend on the `model_instance` and
                   `audioprocessing.io_utils.get_audio_duration`.
    """
    # Validate input_path
    if not os.path.isdir(input_path):
        # Raising FileNotFoundError for consistency with other file/dir operations
        raise FileNotFoundError(f"Error: Input path '{input_path}' must be an existing directory.")

    # Prevent accidental overwrite by Click, but function itself should be able to overwrite if called directly.
    # The CLI command for this function should handle overwrite confirmation if necessary.
    # For library use, if output_file exists, it will be overwritten by export_dict_to_csv.
    # if os.path.isfile(output_file):
    #     print(f"Warning: Output file '{output_file}' already exists and will be overwritten.")

    results = {}
    # Glob pattern to find files of audio_type, case-insensitive, recursively
    glob_pattern = os.path.join(input_path, '**', f'*.{audio_type.lower()}')
    
    # Using sorted(glob(...)) for deterministic order, though not strictly necessary.
    file_paths = sorted(glob.glob(glob_pattern, recursive=True))

    if not file_paths:
        print(f"No '.{audio_type}' files found in '{input_path}'. No output file will be created.")
        return

    print(f"Calculating MOS scores for {len(file_paths)} '.{audio_type}' files found in '{input_path}'...")
    for path in tqdm(file_paths, desc="Calculating MOS"):
        try:
            # Get audio duration using the utility from io_utils
            duration_ms = get_audio_duration(path)
        except Exception as e:
            # Log error and skip if duration can't be obtained
            print(f"Error getting duration for {path}: {e}. Skipping MOS calculation for this file.")
            results[path] = ["Error: Duration unreadable", "Error: Not calculated"]
            continue
            
        try:
            # Calculate MOS score using the provided model instance
            mos_score = model_instance.calculate_one(path)
        except Exception as e:
            # Catch any exception from model_instance.calculate_one
            print(f"Error calculating MOS for {path}: {e}")
            mos_score = "Error: Calculation failed"
        
        results[path] = [duration_ms, mos_score]

    # Export the results to CSV
    # The export_dict_to_csv function uses a default header: ["Path", "Duration", "MOS"]
    # which matches the data structure prepared in `results`.
    print(f'\nGenerating output CSV file: {output_file}...')
    try:
        export_dict_to_csv(results, output_file) # Default header is ["Path", "Duration", "MOS"]
        print(f"MOS calculation and CSV export complete. Output saved to '{output_file}'.")
    except Exception as e:
        print(f"Error exporting results to CSV '{output_file}': {e}")


def filter_and_save_csv_by_score_and_duration(
    input_csv_path: str, 
    output_csv_path: str, 
    score_column_name: str = 'MOS', 
    duration_column_name: str = 'Duration',
    size_threshold_ms: int = 600000,
    sort_descending: bool = True
    ) -> None:
    """Filters rows from a CSV file based on a score and cumulative duration.

    This function reads an input CSV file (expected to be tab-delimited),
    sorts its rows by a specified `score_column_name` (descending by default),
    and selects rows such that their total duration (from `duration_column_name`,
    in milliseconds) does not exceed `size_threshold_ms`. If `size_threshold_ms`
    is 0, all rows are selected (after sorting). The selected rows are then
    written to a new tab-delimited CSV file specified by `output_csv_path`.

    Args:
        input_csv_path (str): Path to the input CSV file.
        output_csv_path (str): Path to save the filtered output CSV file.
        score_column_name (str): The name of the column in the CSV to use for
                                 sorting samples. Defaults to 'MOS'.
        duration_column_name (str): The name of the column in the CSV that
                                    contains duration values in milliseconds.
                                    Defaults to 'Duration'.
        size_threshold_ms (int): The maximum cumulative duration (in milliseconds)
                                 of selected samples. If 0, no duration limit is
                                 applied. Defaults to 600,000 ms (10 minutes).
        sort_descending (bool): If True, sorts by `score_column_name` in
                                descending order. If False, sorts in ascending
                                order. Defaults to True.

    Returns:
        None. Results are written to `output_csv_path`.

    Raises:
        FileNotFoundError: If `input_csv_path` does not exist.
        ValueError: If `score_column_name` or `duration_column_name` are not
                    found in the CSV header, or if duration values cannot be
                    parsed as floats.
        Exception: Catches and prints general exceptions during CSV processing or sorting.
    """
    selected_rows = []
    total_duration_ms = 0.0 # Use float for total_duration to handle potential float durations

    try:
        with open(input_csv_path, 'r', newline='', encoding='utf-8') as csvfile:
            # Assuming tab delimiter as per original script and common usage in this project
            reader = csv.DictReader(csvfile, delimiter="\t") 
            fieldnames = reader.fieldnames

            if not fieldnames:
                print(f"Warning: Input CSV '{input_csv_path}' is empty or has no header.")
                # Create an empty output file with header if possible, or just return
                if output_csv_path:
                     with open(output_csv_path, 'w', newline='', encoding='utf-8') as out_csvfile:
                        writer = csv.DictWriter(out_csvfile, fieldnames=[], delimiter="\t")
                        writer.writeheader()
                return

            # Validate essential column names
            if score_column_name not in fieldnames:
                raise ValueError(f"Error: Score column '{score_column_name}' not found in CSV header: {fieldnames}")
            if duration_column_name not in fieldnames:
                raise ValueError(f"Error: Duration column '{duration_column_name}' not found in CSV header: {fieldnames}")
            
            # Define a key function for sorting, robust to non-floatable score values
            def sort_key(row_dict):
                try:
                    return float(row_dict[score_column_name])
                except (ValueError, TypeError): # Catch if score is not a number (e.g. "Error", None)
                    # Place non-numeric scores at the "worst" end of the sort.
                    return float('-inf') if sort_descending else float('inf')

            try:
                # Read all rows and sort them. Sorting in memory is acceptable for moderately sized CSVs.
                # For very large CSVs, an on-disk sort or chunked processing might be needed.
                all_rows = list(reader)
                sorted_rows = sorted(all_rows, key=sort_key, reverse=sort_descending)
            except Exception as e: # Catch any error during initial read or sort
                print(f"Error reading or sorting CSV '{input_csv_path}': {e}")
                # Potentially re-raise or handle more gracefully
                raise

            # Iterate through sorted rows and select based on cumulative duration
            for row in sorted_rows:
                try:
                    current_duration = float(row[duration_column_name])
                except (ValueError, TypeError):
                    print(f"Warning: Could not parse duration '{row.get(duration_column_name)}' for row: {row}. Skipping this row.")
                    continue # Skip rows where duration cannot be parsed

                # If threshold is 0, it means no limit, so always add the row.
                if size_threshold_ms == 0 or (total_duration_ms + current_duration <= size_threshold_ms):
                    selected_rows.append(row)
                    total_duration_ms += current_duration
                # If threshold is met and not 0, stop adding rows (unless more can fit under strict <=)
                # This implicitly handles the case where the loop continues but no more rows are added.

    except FileNotFoundError:
        print(f"Error: Input CSV file not found at '{input_csv_path}'")
        raise # Re-raise for CLI to catch or for library user to handle
    except ValueError as e: # Catch ValueError from column name checks
        print(str(e))
        raise # Re-raise
    except Exception as e: # Catch other unexpected errors during file open or initial read
        print(f"An unexpected error occurred with input CSV '{input_csv_path}': {e}")
        raise # Re-raise

    # Write the selected rows to the output CSV
    try:
        with open(output_csv_path, 'w', newline='', encoding='utf-8') as out_csvfile:
            # Ensure fieldnames are available; if input was empty, fieldnames might be None or empty.
            # Use fieldnames obtained from DictReader, which should be correct.
            writer = csv.DictWriter(out_csvfile, fieldnames=fieldnames if fieldnames else [], delimiter="\t", quotechar='|') # Using original quotechar
            writer.writeheader()
            writer.writerows(selected_rows)
        
        print(f"Selected {len(selected_rows)} rows. Total duration: {total_duration_ms / 1000.0:.2f}s. Output saved to '{output_csv_path}'.")
    except Exception as e:
        print(f"Error writing output CSV to '{output_csv_path}': {e}")
        raise # Re-raise for CLI or library user
