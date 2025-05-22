"""Input/output utility functions for the audioprocessing library.

This module provides common helper functions used across the audioprocessing
library, particularly for tasks related to file system operations (like ensuring
directory existence, copying files based on CSV data) and CSV file manipulation
(like exporting dictionaries to CSV, joining CSV files). It also includes
audio-specific I/O, such as getting audio duration.
"""
import csv
from pydub import AudioSegment
from pydub.exceptions import CouldntDecodeError # For documenting exceptions

def get_audio_duration(path: str) -> int:
    """Gets the duration of an audio file in milliseconds.

    Args:
        path (str): The path to the audio file.

    Returns:
        int: The duration of the audio file in milliseconds.

    Raises:
        FileNotFoundError: If the audio file specified by `path` does not exist
                           (raised by `AudioSegment.from_file`).
        CouldntDecodeError: If the audio file cannot be decoded by pydub
                            (e.g., unsupported format, corrupted file).
    """
    # pydub's from_file can raise FileNotFoundError or CouldntDecodeError
    audio = AudioSegment.from_file(path)
    return len(audio) # len(AudioSegment) returns duration in milliseconds.

def export_dict_to_csv(dictionary: dict, csv_path: str, headers: list[str] = None):
    """Exports a dictionary to a CSV file with a specific header.

    The dictionary is expected to have keys representing the first column's data
    (e.g., file paths), and values as lists or tuples containing the data for
    the subsequent columns. The number of elements in these lists/tuples should
    match the number of headers provided (excluding the first implicit header 
    if `headers` corresponds only to the value part).

    Example:
        results = {"file1.wav": [6000, 4.5], "file2.wav": [5000, 4.2]}
        export_dict_to_csv(results, "output.csv", headers=["SourcePath", "DurationMS", "MOSScore"])
        This would create a CSV with "SourcePath" as the first column header.
        Alternatively, if headers are for the values only:
        export_dict_to_csv(results, "output.csv", headers=["DurationMS", "MOSScore"])
        The function will prepend a default "Key" or "Path" header for the keys.
        The original script used a fixed header: ["Path", "Duration", "MOS"]. This
        version is more flexible. If `headers` is None, it defaults to the original
        fixed header.

    Args:
        dictionary (dict): The dictionary to export. Keys are first column items,
                           values are lists/tuples for other columns.
        csv_path (str): The path to the CSV file to create.
        headers (list[str], optional): A list of strings for the CSV header.
            If None, defaults to `["Path", "Duration", "MOS"]`. The first header
            is for the dictionary keys.
    """
    if headers is None:
        # Default to the original fixed header if none provided
        final_headers = ["Path", "Duration", "MOS"]
    else:
        # If headers are provided, assume the first one is for the keys,
        # or if it's meant for values, ensure key header is present.
        # This logic can be adjusted based on how `headers` is intended to be used.
        # For now, let's assume `headers` is the full list including the key's header.
        final_headers = headers

    with open(csv_path, 'w', newline='', encoding='utf-8') as ofile:
        writer = csv.writer(ofile, delimiter='\t', quotechar='|') # Original delimiter/quotechar
        
        writer.writerow(final_headers)
        
        for key, value_list in dictionary.items():
            # Ensure value_list is indeed a list or tuple to be unpacked
            if not isinstance(value_list, (list, tuple)):
                # If it's a single value, wrap it in a list
                # This might happen if dict stores {key: score} instead of {key: [val1, val2]}
                value_list = [value_list]
            
            # The number of values in value_list should match final_headers length - 1
            if len(value_list) != len(final_headers) -1:
                print(f"Warning: Data for key '{key}' (values: {value_list}) "
                      f"does not match header count (expected {len(final_headers)-1} values). "
                      f"Row will be written as is, possibly misaligned.")
            writer.writerow([key] + list(value_list))

import os
import shutil

def _ensure_dir_exists(directory_path: str) -> None:
    """Ensures a directory exists at the specified path.

    If the directory does not exist, it is created. If a file exists at the path,
    or if the path exists but is not a directory, an error is raised.

    Args:
        directory_path (str): The path to the directory to check/create.

    Raises:
        NotADirectoryError: If `directory_path` exists but is not a directory.
        OSError: If `os.makedirs` fails for reasons other than the path existing
                 (e.g., permission issues).
    """
    if not os.path.exists(directory_path):
        os.makedirs(directory_path) # Can raise OSError for permission issues etc.
    elif not os.path.isdir(directory_path):
        # This specific check helps differentiate between non-existence and wrong type.
        raise NotADirectoryError(f"Path '{directory_path}' exists but is not a directory.")
    # If it exists and is a directory, do nothing.

def copy_files_from_csv_column(
    csv_filepath: str, 
    file_path_column_name: str, 
    destination_dir: str, 
    csv_delimiter: str = '\t', 
    csv_quotechar: str = '|'
    ) -> bool:
    """Copies files listed in a CSV column to a destination directory.

    Reads a CSV file, extracts file paths from a specified column, and copies
    these files to a destination directory. The base name of the source file is
    used as the name of the copied file in the destination directory. 
    Prints errors for files not found or if copying fails.

    Args:
        csv_filepath (str): Path to the CSV file.
        file_path_column_name (str): Name of the column in the CSV that contains
            the file paths to be copied.
        destination_dir (str): The directory where files will be copied.
        csv_delimiter (str): The delimiter used in the CSV file. Defaults to tab.
        csv_quotechar (str): The quote character used in the CSV file.
            Defaults to '|'.

    Returns:
        bool: True if the CSV was processed (even if some files failed to copy),
              False if the CSV file itself could not be read or the specified
              `file_path_column_name` does not exist in the CSV header.
    """
    copied_count = 0
    skipped_count = 0
    
    _ensure_dir_exists(destination_dir)

    try:
        with open(csv_filepath, 'r', newline='', encoding='utf-8') as file:
            reader = csv.DictReader(file, delimiter=csv_delimiter, quotechar=csv_quotechar)
            
            if file_path_column_name not in reader.fieldnames:
                print(f"Error: Column '{file_path_column_name}' not found in CSV header: {reader.fieldnames}")
                return False # Indicate failure

            for row_number, row in enumerate(reader, start=1): # start=1 for header
                source_file_path = row.get(file_path_column_name)
                if not source_file_path: # Handles empty string, None
                    print(f"Warning: Row {row_number+1}: Empty path found in column '{file_path_column_name}'. Skipping.")
                    skipped_count += 1
                    continue

                if os.path.exists(source_file_path):
                    if not os.path.isfile(source_file_path):
                        print(f"Warning: Row {row_number+1}: Path '{source_file_path}' is not a file. Skipping.")
                        skipped_count += 1
                        continue
                    try:
                        destination_file_path = os.path.join(destination_dir, os.path.basename(source_file_path))
                        shutil.copy(source_file_path, destination_file_path)
                        copied_count += 1
                    except Exception as e:
                        print(f"Error copying file '{source_file_path}' from row {row_number+1}: {e}")
                        skipped_count += 1
                else:
                    print(f"Warning: Row {row_number+1}: Source file not found: '{source_file_path}'. Skipping.")
                    skipped_count += 1
        
        print(f"File copying process complete. Copied {copied_count} files. Skipped or failed for {skipped_count} entries.")
        return True # Indicate success or partial success

    except FileNotFoundError:
        print(f"Error: CSV file not found at '{csv_filepath}'")
        return False
    except Exception as e:
        print(f"An unexpected error occurred while reading CSV '{csv_filepath}': {str(e)}")
        return False


def _get_column_index(header: list[str], column_name: str, csv_path: str) -> int:
    """Gets the index of a column name within a CSV header.

    Internal helper function.

    Args:
        header (list[str]): The list of strings representing the CSV header.
        column_name (str): The name of the column to find.
        csv_path (str): The path to the CSV file (used for error reporting).

    Returns:
        int: The index of `column_name` in the `header`.

    Raises:
        ValueError: If `column_name` is not found in the `header`.
    """
    try:
        return header.index(column_name)
    except ValueError:
        # Raise a new ValueError with a more informative message
        raise ValueError(f"Error: Column '{column_name}' not found in header of '{csv_path}'. Header: {header}")


def join_csv_files_on_key(
    csv1_path: str, 
    csv2_path: str, 
    key_column: str, 
    output_csv_path: str, 
    csv_delimiter: str = '\t', 
    csv_quotechar: str = '|'
    ) -> bool:
    """Joins two CSV files based on a common key column (SQL-like inner join).

    This function reads two CSV files, identifies a common `key_column` in each,
    and creates a new CSV file containing rows where the values in the `key_column`
    match. The output CSV's header will be the header of the first CSV file
    followed by the header of the second CSV file (excluding the `key_column` from
    the second file to avoid duplication). Only rows with matching keys in both
    files are included in the output.

    Args:
        csv1_path (str): Path to the first CSV file (e.g., "left table").
        csv2_path (str): Path to the second CSV file (e.g., "right table").
        key_column (str): The name of the common column to join on. This column
                          must exist in both CSV files.
        output_csv_path (str): Path where the joined CSV data will be saved.
        csv_delimiter (str): The delimiter used in both input and output CSV files.
                             Defaults to tab.
        csv_quotechar (str): The quote character used in both input and output CSV
                             files. Defaults to '|'.

    Returns:
        bool: True if the join operation was successful (or partially successful,
              meaning files were read and output was written, even if no rows
              matched). False if a critical error occurred, such as a file not
              being found or the key column being absent from a header.
    """
    try:
        with open(csv1_path, "r", newline='', encoding='utf-8') as f1:
            reader1 = csv.reader(f1, delimiter=csv_delimiter, quotechar=csv_quotechar)
            header1 = next(reader1)
            data1 = list(reader1)
            key1_idx = _get_column_index(header1, key_column, csv1_path)

        with open(csv2_path, "r", newline='', encoding='utf-8') as f2:
            reader2 = csv.reader(f2, delimiter=csv_delimiter, quotechar=csv_quotechar)
            header2 = next(reader2)
            data2 = list(reader2)
            key2_idx = _get_column_index(header2, key_column, csv2_path)

        # Create output header: header1 + header2 (excluding key from header2)
        output_header = list(header1) # Make a mutable copy
        for i, h_col in enumerate(header2):
            if i != key2_idx:
                output_header.append(h_col)
        
        # Build a dictionary from the second CSV for faster lookups
        data2_dict = {}
        for row2 in data2:
            key_val = row2[key2_idx]
            # Handle multiple rows in csv2 with the same key if necessary (e.g., store a list of rows)
            # For this implementation, last one wins, or store first one found.
            if key_val not in data2_dict: # Store first one found
                 data2_dict[key_val] = [r for i, r in enumerate(row2) if i != key2_idx]


        resulting_data = [output_header]
        matches_found = 0
        
        for row1 in data1:
            key_val_csv1 = row1[key1_idx]
            if key_val_csv1 in data2_dict:
                matches_found +=1
                merged_row = list(row1) # Make a mutable copy
                merged_row.extend(data2_dict[key_val_csv1])
                resulting_data.append(merged_row)
            # else: row from csv1 has no match in csv2, so it's skipped (inner join behavior)

        if matches_found == 0:
            print(f"Warning: No matching rows found between '{csv1_path}' and '{csv2_path}' on key column '{key_column}'. Output file will only contain headers.")
        elif matches_found < len(data1) or matches_found < len(data2_dict): # Using len(data2_dict) as it stores unique keys from csv2
             print(f"Info: Found {matches_found} matching rows. CSV1 had {len(data1)} data rows. CSV2 had {len(data2_dict)} unique keys.")


        with open(output_csv_path, "w", newline='', encoding='utf-8') as f_out:
            writer = csv.writer(f_out, delimiter=csv_delimiter, quotechar=csv_quotechar, quoting=csv.QUOTE_MINIMAL) # QUOTE_ALL if original has it
            writer.writerows(resulting_data)
        
        print(f"CSV join complete. Output saved to '{output_csv_path}'. Found {matches_found} joined rows.")
        return True

    except FileNotFoundError as e:
        print(f"Error: File not found - {e.filename}")
        return False
    except ValueError as e: # Handles _get_column_index errors
        print(str(e))
        return False
    except Exception as e:
        print(f"An unexpected error occurred during CSV join: {str(e)}")
        return False