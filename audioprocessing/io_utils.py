import csv
from pydub import AudioSegment

def get_audio_duration(path:str) -> int:
    # returns the duration of an audio file in ms
    audio = AudioSegment.from_file(path)
    return len(audio) # returns ms - use audio.duration_seconds to get the duration in seconds

def export_dict_to_csv(dictionary, csv_path):
    # Exports a dictionary to a CSV file.
    # The current header is specific to similarity/MOS tasks.
    # Consider making headers an argument for more generic use.
    with open(csv_path, 'w') as ofile:
        writer = csv.writer(ofile, delimiter='\t')
        
        # Header
        writer.writerow(["Path", "Duration", "MOS"]) # Or similarity score
        
        for key, value in dictionary.items():
            writer.writerow([key, value[0], value[1]])

import os
import shutil

def _ensure_dir_exists(directory_path: str):
    """Helper function to ensure a directory exists, creating it if necessary."""
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
    elif not os.path.isdir(directory_path):
        raise NotADirectoryError(f"Path '{directory_path}' exists but is not a directory.")


def copy_files_from_csv_column(
    csv_filepath: str, 
    file_path_column_name: str, 
    destination_dir: str, 
    csv_delimiter: str = '\t', 
    csv_quotechar: str = '|'
    ):
    """
    Reads a CSV file, extracts file paths from a specified column, 
    and copies these files to a destination directory.
    The base name of the source file is used as the name of the copied file in the destination directory.

    Args:
        csv_filepath: Path to the CSV file.
        file_path_column_name: Name of the column containing the file paths.
        destination_dir: Directory where files will be copied.
        csv_delimiter: Delimiter used in the CSV file.
        csv_quotechar: Quote character used in the CSV file.
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
    """Helper to find column index or raise error."""
    try:
        return header.index(column_name)
    except ValueError:
        raise ValueError(f"Error: Key column '{column_name}' not found in header of '{csv_path}'. Header: {header}")


def join_csv_files_on_key(
    csv1_path: str, 
    csv2_path: str, 
    key_column: str, 
    output_csv_path: str, 
    csv_delimiter: str = '\t', 
    csv_quotechar: str = '|'
    ):
    """
    Merges two CSV files based on a common key column (SQL-like JOIN).
    The header of the output CSV will be header_csv1 + header_csv2 (excluding the key column from csv2).
    Rows are joined where the key_column values match.

    Args:
        csv1_path: Path to the first CSV file.
        csv2_path: Path to the second CSV file.
        key_column: The name of the common column to join on.
        output_csv_path: Path to save the merged CSV file.
        csv_delimiter: Delimiter for input and output CSVs.
        csv_quotechar: Quote character for input and output CSVs.
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