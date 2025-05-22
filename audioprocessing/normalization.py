"""Audio normalization utilities.

This module provides tools for normalizing the loudness of audio files.
It includes a custom Click parameter type for ensuring negative number inputs
(typically for dBFS or LUFS values) and a function to process a directory
of WAV files, normalizing them to a target loudness level and saving them
to a new directory while maintaining the original structure.
"""
import os
import click
from pydub import AudioSegment
from pydub.exceptions import CouldntDecodeError # For documenting exceptions

class NegativeNumberParamType(click.ParamType):
    """A Click custom parameter type that accepts only negative numbers.

    This class is used to validate command-line arguments that are expected
    to be negative numerical values, such as loudness levels in dBFS or LUFS.

    Attributes:
        name (str): A human-readable name for this parameter type.
    """
    name = 'negative_numbers_only'

    def convert(self, value: str, param: click.Parameter | None, ctx: click.Context | None) -> float:
        """Converts the string `value` to a negative float.

        Args:
            value (str): The string value to convert, passed by Click from the command line.
            param (click.Parameter | None): The Click Parameter that is using this type.
                                            Can be None if not applicable.
            ctx (click.Context | None): The Click Context. Can be None if not applicable.

        Returns:
            float: The converted negative floating-point number.

        Raises:
            click.BadParameter: If `value` is not a valid number or if it's not negative.
        """
        try:
            number = float(value)
            if number < 0:
                return number
            else:
                self.fail(f'{value} is not a negative number. Values passed in dB LUFS cannot be positive.', param, ctx)
        except ValueError:
            self.fail(f'{value} is not a valid number. Values passed in dB LUFS should be, well... numerical values.', param, ctx)

def process_directory_for_normalization(src_dir: str, dst_dir: str, target_loudness: float) -> None:
    """Normalizes all WAV files in a source directory to a target loudness.

    This function recursively walks through the `src_dir`, finds all `.wav` files,
    normalizes them to the `target_loudness` (typically in LUFS), and saves
    the processed files to the `dst_dir`. The original directory structure from
    `src_dir` is replicated in `dst_dir`.

    Args:
        src_dir (str): The path to the source directory containing `.wav` files
                       to be normalized.
        dst_dir (str): The path to the destination directory where the normalized
                       `.wav` files will be saved. This directory will be created
                       if it does not exist.
        target_loudness (float): The target loudness level for the normalization,
                                 typically specified in LUFS (Loudness Units Full Scale)
                                 or dBFS. This value should be negative (e.g., -23.0).
    
    Returns:
        None

    Raises:
        FileNotFoundError: If `src_dir` does not exist (implicitly via `os.walk`).
        NotADirectoryError: If `src_dir` is not a directory (implicitly via `os.walk`).
        CouldntDecodeError: If a `.wav` file encountered during processing cannot be
                            decoded by pydub (e.g., due to corruption or unsupported
                            sub-format).
        OSError: If there are issues creating directories or writing files in `dst_dir`
                 (e.g., permission errors).
    """
    processed_files_count = 0
    failed_files_count = 0
    # os.walk can raise FileNotFoundError if src_dir does not exist,
    # or NotADirectoryError if src_dir is a file.
    try:
        for root, _, files in os.walk(src_dir):
            for file in files:
                if not file.lower().endswith(".wav"): # Process only .wav files, case-insensitive
                    continue
                
                src_file_path = os.path.join(root, file)
                # Create a relative path from the source base to maintain structure in the destination
                relative_path = os.path.relpath(src_file_path, src_dir)
                dst_file_path = os.path.join(dst_dir, relative_path)

                # Ensure the specific output directory for the current file exists
                try:
                    os.makedirs(os.path.dirname(dst_file_path), exist_ok=True)
                except OSError as e:
                    print(f"Error creating directory {os.path.dirname(dst_file_path)}: {e}. Skipping file {src_file_path}.")
                    failed_files_count += 1
                    continue
                
                try:
                    # Load the audio file. This can raise CouldntDecodeError.
                    audio = AudioSegment.from_file(src_file_path)
                    
                    # Get the current loudness of the audio file in dBFS.
                    # audio.dBFS represents the peak amplitude, but pydub's normalization
                    # via `audio + gain` effectively targets this dBFS value towards the new peak.
                    # For true LUFS normalization, a more sophisticated measurement (like pyloudnorm)
                    # and adjustment would be needed. Pydub's approach is simpler, based on peak dBFS.
                    current_loudness_dbfs = audio.dBFS 
                    
                    # Target loudness is also assumed to be in dBFS for this pydub method.
                    # If current_loudness_dbfs is -float('inf') (silence), gain calculation might be problematic.
                    # Also, if it's already at the target, no change is needed.
                    if current_loudness_dbfs is not None and current_loudness_dbfs != float('-inf') and current_loudness_dbfs != target_loudness:
                        # Calculate the gain needed to reach the target_loudness.
                        # If target_loudness is -23 dBFS and current_loudness_dbfs is -10 dBFS, gain is -13 dB.
                        # If target_loudness is -23 dBFS and current_loudness_dbfs is -30 dBFS, gain is +7 dB.
                        gain_to_apply = target_loudness - current_loudness_dbfs
                        normalized_audio = audio.apply_gain(gain_to_apply)
                    elif current_loudness_dbfs == target_loudness:
                        print(f"Info: File {src_file_path} is already at the target loudness of {target_loudness}dBFS.")
                        normalized_audio = audio # No change needed
                    else: 
                        print(f"Warning: Could not determine loudness or loudness is problematic for {src_file_path} (Loudness: {current_loudness_dbfs}). Skipping normalization for this file.")
                        normalized_audio = audio # Keep original if loudness is indeterminable

                    # Export the normalized (or original if skipped/failed) audio to the target directory.
                    # This can raise OSError if writing fails.
                    normalized_audio.export(dst_file_path, format="wav")
                    processed_files_count += 1

                except CouldntDecodeError as e:
                    print(f"Error decoding file {src_file_path}: {e}. Skipping.")
                    failed_files_count += 1
                except Exception as e: # Catch other potential errors during file processing
                    print(f"An unexpected error occurred processing file {src_file_path}: {e}. Skipping.")
                    failed_files_count += 1

        print(f"\nNormalization process finished for directory '{src_dir}'.")
        print(f"Successfully processed and saved: {processed_files_count} files.")
        if failed_files_count > 0:
            print(f"Failed to process/normalize: {failed_files_count} files. See warnings/errors above.")
        print(f"Output directory: '{dst_dir}'.")
        print("Note: Only '.wav' files were targeted for normalization.")

    except FileNotFoundError:
        print(f"Error: Source directory '{src_dir}' not found.")
        # Re-raise or handle as per library design for error propagation
        raise
    except NotADirectoryError:
        print(f"Error: Source path '{src_dir}' is not a directory.")
        # Re-raise or handle
        raise
    except Exception as e: # Catch-all for unexpected errors like os.walk issues
        print(f"An unexpected error occurred during directory traversal or setup: {str(e)}")
        # Re-raise or handle
        raise
