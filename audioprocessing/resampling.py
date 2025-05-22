"""Audio resampling utilities using pydub.

This module provides functions for changing the sampling rate of audio files.
It supports resampling individual files or entire directories of audio files,
either to a new output directory or in-place. The core resampling capability
is provided by the pydub library.
"""
import os
import glob
from pydub import AudioSegment
from pydub.exceptions import CouldntDecodeError # For documenting exceptions
from tqdm import tqdm

def _ensure_dir(directory_path: str) -> None:
    """Ensure the directory exists, creating it if necessary.
    
    Internal helper function.
    """
    os.makedirs(directory_path, exist_ok=True)

def resample_audio_file(
    input_path: str, 
    output_path: str, 
    target_samplerate: int, 
    audio_format: str = "wav"
    ) -> bool:
    """Resamples a single audio file and saves it to the specified output path.

    This function uses pydub to load an audio file, change its frame rate
    (sample rate), and then export it to the `output_path` in the
    specified `audio_format`.

    Args:
        input_path (str): The path to the input audio file.
        output_path (str): The path where the resampled audio file will be saved.
        target_samplerate (int): The desired target sample rate in Hz (e.g., 22050).
        audio_format (str): The desired format for the output audio file
                            (e.g., "wav", "mp3"). Defaults to "wav".

    Returns:
        bool: True if resampling and export were successful, False otherwise.

    Raises:
        FileNotFoundError: If the `input_path` does not exist (raised by `AudioSegment.from_file`).
        CouldntDecodeError: If pydub cannot decode the input file (e.g., unsupported
                            format, corrupted file).
        OSError: If there's an issue writing the output file (e.g., permission denied,
                 invalid `output_path`).
    """
    try:
        audio = AudioSegment.from_file(input_path)
        resampled_audio = audio.set_frame_rate(target_samplerate)
        # Pydub export handles format and parameters like bitrate or quality if applicable for the format
        # For WAV, it's straightforward. For MP3, one might add bitrate='192k'.
        # The original scripts used 'PCM_16' for soundfile. Pydub handles WAV parameters internally.
        # To ensure 16-bit PCM WAV, we can specify parameters if needed, though default is often suitable.
        # AudioSegment.export typically defaults to PCM 16-bit for WAV.
        resampled_audio.export(output_path, format=audio_format)
        return True
    except Exception as e:
        print(f"Error resampling file {input_path} to {output_path}: {e}")
        return False

def resample_corpus_to_output_dir(
    src_dir: str, 
    dst_dir: str, 
    target_samplerate: int, 
    audio_format: str = "wav"
    ) -> None:
    """Resamples audio files from a source directory to a new destination directory.

    Recursively walks through the `src_dir`, finds all audio files matching
    the `audio_format` (default '.wav', case-insensitive), resamples them to
    the `target_samplerate`, and saves them to a corresponding path in `dst_dir`.
    The original directory structure from `src_dir` is replicated in `dst_dir`.

    Args:
        src_dir (str): The path to the source directory containing audio files.
        dst_dir (str): The path to the destination directory where resampled
                       files will be saved. This directory will be created if it
                       does not exist.
        target_samplerate (int): The desired target sample rate in Hz.
        audio_format (str): The file extension (without the dot) of audio files
                            to process (e.g., "wav", "mp3"). Defaults to "wav".
                            The comparison is case-insensitive.

    Returns:
        None. Prints summary information to the console.

    Raises:
        FileNotFoundError: If `src_dir` does not exist.
        NotADirectoryError: If `src_dir` is not a directory.
        OSError: For issues related to directory creation or file I/O in `dst_dir`.
                 Individual file processing errors (e.g., `CouldntDecodeError`) are
                 caught, reported, and skipped.
    """
    print(f"Starting resampling process from '{src_dir}' to '{dst_dir}' at {target_samplerate}Hz for '.{audio_format}' files.")
    
    # Counters for tracking processed and failed files
    processed_files_count = 0
    successfully_resampled_count = 0
    
    # os.walk can raise FileNotFoundError or NotADirectoryError if src_dir is invalid.
    # These will propagate as per the 'Raises' documentation.
    for root, _, files in os.walk(src_dir):
        for file in files:
            # Process only files matching the specified audio_format, case-insensitive
            if not file.lower().endswith(f".{audio_format.lower()}"): 
                continue
            
            processed_files_count += 1
            src_file_path = os.path.join(root, file)
            
            # Determine the relative path to maintain directory structure
            relative_path = os.path.relpath(src_file_path, src_dir)
            dst_file_path = os.path.join(dst_dir, relative_path)
            
            try:
                # Ensure the specific output subdirectory exists
                _ensure_dir(os.path.dirname(dst_file_path))
            except OSError as e:
                print(f"Error creating directory {os.path.dirname(dst_file_path)}: {e}. Skipping file {src_file_path}.")
                continue # Skip this file if its destination directory cannot be made
            
            # Attempt to resample the individual file
            if resample_audio_file(src_file_path, dst_file_path, target_samplerate, audio_format):
                successfully_resampled_count +=1
            # resample_audio_file already prints its own errors for individual file failures
            
    print(f"\nResampling complete for directory '{src_dir}'.")
    print(f"Total '.{audio_format}' files found and attempted: {processed_files_count}.")
    print(f"Successfully resampled and saved: {successfully_resampled_count} files.")
    if processed_files_count > successfully_resampled_count:
        print(f"Failed to resample or save: {processed_files_count - successfully_resampled_count} files. See previous error messages for details.")
    print(f"Output directory: '{dst_dir}'.")

def resample_corpus_inplace(
    src_dir: str, 
    target_samplerate: int, 
    audio_format: str = "wav"
    ) -> None:
    """Resamples audio files in a directory in-place, overwriting original files.

    Recursively walks through the `src_dir`, finds all audio files matching the
    `audio_format` (default '.wav', case-insensitive), resamples them to the
    `target_samplerate`, and overwrites the original files.
    **Warning: This operation is destructive and modifies files in place.**

    Args:
        src_dir (str): The path to the source directory containing audio files
                       to be resampled.
        target_samplerate (int): The desired target sample rate in Hz.
        audio_format (str): The file extension (without the dot) of audio files
                            to process (e.g., "wav", "mp3"). Defaults to "wav".
                            The comparison is case-insensitive.

    Returns:
        None. Prints summary information to the console.

    Raises:
        FileNotFoundError: If `src_dir` does not exist.
        NotADirectoryError: If `src_dir` is not a directory.
        OSError: For issues related to file I/O (e.g., if a file cannot be
                 written after resampling). Individual file processing errors
                 (e.g., `CouldntDecodeError`) are caught, reported, and skipped.
    """
    print(f"Starting in-place resampling for '.{audio_format}' files in '{src_dir}' at {target_samplerate}Hz.")
    print("WARNING: This operation will overwrite original files.")
    
    # Find all audio files matching the specified format, case-insensitive
    # glob is suitable here as os.walk is not needed for in-place modification of found files.
    glob_pattern = os.path.join(src_dir, '**', f'*.{audio_format.lower()}')
    audio_files = glob.glob(glob_pattern, recursive=True)
    
    if not audio_files:
        print(f"No '.{audio_format}' files found in '{src_dir}'. No action taken.")
        return

    successfully_resampled_count = 0
    
    # Using tqdm for a progress bar
    with tqdm(total=len(audio_files), desc=f"Resampling '.{audio_format}' files in-place") as pbar:
        for audio_file_path in audio_files:
            # For in-place operation, the output path is the same as the input path.
            if resample_audio_file(audio_file_path, audio_file_path, target_samplerate, audio_format):
                successfully_resampled_count +=1
            # resample_audio_file prints errors for individual file failures
            pbar.update(1)
            
    print(f"\nIn-place resampling complete for directory '{src_dir}'.")
    print(f"Total '.{audio_format}' files found and attempted: {len(audio_files)}.")
    print(f"Successfully resampled in-place: {successfully_resampled_count} files.")
    if len(audio_files) > successfully_resampled_count:
        print(f"Failed to resample: {len(audio_files) - successfully_resampled_count} files. See previous error messages for details.")

if __name__ == '__main__':
    # Basic testing (not part of the library structure, for dev only)
    # Create dummy directories and a wav file for testing
    if not os.path.exists("dummy_src"): os.makedirs("dummy_src/subdir")
    if not os.path.exists("dummy_dst"): os.makedirs("dummy_dst")
    
    try:
        # Create a dummy mono WAV file with pydub for testing
        silence = AudioSegment.silent(duration=1000, frame_rate=48000) # 1 sec, 48kHz
        silence.export("dummy_src/subdir/test_48k.wav", format="wav")

        print("Testing resample_corpus_to_output_dir...")
        resample_corpus_to_output_dir("dummy_src", "dummy_dst_output", 22050)
        if os.path.exists("dummy_dst_output/subdir/test_48k.wav"):
            check_audio = AudioSegment.from_file("dummy_dst_output/subdir/test_48k.wav")
            print(f"Output file sample rate: {check_audio.frame_rate}Hz (Expected 22050Hz)")

        print("\nTesting resample_corpus_inplace...")
        # Copy the original file again for inplace test
        silence.export("dummy_src/subdir/test_inplace_48k.wav", format="wav")
        resample_corpus_inplace("dummy_src", 16000)
        if os.path.exists("dummy_src/subdir/test_inplace_48k.wav"):
            check_audio_inplace = AudioSegment.from_file("dummy_src/subdir/test_inplace_48k.wav")
            print(f"Inplace file sample rate: {check_audio_inplace.frame_rate}Hz (Expected 16000Hz)")

    except Exception as e:
        print(f"Error during test setup or execution: {e}")
    finally:
        # Clean up (optional, comment out to inspect files)
        # import shutil
        # if os.path.exists("dummy_src"): shutil.rmtree("dummy_src")
        # if os.path.exists("dummy_dst_output"): shutil.rmtree("dummy_dst_output")
        print("\nNote: Dummy files/dirs created. For CLI testing, use `audiotools ...`")
