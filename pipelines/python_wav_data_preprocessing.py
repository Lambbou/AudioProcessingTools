"""WAV audio data preprocessing pipeline.

This script orchestrates a series of audio processing steps for WAV files:
1. Resampling: Changes the sampling rate of audio files.
2. Normalization: Adjusts the loudness of audio files to a target level.
3. Silence Trimming: Removes leading and trailing silences from audio files.

The pipeline takes an input directory, processes the WAV files within it,
and saves the final results to an output directory. Intermediate directories
are created for each step and are cleaned up upon successful completion of
the subsequent step.

This script uses the `audioprocessing` library for the core audio operations.
"""
import os
import shutil
import click

# Assuming audioprocessing package is installed or in PYTHONPATH
try:
    from audioprocessing.resampling import resample_corpus_to_output_dir
    from audioprocessing.normalization import process_directory_for_normalization, NegativeNumberParamType
    from audioprocessing.silence import process_corpus_for_silence_trimming
except ImportError as e:
    print(f"Error: Could not import from 'audioprocessing' library. Ensure it is installed and in PYTHONPATH.")
    print(f"Details: {e}")
    # Exit if library is not found, as the script cannot function.
    # Using a more specific exit code could be useful for scripting.
    exit(1) 


def _ensure_dir_exists(dir_path: str, create: bool = True) -> bool:
    """Checks if a directory exists, optionally creates it.
    
    Args:
        dir_path (str): The path to the directory.
        create (bool): If True, creates the directory if it doesn't exist.
                       Defaults to True.
                       
    Returns:
        bool: True if the directory exists (or was successfully created),
              False otherwise (e.g., if `create` is False and it doesn't exist,
              or if a file exists at the path).
    """
    if os.path.exists(dir_path):
        if not os.path.isdir(dir_path):
            click.secho(f"Error: Path '{dir_path}' exists but is not a directory.", fg="red")
            return False
        return True # Directory exists and is a directory
    elif create:
        try:
            os.makedirs(dir_path)
            click.secho(f"Created directory: {dir_path}", fg="green")
            return True
        except OSError as e:
            click.secho(f"Error creating directory '{dir_path}': {e}", fg="red")
            return False
    return False # Doesn't exist and create is False


def _remove_dir_if_exists(dir_path: str, step_name_for_error: str) -> None:
    """Removes a directory and its contents if it exists. Handles errors."""
    if os.path.exists(dir_path):
        try:
            shutil.rmtree(dir_path)
            click.secho(f"Successfully removed intermediate directory: {dir_path}", fg="yellow")
        except OSError as e:
            # Non-critical error for cleanup, so just warn.
            click.secho(f"Warning: Could not remove intermediate directory '{dir_path}' after {step_name_for_error}: {e}", fg="yellow")


def _move_contents(src_dir: str, dst_dir: str) -> bool:
    """Moves all contents from src_dir to dst_dir.
    
    Assumes dst_dir exists. Individual files/dirs from src_dir are moved
    into dst_dir. If items with the same name exist in dst_dir, they may be
    overwritten depending on shutil.move behavior (platform-dependent, often yes).
    
    Args:
        src_dir (str): The source directory.
        dst_dir (str): The destination directory.
        
    Returns:
        bool: True if all items moved successfully, False otherwise.
    """
    all_moved = True
    for item_name in os.listdir(src_dir):
        src_item_path = os.path.join(src_dir, item_name)
        dst_item_path = os.path.join(dst_dir, item_name)
        try:
            shutil.move(src_item_path, dst_item_path)
        except Exception as e:
            click.secho(f"Error moving '{src_item_path}' to '{dst_item_path}': {e}", fg="red")
            all_moved = False
    return all_moved


@click.command(context_settings=dict(help_option_names=['-h', '--help']))
@click.option('--input_dir', '-i', 
              type=click.Path(exists=True, file_okay=False, dir_okay=True, readable=True, resolve_path=True), 
              required=True, help="Path to the directory containing input WAV files.")
@click.option('--output_dir', '-o', 
              type=click.Path(file_okay=False, dir_okay=True, writable=True, resolve_path=True), 
              required=True, help="Path to the directory where processed files will be saved.")
@click.option('--samplerate', '-sr', type=int, default=22050, show_default=True, 
              help="Target sampling rate for resampling (e.g., 22050 Hz).")
@click.option('--audio_format', '-af', type=str, default="wav", show_default=True,
              help="Audio format to process and output (e.g., 'wav').")
@click.option('--db', '-d', 'target_loudness_dbfs', 
              type=NegativeNumberParamType(), default=-23.0, show_default=True, 
              help="Target loudness for normalization in dBFS (e.g., -23.0). Must be negative.")
@click.option('--silence-thresh', '-st', 'silence_thresh_dbfs', 
              type=int, default=-40, show_default=True, 
              help="Silence threshold in dBFS for silence trimming (e.g., -40).")
@click.option('--min-silence-len', '-msl', 'min_silence_len_ms', 
              type=int, default=500, show_default=True, 
              help="Minimum silence length in milliseconds for trimming (e.g., 500).")
@click.option('--padding', '-p', 'padding_ms', 
              type=int, default=50, show_default=True, 
              help="Padding in milliseconds to keep around non-silent segments (e.g., 50).")
@click.option('--no-reports', 'create_reports', is_flag=True, default=True,
              help="Disable creation of .txt report files for silence trimming.")
def preprocess_wav_pipeline(
    input_dir: str,
    output_dir: str,
    samplerate: int,
    audio_format: str, # Added for consistency, though resampling functions handle it
    target_loudness_dbfs: float,
    silence_thresh_dbfs: int,
    min_silence_len_ms: int,
    padding_ms: int,
    create_reports: bool
    ) -> None:
    """Runs a WAV audio preprocessing pipeline: Resample -> Normalize -> Trim Silence.
    
    The pipeline processes WAV files from `input_dir` and saves the final
    results in `output_dir`. Intermediate directories are created within
    `output_dir` for each step and are cleaned up if the subsequent step
    is successful. If any step fails, the script will print an error and exit.
    """
    click.secho("Starting WAV audio preprocessing pipeline...", fg="cyan", bold=True)
    click.secho(f"Input directory: {input_dir}", fg="cyan")
    click.secho(f"Output directory: {output_dir}", fg="cyan")
    click.secho(f"Parameters: Samplerate={samplerate}Hz, Format='.{audio_format}', Target Loudness={target_loudness_dbfs}dBFS, "
                f"Silence Thresh={silence_thresh_dbfs}dBFS, Min Silence Len={min_silence_len_ms}ms, Padding={padding_ms}ms, "
                f"Create Reports={create_reports}", fg="cyan")

    # --- Setup Output and Intermediate Directories ---
    if not _ensure_dir_exists(output_dir, create=True):
        # _ensure_dir_exists already prints error
        click.secho("Exiting due to output directory issue.", fg="red", bold=True)
        exit(1)

    resample_dir = os.path.join(output_dir, "01_resampled")
    norm_dir = os.path.join(output_dir, "02_normalized")
    trim_dir = os.path.join(output_dir, "03_trimmed_silence")

    # Clean up any previous intermediate directories to start fresh
    # This is important if a previous run failed mid-way.
    click.secho("Cleaning up any previous intermediate directories...", fg="yellow")
    _remove_dir_if_exists(resample_dir, "initial cleanup")
    _remove_dir_if_exists(norm_dir, "initial cleanup")
    _remove_dir_if_exists(trim_dir, "initial cleanup")
    
    # Create fresh intermediate directories
    if not all([_ensure_dir_exists(d) for d in [resample_dir, norm_dir, trim_dir]]):
        click.secho("Failed to create one or more intermediate directories. Exiting.", fg="red", bold=True)
        exit(1)
    
    current_input_dir = input_dir
    current_step_output_dir = resample_dir

    # --- Step 1: Resample ---
    click.secho("\nStep 1: Resampling audio files...", fg="blue", bold=True)
    try:
        resample_corpus_to_output_dir(
            src_dir=current_input_dir, 
            dst_dir=current_step_output_dir, 
            target_samplerate=samplerate, 
            audio_format=audio_format # Pass audio_format here
        )
        click.secho(f"Resampling successful. Output in: {current_step_output_dir}", fg="green")
    except Exception as e:
        click.secho(f"Error during resampling: {e}", fg="red", bold=True)
        click.secho("Pipeline aborted at resampling step.", fg="red", bold=True)
        exit(1)

    # Update input for next step
    current_input_dir = current_step_output_dir
    current_step_output_dir = norm_dir

    # --- Step 2: Normalize ---
    click.secho("\nStep 2: Normalizing audio files...", fg="blue", bold=True)
    try:
        process_directory_for_normalization(
            src_dir=current_input_dir, 
            dst_dir=current_step_output_dir, 
            target_loudness=target_loudness_dbfs
        )
        click.secho(f"Normalization successful. Output in: {current_step_output_dir}", fg="green")
        # Cleanup previous step's directory
        _remove_dir_if_exists(current_input_dir, "normalization") # current_input_dir is resample_dir here
    except Exception as e:
        click.secho(f"Error during normalization: {e}", fg="red", bold=True)
        click.secho("Pipeline aborted at normalization step.", fg="red", bold=True)
        exit(1)

    # Update input for next step
    current_input_dir = current_step_output_dir
    current_step_output_dir = trim_dir

    # --- Step 3: Trim Silence ---
    click.secho("\nStep 3: Trimming silence from audio files...", fg="blue", bold=True)
    try:
        process_corpus_for_silence_trimming(
            src_dir=current_input_dir,
            dst_dir=current_step_output_dir,
            silence_thresh_dbfs=silence_thresh_dbfs,
            min_silence_len_ms=min_silence_len_ms,
            padding_ms=padding_ms,
            create_report_files=create_reports, # Pass the flag
            audio_format=audio_format # Pass audio_format here
        )
        click.secho(f"Silence trimming successful. Output in: {current_step_output_dir}", fg="green")
        # Cleanup previous step's directory
        _remove_dir_if_exists(current_input_dir, "silence trimming") # current_input_dir is norm_dir here
    except Exception as e:
        click.secho(f"Error during silence trimming: {e}", fg="red", bold=True)
        click.secho("Pipeline aborted at silence trimming step.", fg="red", bold=True)
        exit(1)
        
    # --- Step 4: Final Move ---
    click.secho("\nStep 4: Moving processed files to final output directory...", fg="blue", bold=True)
    if _move_contents(current_step_output_dir, output_dir): # current_step_output_dir is trim_dir
        click.secho(f"Successfully moved files to: {output_dir}", fg="green")
        _remove_dir_if_exists(current_step_output_dir, "final move") # Remove trim_dir
    else:
        click.secho(f"Error moving files from '{current_step_output_dir}' to '{output_dir}'. "
                    f"Please check messages above. Processed files remain in '{current_step_output_dir}'.", 
                    fg="red", bold=True)
        exit(1)

    click.secho("\nAudio preprocessing pipeline completed successfully!", fg="green", bold=True)

if __name__ == '__main__':
    # This makes the script executable.
    # When run, Click will parse arguments and call preprocess_wav_pipeline.
    preprocess_wav_pipeline()
