"""Command-line interface for the audioprocessing library.

This module defines the command-line interface for interacting with the
audioprocessing library's functionalities. It uses Click to create a
user-friendly CLI with various subcommands for different audio processing
tasks such as normalization, similarity computation, MOS scoring,
sample selection, phonetization, resampling, silence trimming, and
file/CSV manipulation.

The main entry point is `audiotools`, which is registered via `setup.py`.
Each subcommand corresponds to a specific function within the library,
allowing users to apply these tools directly from the command line.
"""
import click
from .normalization import process_directory_for_normalization, NegativeNumberParamType
from .similarity import (
    calculate_similarity_for_directory, 
    calculate_similarity_for_speaker_directory_structure, 
    speaker_encoder
)
from .selection import (
    calculate_mos_for_directory,
    filter_and_save_csv_by_score_and_duration,
    mos_model
)

@click.group()
def main_cli():
    """Main entry point for the audioprocessing tools CLI.
    
    This function serves as the main group for all CLI commands offered
    by the audioprocessing library.
    """
    pass

@main_cli.command()
@click.argument('src_directory', type=click.Path(exists=True, file_okay=False, dir_okay=True))
@click.argument('dst_directory', type=click.Path(file_okay=False, dir_okay=True))
@click.option('--db', 'target_loudness', type=NegativeNumberParamType(), default=-23.0, help='The desired loudness in dB LUFS (e.g., -23.0).')
def normalize(src_directory: str, dst_directory: str, target_loudness: float):
    """Normalizes WAV files from a source directory to a target loudness.

    This command processes all WAV files within the `src_directory`, normalizes
    their loudness to the specified `target_loudness` in dB LUFS, and saves
    the processed files to the `dst_directory`, maintaining the original
    directory structure.

    Args:
        src_directory (str): The path to the source directory containing WAV files.
        dst_directory (str): The path to the destination directory where normalized
                             files will be saved. This directory will be created if
                             it doesn't exist.
        target_loudness (float): The target loudness in dB LUFS (e.g., -23.0).
                                 This value must be negative.
    """
    try:
        process_directory_for_normalization(src_directory, dst_directory, target_loudness)
        click.echo(f"Normalization of '{src_directory}' to '{dst_directory}' complete.")
    except Exception as e:
        click.echo(f"Error during normalization: {e}", err=True)

@main_cli.command(name="compute-similarity")
@click.argument('ref_path', type=click.Path(exists=True, dir_okay=False)) # Reference audio file
@click.argument('input_path', type=click.Path(exists=True, file_okay=False, dir_okay=True)) 
@click.argument('output_file', type=click.Path(dir_okay=False))
@click.option('-t', '--type', 'audio_type', type=click.STRING, default="wav", help='The type of audio file to look for (e.g., wav, flac).')
def cli_calculate_similarity_for_directory(ref_path: str, input_path: str, output_file: str, audio_type: str):
    """Computes similarity between audio files in a directory and a reference file.

    Calculates the cosine similarity between each audio file (of the specified `audio_type`)
    in the `input_path` directory and the `ref_path` audio file.
    The results, including file paths, duration, and similarity scores, are saved
    to a CSV file specified by `output_file`.

    Args:
        ref_path (str): Path to the reference audio file.
        input_path (str): Path to the directory containing audio files to process.
        output_file (str): Path to save the output CSV file.
        audio_type (str): The type/extension of audio files to process (e.g., 'wav', 'flac').
                          Defaults to 'wav'.
    """
    try:
        calculate_similarity_for_directory(ref_path, input_path, output_file, audio_type, speaker_encoder)
        click.echo(f"Similarity computation complete. Output saved to '{output_file}'.")
    except Exception as e:
        click.echo(f"Error during similarity computation: {e}", err=True)

@main_cli.command(name="compute-similarity-advanced")
@click.option('--data_dir', type=click.Path(exists=True, file_okay=False, dir_okay=True), required=True, help='Directory of wav files to test (structure : data_dir/model_name/speaker_name/wav_files)')
@click.option('--ref_dir', type=click.Path(exists=True, file_okay=False, dir_okay=True), required=True, help='Directory of reference wav files (structure : ref_dir/speaker_name/wav_files)')
@click.option('--output_csv_file', type=click.Path(dir_okay=False), required=True, help='Where to save the results of the objective evaluation')
@click.option('--output_log_file', type=click.Path(dir_okay=False), required=True, help='Where to save the means/std of the objective evaluation')
@click.option('--output_model_stats_file', type=click.Path(dir_okay=False), required=True, help='Where to save the means/std of the objective evaluation for models')
@click.option('--output_speaker_stats_file', type=click.Path(dir_okay=False), required=True, help='Where to save the means/std of the objective evaluation for speakers')
def cli_calculate_similarity_for_speaker_directory_structure(
    data_dir: str, ref_dir: str, output_csv_file: str, output_log_file: str, 
    output_model_stats_file: str, output_speaker_stats_file: str
    ):
    """Computes similarity for complex directory structures and outputs detailed statistics.

    This command processes audio files organized in a specific directory structure:
    `data_dir/model_name/speaker_name/wav_files`. It compares these files against
    reference audio files located in `ref_dir/speaker_name/wav_files`.
    It calculates cosine similarity and Euclidean distance, outputting:
    - A main CSV file (`output_csv_file`) with detailed similarity scores for each file pair.
    - A log file (`output_log_file`) with means, standard deviations, and confidence intervals.
    - Separate CSV files for aggregated statistics per model (`output_model_stats_file`)
      and per speaker (`output_speaker_stats_file`).

    All input and output paths are specified via Click options.
    """
    try:
        calculate_similarity_for_speaker_directory_structure(
            data_dir, 
            ref_dir, 
            output_csv_file, 
            output_log_file, 
            output_model_stats_file, 
            output_speaker_stats_file, 
            speaker_encoder
        )
        click.echo(f"Advanced similarity computation complete. Outputs saved to specified files.")
    except Exception as e:
        click.echo(f"Error during advanced similarity computation: {e}", err=True)

@main_cli.command(name="compute-mos")
@click.argument('input_path', type=click.Path(exists=True, file_okay=False, dir_okay=True))
@click.argument('output_file', type=click.Path(dir_okay=False))
@click.option('-t', '--type', 'audio_type', type=click.STRING, default="wav", help='The type of audio file to look for (e.g., wav, flac).')
@click.option('--cuda/--no-cuda', 'use_cuda', default=True, help="Enable or disable CUDA for WVMOS model.") # Renamed 'cuda' to 'use_cuda'
def cli_calculate_mos_for_directory(input_path: str, output_file: str, audio_type: str, use_cuda: bool):
    """Computes MOS scores for audio files in a directory using WVMOS.

    This command processes all audio files of the specified `audio_type` within
    the `input_path` directory. It uses the WVMOS model to predict a Mean
    Opinion Score (MOS) for each file. Results, including file path, duration,
    and predicted MOS, are saved to the `output_file` CSV.

    Args:
        input_path (str): Path to the directory containing audio files to process.
        output_file (str): Path to save the output CSV file.
        audio_type (str): The type/extension of audio files to process (e.g., 'wav', 'flac').
                          Defaults to 'wav'.
        use_cuda (bool): Flag to enable or disable CUDA for the WVMOS model.
                         Defaults to True (enable CUDA). Note: Model re-initialization
                         based on this flag at runtime is not fully supported if the model
                         was already loaded with a different device setting.
    """
    try:
        # Check for potential mismatch in CUDA settings if model is pre-loaded
        if not use_cuda and mos_model.device.type == 'cuda':
             click.echo("Warning: MOS model loaded with CUDA but --no-cuda specified. Model re-init not supported yet.", err=True)
        elif use_cuda and mos_model.device.type == 'cpu':
             click.echo("Warning: MOS model loaded on CPU but --cuda specified. Model re-init not supported yet.", err=True)
        
        # Actual model re-initialization based on 'use_cuda' flag should ideally be handled 
        # within the calculate_mos_for_directory function or by passing the flag to it
        # if the model can be re-instantiated or moved to a different device.
        # Currently, mos_model is imported with a fixed device setting from selection.py.
        calculate_mos_for_directory(input_path, output_file, audio_type, mos_model)
        click.echo(f"MOS computation complete. Output saved to '{output_file}'.")
    except Exception as e:
        click.echo(f"Error during MOS computation: {e}", err=True)

@main_cli.command(name="select-best-samples")
@click.argument('input_csv', type=click.Path(exists=True, dir_okay=False))
@click.argument('output_csv', type=click.Path(dir_okay=False))
@click.option('--score-col', default='MOS', help='Name of the column containing the score to sort by.')
@click.option('--duration-col', default='Duration', help='Name of the column containing duration values (in ms).')
@click.option('--threshold-ms', default=600000, type=int, help='Total duration threshold in milliseconds for selected samples. Set to 0 for no limit.')
@click.option('--sort-order', type=click.Choice(['asc', 'desc']), default='desc', help='Sort order for the score column.')
def cli_filter_and_save_csv_by_score_and_duration(
    input_csv: str, output_csv: str, score_col: str, 
    duration_col: str, threshold_ms: int, sort_order: str
    ):
    """Selects rows from a CSV based on score and total duration.

    This command filters rows from an `input_csv` file. It sorts the rows
    based on the `score_col` (in `sort_order`, default descending) and
    selects rows such that their cumulative duration (from `duration_col`, in ms)
    does not exceed `threshold_ms`. The selected rows are saved to `output_csv`.
    A threshold of 0 means no limit.

    Args:
        input_csv (str): Path to the input CSV file (must exist).
        output_csv (str): Path to save the output CSV file.
        score_col (str): Name of the column containing the score to sort by (default: 'MOS').
        duration_col (str): Name of the column containing duration values in milliseconds (default: 'Duration').
        threshold_ms (int): Total duration threshold in milliseconds for selected samples (default: 600000).
                            Set to 0 for no limit.
        sort_order (str): Sort order for the score column ('asc' or 'desc', default: 'desc').
    """
    try:
        sort_desc = True if sort_order == 'desc' else False
        filter_and_save_csv_by_score_and_duration(
            input_csv_path=input_csv,
            output_csv_path=output_csv,
            score_column_name=score_col,
            duration_column_name=duration_col,
            size_threshold_ms=threshold_ms,
            sort_descending=sort_desc
        )
        click.echo(f"Sample selection complete. Output CSV saved to '{output_csv}'.")
    except Exception as e:
        click.echo(f"Error during sample selection: {e}", err=True)

from .transcription import process_csv_for_phonetization

@main_cli.command(name="phonetize-csv")
@click.argument('input_csv_file', type=click.Path(exists=True, dir_okay=False, readable=True))
@click.argument('output_csv_file', type=click.Path(writable=True, dir_okay=False))
@click.option('-l', '--lang', 'forced_language_code', type=click.STRING, default=None, help='Force language for all entries (e.g., "en-us", "french"). Overrides language column.')
@click.option('-lc', '--lang-col', 'language_column', type=click.STRING, default='Language', help='Name of the CSV column containing language identifiers for each row.')
@click.option('-bc', '--basename-col', 'basename_column', type=click.STRING, default="Basename", help='Name of the CSV column for the basename or ID.')
@click.option('-tc', '--transcript-col', 'transcript_column', type=click.STRING, default="Transcript", help='Name of the CSV column for the text transcript.')
@click.option('-s', '--separator', 'input_separator', type=click.STRING, default="\t", help='Separator for the input CSV file.')
@click.option('-q', '--quotechar', 'input_quotechar', type=click.STRING, default='|', help='Quote character for the input CSV file.')
@click.option('-os', '--out-separator', 'output_separator', type=click.STRING, default="\t", help='Separator for the output CSV file.')
@click.option('-oq', '--out-quotechar', 'output_quotechar', type=click.STRING, default='|', help='Quote character for the output CSV file.')
def cli_process_csv_for_phonetization(
    input_csv_file, output_csv_file, forced_language_code, language_column, 
    basename_column, transcript_column, input_separator, input_quotechar,
    output_separator: str, output_quotechar: str
    ):
    """Phonetizes transcripts in a CSV file and saves results to a new CSV.

    This command reads an `input_csv_file`, phonetizes text from the
    `transcript_column` for each row, and writes the results along with the
    `basename_column` and language information to `output_csv_file`.
    Language can be determined from a `language_column` in the CSV or forced
    globally using the `forced_language_code` option.

    Args:
        input_csv_file (str): Path to the input CSV file.
        output_csv_file (str): Path to save the output CSV file with phonetizations.
        forced_language_code (str, optional): Language code (e.g., "en-us", "french")
            to force for all entries. Overrides `language_column`. Defaults to None.
        language_column (str): Name of the CSV column containing language identifiers
            for each row (e.g., "english", "fr-fr"). Defaults to "Language".
            Ignored if `forced_language_code` is set.
        basename_column (str): Name of the CSV column for the basename or ID
            (e.g., file path or unique identifier). Defaults to "Basename".
        transcript_column (str): Name of the CSV column containing the text to be
            phonetized. Defaults to "Transcript".
        input_separator (str): Separator character for the input CSV file. Defaults to tab.
        input_quotechar (str): Quote character for the input CSV file. Defaults to '|'.
        output_separator (str): Separator character for the output CSV file. Defaults to tab.
        output_quotechar (str): Quote character for the output CSV file. Defaults to '|'.
    """
    try:
        process_csv_for_phonetization(
            input_csv_path=input_csv_file,
            output_csv_path=output_csv_file,
            transcript_column=transcript_column,
            basename_column=basename_column,
            # If a global language is forced, the language_column from CSV is ignored.
            language_column=language_column if not forced_language_code else None, 
            forced_language_code=forced_language_code,
            input_separator=input_separator,
            input_quotechar=input_quotechar,
            output_separator=output_separator,
            output_quotechar=output_quotechar
        )
        click.echo(f"Phonetization processing complete. Output saved to '{output_csv_file}'.")
    except Exception as e:
        click.echo(f"Error during CSV phonetization: {e}", err=True)

from .resampling import resample_corpus_to_output_dir, resample_corpus_inplace

@main_cli.command(name="resample-corpus")
@click.argument('src_directory', type=click.Path(exists=True, file_okay=False, dir_okay=True, readable=True))
@click.argument('dst_directory', type=click.Path(file_okay=False, dir_okay=True, writable=True)) # Not exists=True, as it will be created
@click.option('--rate', 'target_samplerate', type=click.INT, default=22050, help='The desired sampling rate (e.g., 22050).')
@click.option('--format', 'audio_format', type=click.STRING, default="wav", help='The desired audio output format (e.g., "wav", "mp3").')
def cli_resample_corpus_to_output_dir(src_directory: str, dst_directory: str, target_samplerate: int, audio_format: str):
    """Resamples audio files from a source to a destination directory.

    This command processes audio files (default '.wav') in the `src_directory`,
    resamples them to the `target_samplerate` using `pydub`, and saves them to
    the `dst_directory` with the specified `audio_format`. The original directory
    structure is replicated in the destination.

    Args:
        src_directory (str): Path to the source directory containing audio files.
        dst_directory (str): Path to the destination directory where resampled files
                             will be saved. It will be created if it doesn't exist.
        target_samplerate (int): The desired sampling rate in Hz (e.g., 22050).
        audio_format (str): The desired audio output format (e.g., "wav", "mp3").
                            Defaults to "wav".
    """
    try:
        resample_corpus_to_output_dir(src_directory, dst_directory, target_samplerate, audio_format)
        click.echo(f"Corpus resampling to output directory complete. Output in '{dst_directory}'.")
    except Exception as e:
        click.echo(f"Error during resampling to output directory: {e}", err=True)

@main_cli.command(name="resample-corpus-inplace")
@click.argument('src_directory', type=click.Path(exists=True, file_okay=False, dir_okay=True, readable=True, writable=True))
@click.option('--rate', 'target_samplerate', type=click.INT, default=22050, help='The desired sampling rate (e.g., 22050).')
@click.option('--format', 'audio_format', type=click.STRING, default="wav", help='The desired audio output format (e.g., "wav", "mp3"). This will also determine the file extension if changed.')
def cli_resample_corpus_inplace(src_directory: str, target_samplerate: int, audio_format: str):
    """Resamples audio files in a directory in-place.

    This command processes audio files (default '.wav') in the `src_directory`,
    resamples them to the `target_samplerate` using `pydub`, and overwrites
    the original files with the specified `audio_format`.
    A confirmation prompt is shown before modifying files.

    Args:
        src_directory (str): Path to the source directory containing audio files
                             to be resampled in-place.
        target_samplerate (int): The desired sampling rate in Hz (e.g., 22050).
        audio_format (str): The desired audio output format (e.g., "wav", "mp3").
                            Defaults to "wav". This will also determine the file
                            extension if changed (though current implementation primarily
                            focuses on resampling .wav files).
    """
    # Display a confirmation prompt because this operation is destructive.
    if not click.confirm(f"This will overwrite original .wav files in '{src_directory}'. Do you want to continue?", abort=True):
        # abort=True will exit if user says no.
        return 
    try:
        resample_corpus_inplace(src_directory, target_samplerate, audio_format)
        click.echo(f"Corpus resampling in-place complete in '{src_directory}'.")
    except Exception as e:
        click.echo(f"Error during in-place resampling: {e}", err=True)

# NegativeNumberParamType is already in normalization, used by trim_silence's --db option
from .normalization import NegativeNumberParamType 
from .silence import process_corpus_for_silence_trimming

@main_cli.command(name="trim-silence-corpus")
@click.argument('src_directory', type=click.Path(exists=True, file_okay=False, dir_okay=True, readable=True))
@click.argument('dst_directory', type=click.Path(file_okay=False, dir_okay=True, writable=True))
@click.option('--db', 'silence_thresh_dbfs', type=NegativeNumberParamType(), default=-40, help='Silence threshold in dBFS (e.g., -40). Lower values are more sensitive.')
@click.option('--min-len', 'min_silence_len_ms', type=click.INT, default=1000, help='Minimum duration of silence (in ms) to be considered for trimming (default: 1000).')
@click.option('--padding', 'padding_ms', type=click.INT, default=50, help='Padding (in ms) to keep around non-silent segments (default: 50). Passed to pydub keep_silence.')
@click.option('--reports/--no-reports', 'create_report_files', default=True, help='Create .txt report files for each trimmed audio file (default: True).')
@click.option('--format', 'audio_format', type=click.STRING, default="wav", help='Audio file format to process (default: "wav").')
def cli_process_corpus_for_silence_trimming(
    src_directory, 
    dst_directory, 
    silence_thresh_dbfs, 
    min_silence_len_ms,
    padding_ms,
    create_report_files,
    audio_format: str
    ):
    """Trims silence from audio files in a directory.

    Processes audio files (specified by `audio_format`, default 'wav') in the
    `src_directory`. It trims leading and trailing silence based on the
    `silence_thresh_dbfs` and `min_silence_len_ms`. The trimmed audio, with
    optional `padding_ms` around non-silent segments, is saved to the
    `dst_directory`, replicating the original structure.
    Optionally, it creates `.txt` report files detailing the amount of trimmed silence.

    Args:
        src_directory (str): Path to the source directory.
        dst_directory (str): Path to the destination directory.
        silence_thresh_dbfs (int): Silence threshold in dBFS (e.g., -40).
                                   Lower values are more sensitive to silence.
        min_silence_len_ms (int): Minimum duration of silence (in ms) to be
                                  considered for trimming (default: 1000).
        padding_ms (int): Padding (in ms) to keep around non-silent segments
                          (default: 50). This is passed to pydub's `keep_silence`
                          parameter in `split_on_silence`.
        create_report_files (bool): If True, creates a .txt report file for each
                                    trimmed audio file detailing original and
                                    trimmed durations. Defaults to True.
        audio_format (str): Audio file format to process (default: "wav").
    """
    try:
        process_corpus_for_silence_trimming(
            src_dir=src_directory,
            dst_dir=dst_directory,
            silence_thresh_dbfs=silence_thresh_dbfs,
            min_silence_len_ms=min_silence_len_ms,
            padding_ms=padding_ms,
            create_report_files=create_report_files,
            audio_format=audio_format
        )
        click.echo(f"Silence trimming complete. Output in '{dst_directory}'.")
    except Exception as e:
        click.echo(f"Error during silence trimming: {e}", err=True)

from .io_utils import copy_files_from_csv_column, join_csv_files_on_key

@main_cli.command(name="copy-selected-files")
@click.argument('csv_filepath', type=click.Path(exists=True, dir_okay=False, readable=True))
@click.argument('destination_dir', type=click.Path(file_okay=False, writable=True)) # No exists=True, will be created
@click.option('--path-col', 'file_path_column_name', type=click.STRING, default="Basename", help='Name of the column in CSV containing file paths to copy (default: Basename).')
@click.option('--delimiter', 'csv_delimiter', type=click.STRING, default="\t", help='Delimiter for the input CSV file (default: tab).')
@click.option('--quotechar', 'csv_quotechar', type=click.STRING, default='|', help='Quote character for the input CSV file (default: |).')
def cli_copy_files_from_csv_column(
    csv_filepath: str, destination_dir: str, file_path_column_name: str, 
    csv_delimiter: str, csv_quotechar: str
    ):
    """Copies files listed in a CSV column to a destination directory.

    Reads a CSV file specified by `csv_filepath`. For each row, it extracts
    a file path from the column named `file_path_column_name` (default 'Basename')
    and copies the specified file to the `destination_dir`.
    The base name of the source file is used for the copied file.

    Args:
        csv_filepath (str): Path to the input CSV file.
        destination_dir (str): Path to the directory where files will be copied.
                               This directory will be created if it doesn't exist.
        file_path_column_name (str): Name of the column in the CSV that contains
                                     the full paths to the files to be copied.
                                     Defaults to "Basename".
        csv_delimiter (str): Delimiter used in the input CSV file. Defaults to tab.
        csv_quotechar (str): Quote character used in the input CSV file. Defaults to '|'.
    """
    try:
        success = copy_files_from_csv_column(
            csv_filepath=csv_filepath,
            file_path_column_name=file_path_column_name,
            destination_dir=destination_dir,
            csv_delimiter=csv_delimiter,
            csv_quotechar=csv_quotechar
        )
        if success:
            click.echo(f"File copying based on CSV '{csv_filepath}' complete. Check console for details.")
        else:
            click.echo(f"File copying based on CSV '{csv_filepath}' encountered errors. Check console for details.", err=True)
    except Exception as e:
        click.echo(f"Error during file copying from CSV: {e}", err=True)

@main_cli.command(name="join-csv-files")
@click.argument('csv1_path', type=click.Path(exists=True, dir_okay=False, readable=True))
@click.argument('csv2_path', type=click.Path(exists=True, dir_okay=False, readable=True))
@click.argument('key_column', type=click.STRING)
@click.option('-o', '--output', 'output_csv_path', type=click.Path(writable=True, dir_okay=False), default="joined_data.csv", help="Output CSV file name (default: joined_data.csv).")
@click.option('--delimiter', 'csv_delimiter', type=click.STRING, default="\t", help='Delimiter for input and output CSV files (default: tab).')
@click.option('--quotechar', 'csv_quotechar', type=click.STRING, default='|', help='Quote character for input and output CSV files (default: |).')
def cli_join_csv_files_on_key(
    csv1_path: str, csv2_path: str, key_column: str, 
    output_csv_path: str, csv_delimiter: str, csv_quotechar: str
    ):
    """Joins two CSV files based on a common key column.

    This command performs an inner join of two CSV files (`csv1_path` and `csv2_path`)
    based on matching values in the specified `key_column`. The header of the
    output CSV (`output_csv_path`) consists of all columns from the first CSV,
    followed by columns from the second CSV (excluding its key column).
    Rows are included in the output if the key exists in both input files.

    Args:
        csv1_path (str): Path to the first input CSV file.
        csv2_path (str): Path to the second input CSV file.
        key_column (str): The name of the common column to join on.
        output_csv_path (str): Path to save the joined CSV data.
                               Defaults to "joined_data.csv".
        csv_delimiter (str): Delimiter character for both input and output CSV files.
                             Defaults to tab.
        csv_quotechar (str): Quote character for both input and output CSV files.
                             Defaults to '|'.
    """
    try:
        success = join_csv_files_on_key(
            csv1_path=csv1_path,
            csv2_path=csv2_path,
            key_column=key_column,
            output_csv_path=output_csv_path,
            csv_delimiter=csv_delimiter,
            csv_quotechar=csv_quotechar
        )
        if success:
            click.echo(f"CSV join operation complete. Output saved to '{output_csv_path}'.")
        else:
            click.echo(f"CSV join operation encountered errors. Check console for details.", err=True)
    except Exception as e:
        click.echo(f"Error during CSV join: {e}", err=True)

if __name__ == '__main__':
    main_cli()
