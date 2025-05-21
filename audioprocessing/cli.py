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
    """Main entry point for audioprocessing tools CLI."""
    pass

@main_cli.command()
@click.argument('src_directory', type=click.Path(exists=True, file_okay=False, dir_okay=True))
@click.argument('dst_directory', type=click.Path(file_okay=False, dir_okay=True))
@click.option('--db', 'target_loudness', type=NegativeNumberParamType(), default=-23.0, help='The desired loudness in dB LUFS (e.g., -23.0).')
def normalize(src_directory, dst_directory, target_loudness):
    """Normalizes WAV files in src_directory to target_loudness and saves to dst_directory."""
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
def cli_calculate_similarity_for_directory(ref_path, input_path, output_file, audio_type):
    """Computes similarity between audio files in input_path and a ref_path."""
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
def cli_calculate_similarity_for_speaker_directory_structure(data_dir, ref_dir, output_csv_file, output_log_file, output_model_stats_file, output_speaker_stats_file):
    """Computes similarity for complex directory structures and outputs detailed statistics."""
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
@click.option('--cuda/--no-cuda', default=True, help="Enable or disable CUDA for WVMOS model.")
def cli_calculate_mos_for_directory(input_path, output_file, audio_type, cuda):
    """Computes MOS scores for audio files in input_path using WVMOS."""
    try:
        if not cuda and mos_model.device.type == 'cuda':
             click.echo("Warning: MOS model loaded with CUDA but --no-cuda specified. Model re-init not supported yet.", err=True)
        elif cuda and mos_model.device.type == 'cpu':
             click.echo("Warning: MOS model loaded on CPU but --cuda specified. Model re-init not supported yet.", err=True)
        # Actual model re-initialization based on 'cuda' flag should be handled in selection.py or by passing flag
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
def cli_filter_and_save_csv_by_score_and_duration(input_csv, output_csv, score_col, duration_col, threshold_ms, sort_order):
    """Selects rows from a CSV based on score and total duration, saving to a new CSV."""
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
    output_separator, output_quotechar
    ):
    """Phonetizes transcripts in a CSV file and saves results to a new CSV."""
    try:
        process_csv_for_phonetization(
            input_csv_path=input_csv_file,
            output_csv_path=output_csv_file,
            transcript_column=transcript_column,
            basename_column=basename_column,
            language_column=language_column if not forced_language_code else None, # Ignore lang_col if lang is forced
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
def cli_resample_corpus_to_output_dir(src_directory, dst_directory, target_samplerate, audio_format):
    """Resamples .wav files from src_directory to dst_directory with a new sample rate/format."""
    try:
        resample_corpus_to_output_dir(src_directory, dst_directory, target_samplerate, audio_format)
        click.echo(f"Corpus resampling to output directory complete. Output in '{dst_directory}'.")
    except Exception as e:
        click.echo(f"Error during resampling to output directory: {e}", err=True)

@main_cli.command(name="resample-corpus-inplace")
@click.argument('src_directory', type=click.Path(exists=True, file_okay=False, dir_okay=True, readable=True, writable=True))
@click.option('--rate', 'target_samplerate', type=click.INT, default=22050, help='The desired sampling rate (e.g., 22050).')
@click.option('--format', 'audio_format', type=click.STRING, default="wav", help='The desired audio output format (e.g., "wav", "mp3"). This will also determine the file extension if changed.')
def cli_resample_corpus_inplace(src_directory, target_samplerate, audio_format):
    """Resamples .wav files in src_directory inplace to a new sample rate/format."""
    # Warning for in-place operations
    if not click.confirm(f"This will overwrite original .wav files in '{src_directory}'. Do you want to continue?", abort=True):
        return # Should be handled by abort=True, but as a safeguard.
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
    audio_format
    ):
    """Trims silence from audio files in src_directory and saves to dst_directory."""
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
def cli_copy_files_from_csv_column(csv_filepath, destination_dir, file_path_column_name, csv_delimiter, csv_quotechar):
    """Copies files listed in a CSV column to a destination directory."""
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
def cli_join_csv_files_on_key(csv1_path, csv2_path, key_column, output_csv_path, csv_delimiter, csv_quotechar):
    """Joins two CSV files based on a common key column."""
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
