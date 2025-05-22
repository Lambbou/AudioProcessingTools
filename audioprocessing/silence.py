"""Audio silence detection and trimming utilities.

This module provides functionalities for detecting and removing silence from
audio files using the pydub library. It includes tools to process individual
audio files or entire directories of audio files, saving the trimmed versions
and optionally generating reports about the trimming process.
"""
import os
from pydub import AudioSegment
from pydub.silence import split_on_silence
from pydub.exceptions import CouldntDecodeError # For documenting exceptions
from tqdm import tqdm

def _ensure_dir(directory_path: str) -> None:
    """Ensure the directory exists, creating it if necessary.
    
    Internal helper function.
    """
    os.makedirs(directory_path, exist_ok=True)

def trim_audio_file_pydub(
    input_path: str, 
    output_path: str, 
    silence_thresh_dbfs: int, 
    min_silence_len_ms: int = 1000, 
    padding_ms: int = 50, 
    audio_format: str = "wav"
) -> tuple[int, int] | None:
    """Trims silence from an audio file using pydub and saves the result.

    This function identifies non-silent segments in an audio file based on the
    provided silence threshold and minimum silence length. It then concatenates
    these non-silent segments, optionally adding padding around them (via
    `keep_silence` in `pydub.silence.split_on_silence`), and exports the
    result to a new audio file.

    Args:
        input_path (str): Path to the input audio file.
        output_path (str): Path to save the trimmed audio file.
        silence_thresh_dbfs (int): The threshold (in dBFS) below which audio is
            considered silent. For example, -40 means parts of audio quieter
            than -40 dBFS are considered silence.
        min_silence_len_ms (int): The minimum duration (in milliseconds) that
            a segment of audio must be silent for it to be considered a
            "silence" break.
        padding_ms (int): Amount of silence (in milliseconds) to keep at the
            beginning and end of each detected non-silent chunk. This is passed
            to the `keep_silence` parameter of `pydub.silence.split_on_silence`.
            If set to 0, no padding is added around the chunks.
        audio_format (str): The format for the output audio file (e.g., "wav",
            "mp3"). Defaults to "wav".

    Returns:
        tuple[int, int] | None: A tuple containing the original duration and
            the trimmed duration of the audio in milliseconds, i.e.,
            `(original_length_ms, trimmed_length_ms)`, if successful.
            Returns `None` if an error occurs during processing.

    Raises:
        FileNotFoundError: If `input_path` does not exist (from `AudioSegment.from_file`).
        CouldntDecodeError: If pydub cannot decode the input file (e.g.,
                            unsupported format or corrupted file).
        OSError: If there are issues writing the output file (e.g., permission errors).
    """
    try:
        audio = AudioSegment.from_file(input_path)
        original_length_ms = len(audio)

        # split_on_silence parameters:
        # - audio_segment: the audio to process
        # - min_silence_len: minimum length of a silence to be used for splitting (in ms)
        # - silence_thresh: silence threshold in dBFS (lower means more sensitive to silence)
        # - keep_silence: amount of silence to keep at the beginning and end of each non-silent chunk (in ms or bool)
        # The original script did not use keep_silence in split_on_silence but added fixed padding later.
        
        audio_segments = split_on_silence(
            audio,
            min_silence_len=min_silence_len_ms,
            silence_thresh=silence_thresh_dbfs,
            keep_silence=padding_ms # This adds padding around each segment.
                                    # If we want behavior identical to original (padding only if trimmed, and only at very start/end),
                                    # we'd set keep_silence=False and add padding manually after concatenation.
                                    # For now, let's use keep_silence for a more robust padding.
        )

        if not audio_segments: # If the entire file is considered silence
            trimmed_audio = AudioSegment.empty()
        else:
            trimmed_audio = sum(audio_segments, AudioSegment.empty())
        
        trimmed_length_ms = len(trimmed_audio)

        # Original script's logic: if data was trimmed, add 50ms silences before and after.
        # This is slightly different from keep_silence in split_on_silence if there are multiple segments.
        # To precisely match: set keep_silence=False above, then:
        # if trimmed_length_ms < original_length_ms and trimmed_length_ms > 0 and padding_ms > 0:
        #    trimmed_audio = AudioSegment.silent(duration=padding_ms) + trimmed_audio + AudioSegment.silent(duration=padding_ms)
        #    trimmed_length_ms = len(trimmed_audio) # Recalculate length
        # For this refactor, using keep_silence in split_on_silence is a common way to handle padding.
        # If exact original padding logic is critical, it can be adjusted.

        trimmed_audio.export(output_path, format=audio_format)
        return original_length_ms, trimmed_length_ms
    except Exception as e:
        print(f"Error trimming file {input_path} to {output_path}: {e}")
        return None

def process_corpus_for_silence_trimming(
    src_dir: str, 
    dst_dir: str, 
    silence_thresh_dbfs: int, 
    min_silence_len_ms: int = 1000, 
    padding_ms: int = 50,
    create_report_files: bool = True,
    audio_format: str = "wav"
) -> None:
    """Processes a directory of audio files to trim silence from each file.

    Recursively walks through the `src_dir`, finds all audio files matching the
    specified `audio_format` (case-insensitive), and applies silence trimming
    using `trim_audio_file_pydub`. Trimmed files are saved to a corresponding
    path in `dst_dir`, maintaining the original directory structure.
    Optionally, a `.txt` report file can be created for each processed audio
    file, detailing the original and trimmed durations.

    Args:
        src_dir (str): The path to the source directory containing audio files.
        dst_dir (str): The path to the destination directory where trimmed audio
                       files (and reports) will be saved. This directory will be
                       created if it does not exist.
        silence_thresh_dbfs (int): The silence threshold in dBFS passed to
                                   `trim_audio_file_pydub`.
        min_silence_len_ms (int): The minimum silence length in milliseconds
                                  passed to `trim_audio_file_pydub`.
        padding_ms (int): The padding in milliseconds to keep around non-silent
                          segments, passed to `trim_audio_file_pydub`.
        create_report_files (bool): If True, a `.txt` file is created for each
                                    processed audio file, containing its original
                                    duration, trimmed duration, and the amount of
                                    silence removed. Defaults to True.
        audio_format (str): The file extension (without the dot) of audio files
                            to process (e.g., "wav", "mp3"). Defaults to "wav".

    Returns:
        None. Results are written to `dst_dir`, and progress is printed to the console.

    Raises:
        FileNotFoundError: If `src_dir` does not exist.
        NotADirectoryError: If `src_dir` is not a directory.
        OSError: For issues related to directory creation or file I/O in `dst_dir`.
                 Individual file processing errors (e.g., `CouldntDecodeError` from
                 `trim_audio_file_pydub`) are caught, reported, and skipped, allowing
                 the batch process to continue.
    """
    print(f"Trimming silence from '.{audio_format}' files in '{src_dir}' to '{dst_dir}'.")
    print(f"Parameters: Threshold={silence_thresh_dbfs}dBFS, Min Silence={min_silence_len_ms}ms, Padding={padding_ms}ms")

    processed_files = 0
    failed_files = 0

    for root, _, files in tqdm(list(os.walk(src_dir)), desc="Processing directories"):
        for file in files:
            if not file.lower().endswith(f".{audio_format}"):
                continue
            
            src_file_path = os.path.join(root, file)
            relative_path = os.path.relpath(src_file_path, src_dir)
            dst_file_path = os.path.join(dst_dir, relative_path)
            
            _ensure_dir(os.path.dirname(dst_file_path))
            
            trim_result = trim_audio_file_pydub(
                src_file_path, 
                dst_file_path, 
                silence_thresh_dbfs, 
                min_silence_len_ms,
                padding_ms, # This is now passed to keep_silence in trim_audio_file_pydub
                audio_format
            )
            
            if trim_result:
                original_length_ms, trimmed_length_ms = trim_result
                processed_files += 1
                if create_report_files:
                    report_file_path = os.path.splitext(dst_file_path)[0] + ".txt"
                    try:
                        with open(report_file_path, 'w') as report_file:
                            report_file.write("original_duration_ms,trimmed_duration_ms,silence_removed_ms\n")
                            report_file.write(f"{original_length_ms},{trimmed_length_ms},{original_length_ms - trimmed_length_ms}\n")
                    except Exception as e:
                        print(f"Error writing report file {report_file_path}: {e}")
            else:
                failed_files += 1
                
    print(f"Silence trimming complete. Processed {processed_files} files. Failed to process {failed_files} files.")
    if not create_report_files:
        print("Report file creation was disabled.")
    print(f"Output directory: '{dst_dir}'. Only '.{audio_format}' files were processed.")

if __name__ == '__main__':
    # Basic testing
    dummy_src = "dummy_trim_src"
    dummy_dst = "dummy_trim_dst"
    _ensure_dir(os.path.join(dummy_src, "subdir"))

    try:
        # Create a dummy WAV file: 500ms silence, 1s tone, 500ms silence
        tone = AudioSegment.sine(1000, duration=1000).to_audio_segment(frame_rate=44100) # 1s tone
        silence_start = AudioSegment.silent(duration=500, frame_rate=44100)
        silence_end = AudioSegment.silent(duration=500, frame_rate=44100)
        test_audio = silence_start + tone + silence_end
        test_file_path = os.path.join(dummy_src, "subdir", "test_trim.wav")
        test_audio.export(test_file_path, format="wav")
        print(f"Created test file: {test_file_path} (Duration: {len(test_audio)}ms)")

        process_corpus_for_silence_trimming(
            dummy_src, 
            dummy_dst, 
            silence_thresh_dbfs=-50, # pydub default is -16. Louder sounds need lower (more negative) threshold.
            min_silence_len_ms=200, # Shorter than default 1000ms to catch 500ms silences
            padding_ms=50 # Keep 50ms padding
        )

        output_check_path = os.path.join(dummy_dst, "subdir", "test_trim.wav")
        if os.path.exists(output_check_path):
            trimmed = AudioSegment.from_file(output_check_path)
            print(f"Trimmed file duration: {len(trimmed)}ms (Expected around 1000ms + 2*50ms padding = 1100ms)")
            report_path = os.path.join(dummy_dst, "subdir", "test_trim.txt")
            if os.path.exists(report_path):
                with open(report_path, 'r') as f:
                    print(f"Report content:\n{f.read()}")
        else:
            print(f"Output file {output_check_path} not found.")

    except Exception as e:
        print(f"Error during test: {e}")
    finally:
        # import shutil
        # if os.path.exists(dummy_src): shutil.rmtree(dummy_src)
        # if os.path.exists(dummy_dst): shutil.rmtree(dummy_dst)
        print("\nNote: Dummy files/dirs created. For CLI testing, use `audiotools ...`")
