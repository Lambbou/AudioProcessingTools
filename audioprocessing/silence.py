import os
from pydub import AudioSegment
from pydub.silence import split_on_silence
from tqdm import tqdm

def _ensure_dir(directory_path):
    """Ensure the directory exists."""
    os.makedirs(directory_path, exist_ok=True)

def trim_audio_file_pydub(
    input_path: str, 
    output_path: str, 
    silence_thresh_dbfs: int, 
    min_silence_len_ms: int = 1000, 
    padding_ms: int = 50, # Corresponds to keep_silence in split_on_silence if used differently
    audio_format: str = "wav"
) -> tuple[int, int] | None:
    """
    Trims silence from a single audio file using pydub and saves it.

    Args:
        input_path: Path to the input audio file.
        output_path: Path to save the trimmed audio file.
        silence_thresh_dbfs: Silence threshold in dBFS.
        min_silence_len_ms: Minimum length of silence (in ms) to be considered for splitting.
        padding_ms: Amount of silence (in ms) to leave at the beginning and end if trimming occurs.
                    The original script added 50ms unconditionally if any trimming happened.
                    pydub's split_on_silence has `keep_silence` which can be used for padding around chunks.
                    Here, we'll replicate the original script's behavior of adding padding to the final concatenated audio if it was trimmed.
        audio_format: The format for the output audio file.

    Returns:
        A tuple (original_length_ms, trimmed_length_ms) if successful, else None.
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
):
    """
    Recursively walks src_dir, trims silence from .wav files,
    saves them to dst_dir, and optionally creates report files.
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
