import os
import glob
from pydub import AudioSegment
from tqdm import tqdm

def _ensure_dir(directory_path):
    """Ensure the directory exists."""
    os.makedirs(directory_path, exist_ok=True)

def resample_audio_file(input_path: str, output_path: str, target_samplerate: int, audio_format: str = "wav"):
    """
    Resamples a single audio file and saves it to the output path.
    Uses pydub for loading, resampling, and exporting.
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

def resample_corpus_to_output_dir(src_dir: str, dst_dir: str, target_samplerate: int, audio_format: str = "wav"):
    """
    Recursively walks src_dir, resamples .wav files to target_samplerate,
    and saves them to a corresponding path in dst_dir, replicating directory structure.
    """
    print(f"Resampling files from '{src_dir}' to '{dst_dir}' at {target_samplerate}Hz.")
    file_count = 0
    resampled_count = 0
    
    for root, _, files in os.walk(src_dir):
        for file in files:
            # Process only .wav files as per original scripts
            if not file.lower().endswith(".wav"): 
                continue
            
            file_count += 1
            src_file_path = os.path.join(root, file)
            relative_path = os.path.relpath(src_file_path, src_dir)
            dst_file_path = os.path.join(dst_dir, relative_path)
            
            _ensure_dir(os.path.dirname(dst_file_path))
            
            if resample_audio_file(src_file_path, dst_file_path, target_samplerate, audio_format):
                resampled_count +=1
            
    print(f"Processed {file_count} '.wav' files. Successfully resampled {resampled_count} files.")
    if file_count > resampled_count:
        print(f"Warning: {file_count - resampled_count} files could not be resampled.")
    print(f"Output directory: '{dst_dir}'. Only '.wav' files were processed and copied.")


def resample_corpus_inplace(src_dir: str, target_samplerate: int, audio_format: str = "wav"):
    """
    Recursively walks src_dir and resamples .wav files inplace to target_samplerate.
    Original files are overwritten.
    """
    print(f"Resampling files in-place in '{src_dir}' at {target_samplerate}Hz.")
    
    # Find all .wav files recursively
    audio_files = glob.glob(os.path.join(src_dir, '**/*.wav'), recursive=True)
    if not audio_files:
        print("No '.wav' files found to resample.")
        return

    resampled_count = 0
    with tqdm(total=len(audio_files), desc="Resampling audio files in-place") as pbar:
        for audio_file_path in audio_files:
            # For in-place, output_path is the same as input_path
            if resample_audio_file(audio_file_path, audio_file_path, target_samplerate, audio_format):
                resampled_count +=1
            pbar.update(1)
            
    print(f"Processed {len(audio_files)} '.wav' files. Successfully resampled {resampled_count} files in-place.")
    if len(audio_files) > resampled_count:
        print(f"Warning: {len(audio_files) - resampled_count} files could not be resampled.")

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
