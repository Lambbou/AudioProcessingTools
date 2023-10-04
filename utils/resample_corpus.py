import os
import click
import librosa
import soundfile as sf

def resample_audio(src_dir:str, dst_dir:str, target_sr:int):
    try:
        # Recursively create the same directory structure in the target directory
        for root, _, files in os.walk(src_dir):
            for file in files:
                src_file_path = os.path.join(root, file)
                relative_path = os.path.relpath(src_file_path, src_dir)
                dst_file_path = os.path.join(dst_dir, relative_path)

                # Create the target directory if it doesn't exist
                os.makedirs(os.path.dirname(dst_file_path), exist_ok=True)

                audio, sr = librosa.load(src_file_path, sr=target_sr)
                sf.write(dst_file_path, audio, sr, 'PCM_16')

        print(f"Directory structure replicated and audio files normalized from '{src_dir}' to '{dst_dir}'.")
        print("WARNING ! Although the directory structure was copied, ONLY THE WAV FILES were copied.")
    except Exception as e:
        print(f"Error: {str(e)}")

@click.command()
@click.argument('src_directory', type=click.Path(exists=True))
@click.argument('dst_directory', type=click.Path())
@click.option('--rate', type=click.INT, default=22050, help='The desired sampling rate.')
def resample_and_replicate(src_directory, dst_directory, rate):
    resample_audio(src_directory, dst_directory, rate)

if __name__ == '__main__':
    resample_and_replicate()
