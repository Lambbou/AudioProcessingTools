import os
import click
from pydub import AudioSegment

class NegativeNumberParamType(click.ParamType):
    name = 'negative_numbers_only'

    def convert(self, value, param, ctx):
        try:
            number = float(value)
            if number < 0:
                return number
            else:
                self.fail(f'{value} is not a negative number. Values passed in dB LUFS cannot be positive.', param, ctx)
        except ValueError:
            self.fail(f'{value} is not a valid number. Values passed in dB LUFS should be, well... numerical values.', param, ctx)

def normalize_audio(src_dir:str, dst_dir:str, target_loudness:int):
    try:
        # Recursively create the same directory structure in the target directory
        for root, _, files in os.walk(src_dir):
            for file in files:
                src_file_path = os.path.join(root, file)
                relative_path = os.path.relpath(src_file_path, src_dir)
                dst_file_path = os.path.join(dst_dir, relative_path)

                # Create the target directory if it doesn't exist
                os.makedirs(os.path.dirname(dst_file_path), exist_ok=True)

                # Load the audio file and normalize to -23dB LUFS
                audio = AudioSegment.from_file(src_file_path)
                loudness = audio.dBFS
                target_lufs = target_loudness  # Target LUFS level

                if loudness is not None and loudness != target_lufs:
                    # Calculate the gain required to reach the target LUFS level
                    gain = target_lufs - loudness
                    audio = audio + gain  # Apply the gain to normalize

                # Export the normalized audio to the target directory
                audio.export(dst_file_path, format="wav")

        print(f"Directory structure replicated and audio files normalized from '{src_dir}' to '{dst_dir}'.")
        print("WARNING ! Although the directory structure was copied, ONLY THE WAV FILES were copied.")
    except Exception as e:
        print(f"Error: {str(e)}")

@click.command()
@click.argument('src_directory', type=click.Path(exists=True))
@click.argument('dst_directory', type=click.Path())
@click.option('--db', type=NegativeNumberParamType(), default=-23, help='The desired loudness in dB LUFS (Loudness Units Full Scale).')
def normalize_and_replicate(src_directory, dst_directory, db):
    normalize_audio(src_directory, dst_directory, db)

if __name__ == '__main__':
    normalize_and_replicate()

