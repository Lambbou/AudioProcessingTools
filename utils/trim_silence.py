import os
import click
from pydub import AudioSegment
from pydub.silence import split_on_silence

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

def trim_audio_file(input_wav:str, output_wav:str, threshold:int):
    audio = AudioSegment.from_wav(input_wav)

    audio_segments = split_on_silence(audio, silence_thresh=threshold)

    trimmed_audio = AudioSegment.empty()
    for segment in audio_segments:
        trimmed_audio += segment
        
    if len(audio) < len(trimmed_audio):
        # If data was trimmed, we add small 50ms silences before and after the signal 
        trimmed_audio = AudioSegment.silent(duration=50) + trimmed_audio + AudioSegment.silent(duration=50)

    trimmed_audio.export(output_wav, format="wav")
    
    return len(audio), len(trimmed_audio)

def trim_and_replicate(src_dir, dst_dir, threshold:int):
    try:
        # Recursively create the same directory structure in the target directory
        for root, _, files in os.walk(src_dir):
            for file in files:
                if not file.endswith(".wav"):
                    continue
                src_file_path = os.path.join(root, file)
                relative_path = os.path.relpath(src_file_path, src_dir)
                dst_file_path = os.path.join(dst_dir, relative_path)

                # Create the target directory if it doesn't exist
                os.makedirs(os.path.dirname(dst_file_path), exist_ok=True)

                lenght, length_trimmed = trim_audio_file(src_file_path, dst_file_path, threshold)

                # Create a corresponding empty .txt file
                txt_file_path = os.path.splitext(dst_file_path)[0] + ".txt"
                
                # Add annotations
                with open(txt_file_path, 'w') as txt_file:
                    txt_file.write("lenght,trimmed_file,duration_trimmed_portion\n")
                    txt_file.write(f"{lenght},{length_trimmed},{lenght - length_trimmed}\n")

        print(f"Directory structure and empty .txt files replicated from '{src_dir}' to '{dst_dir}'. All wav files have been trimmed.")
        print("WARNING ! Although the directory structure was copied, ONLY THE WAV FILES were copied.")
    except Exception as e:
        print(f"Error: {str(e)}")

@click.command()
@click.argument('src_directory', type=click.Path(exists=True))
@click.argument('dst_directory', type=click.Path())
@click.option('--db', type=NegativeNumberParamType(), default=-40, help='The desired loudness threshold in dB LUFS (Loudness Units Full Scale) to identify silent areas.')
def trim_and_replicate_structure(src_directory, dst_directory, db):
    trim_and_replicate(src_directory, dst_directory, db)

if __name__ == '__main__':
    trim_and_replicate_structure()
