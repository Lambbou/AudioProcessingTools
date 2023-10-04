import os
import glob
import librosa
import click
import soundfile as sf
from tqdm import tqdm

def resample_audio_files(directory, target_sr):
    # Check if the provided path is a directory
    if not os.path.isdir(directory):
        click.echo('Error: The provided path is not a directory.')
        return

    # Find all audio files in the directory
    audio_files = glob.glob(os.path.join(directory, '**/*.wav'), recursive=True)

    # Resample and save each audio file
    with tqdm(total=len(audio_files), desc="Resampling audio files") as pbar:
        for audio_file in audio_files:
            y, sr = librosa.load(audio_file, sr=target_sr)
            sf.write(audio_file, y, sr, 'PCM_16') # PCM_24
            pbar.update(1)

@click.command()
@click.argument('directory', type=click.Path(exists=True, file_okay=False, resolve_path=True))
@click.option('--target-sr', default=22050, help='Target sampling rate')
def main(directory, target_sr):
    resample_audio_files(directory, target_sr)

if __name__ == '__main__':
    main()

