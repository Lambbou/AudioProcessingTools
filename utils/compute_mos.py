import os
import csv
from glob import glob
from tqdm import tqdm
from pydub import AudioSegment

from wvmos import get_wvmos
model = get_wvmos(cuda=True) # Update if necessary

import click
import os

def export_dict_to_csv(dictionary, csv_path):
    with open(csv_path, 'w') as ofile:
        writer = csv.writer(ofile, delimiter='\t')
        
        # Header
        writer.writerow(["Path", "Duration", "MOS"])
        
        for key, value in dictionary.items():
            writer.writerow([key, value[0], value[1]])

def get_audio_duration(path:str) -> int:
    # returns the duration of an audio file in ms
    audio = AudioSegment.from_file(path)
    return len(audio) # returns ms - use audio.duration_seconds to get the duration in seconds

@click.command()
@click.argument('input_path', type=click.Path(exists=True, dir_okay=True))
@click.argument('output_file', type=click.Path(dir_okay=False))
@click.option('-t', '--type', type=click.STRING, default="wav", help='The type of audio file to look for (flac, wav, etc).')
def process_csv(input_path, output_file, type):

    # Check if the input is an existing path
    if not os.path.isdir(input_path):
        click.echo('Error: Input path must be an existing directory.')
        return

    # Check if the output file already exists
    if os.path.isfile(output_file):
        click.echo('Error: Output file already exists.')
        return

    results = {}
    for path in tqdm(sorted(glob(os.path.join(input_path, '**/*.'+type), recursive=True))):
        duration = get_audio_duration(path)
        mos = model.calculate_one(path)
        results[path] = [duration, mos]

    # Perform operations on the CSV file and generate output
    click.echo(f'Generating output file: {output_file}...\r')
    export_dict_to_csv(results, output_file)
    click.echo("done.")

if __name__ == '__main__':
    process_csv()

