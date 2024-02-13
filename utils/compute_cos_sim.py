import os
import csv
import click
from glob import glob
from tqdm import tqdm
from pydub import AudioSegment

# The code for computing the Cosine Similarity
from scipy.spatial.distance import cosine, euclidean
from scipy.stats import bootstrap

# The resemblizer model/API
from resemblyzer import VoiceEncoder, preprocess_wav
speaker_encoder = VoiceEncoder()

# Computes the embedding for file filepath using model and returns it.
# Arguments: 
#   filepath (str): the path to the wav file with the embedding to compute.
#   model (): the model to use for computing the embedding. 
# Returns: 
#   a [tensor], the extracted embedding.
def get_embedding(filepath:str, model):
    ref_wav = preprocess_wav(filepath)
    ref_embed = model.embed_utterance(ref_wav)
    return ref_embed


def compute_cosine_similarity(x, y):
    return 1 - cosine(x, y)


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


def compute_similarity(target:str, reference:str) -> float:
    try:
        # Compute speaker similarity
        ref_embed = get_embedding(reference, speaker_encoder)
        cloned_embed = get_embedding(target, speaker_encoder)
        cosine_similarity = compute_cosine_similarity(ref_embed, cloned_embed)
        euclidean_distance = euclidean(ref_embed, cloned_embed)
    except Exception as e:
        print("An exception occured:")
        exit(e)

@click.command()
@click.argument('ref_path', type=click.Path(exists=True, dir_okay=True)) 
@click.argument('input_path', type=click.Path(exists=True, dir_okay=True)) # !!! On suppose que tous les audios sont du même système/modèle et locuteur
@click.argument('output_file', type=click.Path(dir_okay=False))
@click.option('-t', '--type', type=click.STRING, default="wav", help='The type of audio file to look for (flac, wav, etc).')
def process_csv(ref_path, input_path, output_file, type):
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
        similarity = compute_similarity(path, reference)
        results[path] = [duration, similarity]

    # Perform operations on the CSV file and generate output
    click.echo(f'Generating output file: {output_file}...\r')
    export_dict_to_csv(results, output_file)
    click.echo("done.")

if __name__ == '__main__':
    process_csv()

