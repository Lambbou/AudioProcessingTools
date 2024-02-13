"""
This module provides functions for computing cosine similarity between audio files and exporting the results to a CSV file.

Functions:
- get_embedding(filepath:str, model): Computes the embedding for a given audio file using a specified model.
- compute_cosine_similarity(x, y): Computes the cosine similarity between two embeddings.
- export_dict_to_csv(dictionary, csv_path): Exports a dictionary to a CSV file.
- get_audio_duration(path:str) -> int: Returns the duration of an audio file in milliseconds.
- compute_similarity(target:str, reference:str) -> float: Computes the similarity between a target audio file and a reference audio file.
- process_csv(ref_path, input_path, output_file, type): Processes a directory of audio files, computes the similarity for each file, and exports the results to a CSV file.

Example usage:
    python compute_cos_sim.py /path/to/reference /path/to/input /path/to/output.csv -t wav
"""


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

"""
Computes the embedding for file filepath using model and returns it.

Arguments: 
   filepath (str): the path to the wav file with the embedding to compute.
   model (VoiceEncoder): the model to use for computing the embedding. 

Returns: 
   tensor, the extracted embedding.
"""
def get_embedding(filepath:str, model):
    ref_wav = preprocess_wav(filepath)
    ref_embed = model.embed_utterance(ref_wav)
    return ref_embed

"""
Computes the cosine similarity between two embeddings x and y.

Arguments:
    x (tensor): the first embedding.
    y (tensor): the second  embedding.
    
Returns:   
    float: the cosine similarity between x and y.
"""
def compute_cosine_similarity(x, y):
    return 1 - cosine(x, y)

"""
Exports a dictionary to a CSV file.

Arguments:
    dictionary (dict): the dictionary to export.
    csv_path (str): the path to the CSV file to create.
"""
def export_dict_to_csv(dictionary, csv_path):
    with open(csv_path, 'w') as ofile:
        writer = csv.writer(ofile, delimiter='\t')
        
        # Header
        writer.writerow(["Path", "Duration", "MOS"])
        
        for key, value in dictionary.items():
            writer.writerow([key, value[0], value[1]])


"""
Returns the duration of an audio file in milliseconds.

Arguments:
    path (str): the path to the audio file.

Returns:
    int: the duration of the audio file in milliseconds.
"""
def get_audio_duration(path:str) -> int:
    # returns the duration of an audio file in ms
    audio = AudioSegment.from_file(path)
    return len(audio) # returns ms - use audio.duration_seconds to get the duration in seconds

"""
Computes the similarity between a target audio file and a reference audio file.

Arguments:
    target (str): the path to the target audio file.
    reference (str): the path to the reference audio file.
    
Returns:
    float: the similarity between the target and reference audio files.
"""
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

