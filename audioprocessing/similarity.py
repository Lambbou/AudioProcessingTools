import os
import csv
from glob import glob
from tqdm import tqdm
from pydub import AudioSegment # Keep for compute_similarity, though get_audio_duration is in io_utils
from scipy.spatial.distance import cosine, euclidean
from scipy.stats import bootstrap
import numpy as np

# The resemblyzer model/API
from resemblyzer import VoiceEncoder, preprocess_wav

# Functions from audioprocessing.io_utils
from .io_utils import get_audio_duration, export_dict_to_csv

# Initialize the VoiceEncoder model
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
Computes the similarity between a target audio file and a reference audio file.

Arguments:
    target (str): the path to the target audio file.
    reference (str): the path to the reference audio file.
    speaker_encoder_model (VoiceEncoder): The speaker encoder model to use.
    
Returns:
    tuple: (cosine_similarity, euclidean_distance) or None if an exception occurs.
"""
def compute_similarity(target:str, reference:str, speaker_encoder_model) -> tuple | None:
    try:
        # Compute speaker similarity
        ref_embed = get_embedding(reference, speaker_encoder_model)
        cloned_embed = get_embedding(target, speaker_encoder_model)
        cosine_similarity_score = compute_cosine_similarity(ref_embed, cloned_embed)
        euclidean_distance_score = euclidean(ref_embed, cloned_embed)
        return cosine_similarity_score, euclidean_distance_score
    except Exception as e:
        print(f"An exception occurred in compute_similarity: {e}")
        return None

"""
Processes a directory of audio files, computes the similarity for each file against a reference, 
and exports the results to a CSV file. (Refactored from process_csv in compute_cos_sim.py)
"""
def calculate_similarity_for_directory(ref_path: str, input_path: str, output_file: str, audio_type: str, speaker_encoder_model):
    # Check if the input is an existing path
    if not os.path.isdir(input_path):
        print('Error: Input path must be an existing directory.')
        return

    # Check if the output file already exists
    if os.path.isfile(output_file):
        print('Error: Output file already exists.')
        return

    results = {}
    for path in tqdm(sorted(glob(os.path.join(input_path, '**/*.' + audio_type), recursive=True))):
        duration = get_audio_duration(path) # Using imported function
        similarity_scores = compute_similarity(path, ref_path, speaker_encoder_model)
        if similarity_scores:
            # Storing only cosine similarity for now, as export_dict_to_csv expects one score
            # TODO: Update export_dict_to_csv or results structure if more scores are needed in CSV
            results[path] = [duration, similarity_scores[0]] 
        else:
            results[path] = [duration, "Error"]


    # Perform operations on the CSV file and generate output
    print(f'Generating output file: {output_file}...\r')
    # Using imported function. Note: header in export_dict_to_csv is ["Path", "Duration", "MOS"]
    # We are writing similarity score under "MOS" column.
    export_dict_to_csv(results, output_file)
    print("done.")

"""
Computes Cosine Similarity for a given set of audio files with a complex directory structure,
calculates statistics, and exports them.
(Refactored from resemblyzer_inference_with_different_speakers.py)
"""
def calculate_similarity_for_speaker_directory_structure(
    data_dir: str, 
    ref_dir: str, 
    output_csv_file: str, 
    output_log_file: str, 
    output_model_stats_file: str, 
    output_speaker_stats_file: str, 
    speaker_encoder_model):

    with open(output_csv_file, 'w') as outfi, \
         open(output_log_file, 'w') as log_outfi, \
         open(output_speaker_stats_file, 'w') as stats_speaker_outfi, \
         open(output_model_stats_file, 'w') as stats_model_outfi:
        
        fieldnames = ['model', 'speaker', 'cloned_wav','ref', 'cosine_similarity', 'euclidean_similarity']
        writer = csv.DictWriter(outfi, fieldnames=fieldnames)
        writer.writeheader()

        dict_stats_speaker = {}
        dict_stats_model = {}
        
        for model_name in os.listdir(data_dir):
            if not os.path.isdir(os.path.join(data_dir, model_name)):
                continue
            dict_stats_speaker[model_name] = {}
            path_model_name = os.path.join(data_dir, model_name)
            log_outfi.write(f"Model : {model_name}\n")
            mod_cosine_similarity_list, mod_euclidean_distance_list = [], []
            
            for speaker in os.listdir(path_model_name):
                if not os.path.isdir(os.path.join(path_model_name, speaker)):
                    continue
                log_outfi.write(f"Speaker : {speaker}\n")
                path_model_speaker = os.path.join(path_model_name, speaker)
                sp_cosine_similarity_list, sp_euclidean_distance_list = [], []

                for cloned_wav_filename in os.listdir(path_model_speaker):
                    if cloned_wav_filename.endswith(".wav"):
                        cloned_wav_path = os.path.join(path_model_speaker, cloned_wav_filename)
                        
                        speaker_underscore = speaker + "_"
                        sample_raw_name = cloned_wav_filename.split(speaker_underscore)[-1] if speaker_underscore in cloned_wav_filename else cloned_wav_filename
                        sample_raw_name = sample_raw_name.split("_synthesis")[0] + ".wav" # Heuristic from original script
                        
                        ref_speaker_dir = os.path.join(ref_dir, speaker)
                        ref_wav_path = os.path.join(ref_speaker_dir, sample_raw_name)

                        if not os.path.exists(ref_wav_path):
                            log_outfi.write(f"WARNING: Reference file not found: {ref_wav_path} for cloned: {cloned_wav_path}\n")
                            continue

                        try:
                            similarity_scores = compute_similarity(cloned_wav_path, ref_wav_path, speaker_encoder_model)
                            if similarity_scores:
                                cosine_similarity_score, euclidean_distance_score = similarity_scores
                            else:
                                cosine_similarity_score, euclidean_distance_score = "Error", "Error"
                        except Exception as e:
                            print(f"Error processing {cloned_wav_path} and {ref_wav_path}: {e}")
                            cosine_similarity_score, euclidean_distance_score = "Exception", "Exception"
                        
                        writer.writerow({
                            'model': model_name, 
                            'speaker': speaker, 
                            'cloned_wav': cloned_wav_path, 
                            'ref': ref_wav_path, 
                            'cosine_similarity': cosine_similarity_score, 
                            'euclidean_similarity': euclidean_distance_score
                        })

                        if isinstance(cosine_similarity_score, (float, int)):
                            sp_cosine_similarity_list.append(cosine_similarity_score)
                            mod_cosine_similarity_list.append(cosine_similarity_score)
                
                if sp_cosine_similarity_list:
                    cosine_similarity_mean, cosine_similarity_std = np.mean(sp_cosine_similarity_list), np.std(sp_cosine_similarity_list)
                    log_outfi.write(f"Speaker mean cosine similarity = {cosine_similarity_mean}\n")
                    confidence_interval = bootstrap((sp_cosine_similarity_list,), np.mean, confidence_level=0.95, random_state=np.random.default_rng())
                    log_outfi.write(f"Lower confidence interval = {confidence_interval.confidence_interval.low}\n")
                    log_outfi.write(f"Higher confidence interval = {confidence_interval.confidence_interval.high}\n")
                    ci_range = (confidence_interval.confidence_interval.high - confidence_interval.confidence_interval.low) / 2
                    log_outfi.write(f"Confidence +/- = {ci_range}\n")
                    dict_stats_speaker[model_name][speaker] = f"{cosine_similarity_mean} +/- {ci_range}"
                else:
                    log_outfi.write(f"No similarity scores for speaker {speaker} in model {model_name}\n")


            if mod_cosine_similarity_list:
                cosine_similarity_mean, cosine_similarity_std = np.mean(mod_cosine_similarity_list), np.std(mod_cosine_similarity_list)
                log_outfi.write(f"Model mean cosine similarity = {cosine_similarity_mean}\n")
                confidence_interval = bootstrap((mod_cosine_similarity_list,), np.mean, confidence_level=0.95, random_state=np.random.default_rng())
                log_outfi.write(f"Lower confidence interval = {confidence_interval.confidence_interval.low}\n")
                log_outfi.write(f"Higher confidence interval = {confidence_interval.confidence_interval.high}\n")
                ci_range = (confidence_interval.confidence_interval.high - confidence_interval.confidence_interval.low) / 2
                log_outfi.write(f"Confidence +/- = {ci_range}\n")
                dict_stats_model[model_name] = f"{cosine_similarity_mean} +/- {ci_range}"
            else:
                log_outfi.write(f"No similarity scores for model {model_name}\n")

        fieldnames_speaker = ['Model', 'Speaker', 'Mean Cosine Similarity']
        writer_stats_speaker = csv.DictWriter(stats_speaker_outfi, fieldnames=fieldnames_speaker)
        writer_stats_speaker.writeheader()
        for model_name_key in dict_stats_speaker:
            for speaker_key in dict_stats_speaker[model_name_key]:
                writer_stats_speaker.writerow({'Model': model_name_key, 'Speaker': speaker_key, 'Mean Cosine Similarity': dict_stats_speaker[model_name_key][speaker_key]})

        fieldnames_model = ['Model', 'Mean Cosine Similarity']
        writer_stats_model = csv.DictWriter(stats_model_outfi, fieldnames=fieldnames_model)
        writer_stats_model.writeheader()
        for model_name_key in dict_stats_model:
            writer_stats_model.writerow({'Model': model_name_key, 'Mean Cosine Similarity': dict_stats_model[model_name_key]})
        
    print("Similarity calculation with speaker directory structure complete.")
