import os
import csv
from glob import glob
from tqdm import tqdm
from pydub import AudioSegment # Keep for compute_similarity, though get_audio_duration is in io_utils
from scipy.spatial.distance import cosine, euclidean
from scipy.stats import bootstrap
import numpy as np

"""Audio similarity computation utilities.

This module provides tools for computing similarity between audio files,
primarily focusing on speaker similarity using embeddings generated by the
Resemblyzer library's VoiceEncoder model. It includes functions to:
- Generate speaker embeddings for audio files.
- Calculate cosine similarity between embeddings.
- Process directories of audio files to compute similarity against a reference,
  or within a structured directory for more complex comparisons involving
  multiple speakers and models, including statistical analysis of results.
"""
# The resemblyzer model/API
from resemblyzer import VoiceEncoder, preprocess_wav
from resemblyzer.audio import AudioProcessError # For specific resemblyzer exceptions

# Functions from audioprocessing.io_utils
from .io_utils import get_audio_duration, export_dict_to_csv

# Initialize the VoiceEncoder model globally.
# This instance will be used by default in functions that require a speaker encoder.
# Resemblyzer's VoiceEncoder loads its model weights upon initialization.
# It's generally recommended to initialize it once and reuse the instance for efficiency.
speaker_encoder = VoiceEncoder()


# Note: The original script had comment blocks for docstrings.
# These will be replaced with proper Google-style docstrings.


def get_embedding(filepath: str, model: VoiceEncoder) -> np.ndarray:
    """Computes and returns the speaker embedding for an audio file.

    This function uses the provided Resemblyzer VoiceEncoder model to extract
    a speaker embedding (a high-dimensional vector representing speaker
    characteristics) from the given audio file. The audio file is first
    preprocessed by `preprocess_wav` from Resemblyzer.

    Args:
        filepath (str): The path to the WAV audio file.
        model (VoiceEncoder): An instance of `resemblyzer.VoiceEncoder` to be
                              used for generating the embedding.

    Returns:
        np.ndarray: A NumPy array representing the speaker embedding.

    Raises:
        FileNotFoundError: If the `filepath` does not point to an existing file.
                           (This is raised by `preprocess_wav` if the file is not found).
        AudioProcessError: If Resemblyzer encounters an issue processing the audio,
                           such as the audio being too short or in an invalid/unsupported format.
                           This exception is from `resemblyzer.audio`.
        Exception: Other exceptions may be raised by underlying libraries if the
                   audio file is severely corrupted or unreadable.
    """
    # preprocess_wav loads, resamples (if needed), and normalizes the audio.
    # It can raise FileNotFoundError or AudioProcessError.
    ref_wav = preprocess_wav(filepath)
    # embed_utterance generates the speaker embedding from the preprocessed waveform.
    ref_embed = model.embed_utterance(ref_wav)
    return ref_embed

"""
Computes the cosine similarity between two embeddings x and y.
    ref_wav = preprocess_wav(filepath)
    ref_embed = model.embed_utterance(ref_wav)
    return ref_embed

"""
Computes the cosine similarity between two embeddings x and y.

Arguments:
    x (tensor): the first embedding. # These types will be updated to np.ndarray
    y (tensor): the second  embedding.
    
Returns:   
    float: the cosine similarity between x and y.
"""
def compute_cosine_similarity(x: np.ndarray, y: np.ndarray) -> float:
    """Computes the cosine similarity between two embedding vectors.

    Cosine similarity measures the cosine of the angle between two non-zero
    vectors, providing a measure of their orientation similarity. A value of 1
    means the vectors are identical in orientation, 0 means they are orthogonal,
    and -1 means they are diametrically opposed.

    The formula used is `1 - scipy.spatial.distance.cosine(x, y)`.
    `scipy.spatial.distance.cosine` computes the cosine distance, which is
    `1 - cosine_similarity`.

    Args:
        x (np.ndarray): The first embedding vector (e.g., a speaker embedding).
        y (np.ndarray): The second embedding vector.

    Returns:
        float: The cosine similarity score between vectors `x` and `y`.
               The score ranges from -1.0 to 1.0.

    Raises:
        ValueError: If `x` or `y` are not valid 1-D arrays or if they are
                    all-zero vectors (though `scipy.spatial.distance.cosine`
                    might handle some of these cases by returning specific values
                    like NaN or raising its own errors).
    """
    return 1 - cosine(x, y)

"""
Computes the similarity between a target audio file and a reference audio file.

Arguments:
    target (str): the path to the target audio file.
    reference (str): the path to the reference audio file.
    speaker_encoder_model (VoiceEncoder): The speaker encoder model to use.
    
Returns:
    tuple: (cosine_similarity_score, euclidean_distance_score) or None if an exception occurs.
"""
def compute_similarity(
    target: str, 
    reference: str, 
    speaker_encoder_model: VoiceEncoder
    ) -> tuple[float, float] | None:
    """Computes speaker similarity (cosine and Euclidean) between two audio files.

    This function generates speaker embeddings for both the `target` and `reference`
    audio files using the provided `speaker_encoder_model`. It then calculates
    the cosine similarity and Euclidean distance between these two embeddings.

    Args:
        target (str): Path to the target audio file.
        reference (str): Path to the reference audio file.
        speaker_encoder_model (VoiceEncoder): The Resemblyzer VoiceEncoder instance
                                              to use for generating embeddings.

    Returns:
        tuple[float, float] | None: A tuple containing the cosine similarity score
            (ranging from -1.0 to 1.0) and the Euclidean distance (non-negative float).
            Returns `None` if any exception occurs during embedding generation
            or similarity calculation.

    Raises:
        This function catches exceptions from `get_embedding` (like `FileNotFoundError`,
        `AudioProcessError`) and prints an error message, returning `None` instead
        of re-raising.
    """
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


def calculate_similarity_for_directory(
    ref_path: str, 
    input_path: str, 
    output_file: str, 
    audio_type: str, 
    speaker_encoder_model: VoiceEncoder
    ) -> None:
    """Calculates and exports speaker similarity for a directory of audio files against a reference.

    This function iterates through all audio files of the specified `audio_type`
    (case-insensitive) within the `input_path` directory. For each file, it
    computes the speaker similarity (cosine similarity) against a single
    `ref_path` audio file using the provided `speaker_encoder_model`.

    The results, including the path of each processed file, its duration, and the
    computed cosine similarity score, are saved to a tab-separated CSV file
    specified by `output_file`. The CSV uses the header "Path", "Duration", "MOS"
    (where "MOS" column here stores the similarity score, reusing the
    `export_dict_to_csv` function's default header).

    Args:
        ref_path (str): Path to the single reference audio file.
        input_path (str): Path to the directory containing audio files to be processed.
        output_file (str): Path to save the output CSV file.
        audio_type (str): The file extension (without the dot) of audio files
                          to process (e.g., "wav", "flac").
        speaker_encoder_model (VoiceEncoder): The Resemblyzer VoiceEncoder instance
                                              to use for generating embeddings.

    Returns:
        None. Results are written to `output_file`.

    Raises:
        FileNotFoundError: If `input_path` does not exist or is not a directory.
                           (This is checked explicitly).
                           Also, if `ref_path` or any audio file in `input_path`
                           is not found (implicitly via `get_audio_duration` or
                           `compute_similarity`).
        Exception: Catches and prints general exceptions during processing of individual
                   files (e.g., from `get_audio_duration` or `compute_similarity`),
                   allowing the batch process to continue. The `output_file` might
                   not be created if `input_path` is invalid.
    """
    # Validate input_path and output_file conditions
    if not os.path.isdir(input_path):
        # Raise FileNotFoundError for consistency with other file/dir operations
        raise FileNotFoundError(f"Error: Input path '{input_path}' must be an existing directory.")

    if os.path.isfile(output_file):
        # This check is more of a CLI convenience; for library use, overwriting is often acceptable.
        # Consider removing or making it a warning for library use if CLI handles confirmation.
        print(f"Warning: Output file '{output_file}' already exists and will be overwritten.")
        # Depending on desired library behavior, could raise FileExistsError or return.

    results = {}
    # Glob pattern for case-insensitive search, recursively.
    glob_pattern = os.path.join(input_path, '**', f'*.{audio_type.lower()}')
    file_paths = sorted(glob.glob(glob_pattern, recursive=True)) # Sorted for deterministic output

    if not file_paths:
        print(f"No '.{audio_type}' files found in '{input_path}'. No output file will be created.")
        return

    print(f"Calculating similarity for {len(file_paths)} '.{audio_type}' files in '{input_path}' against reference '{ref_path}'...")
    for path in tqdm(file_paths, desc="Calculating similarity"):
        try:
            duration_ms = get_audio_duration(path)
        except Exception as e:
            print(f"Error getting duration for {path}: {e}. Marking duration as 'Error'.")
            duration_ms = "Error: Duration unreadable"
        
        similarity_scores = compute_similarity(path, ref_path, speaker_encoder_model)
        
        if similarity_scores:
            # Storing only cosine similarity, as export_dict_to_csv by default expects
            # a single value for the third column ("MOS" or score).
            # If Euclidean distance or other scores are needed, export_dict_to_csv
            # would need adjustment (e.g., more flexible headers/value lists).
            results[path] = [duration_ms, similarity_scores[0]] 
        else:
            # compute_similarity prints its own errors. Mark score as "Error".
            results[path] = [duration_ms, "Error: Similarity calculation failed"]

    # Export results. The default header in export_dict_to_csv is ["Path", "Duration", "MOS"].
    # Here, the "MOS" column will contain the cosine similarity score.
    print(f'\nGenerating output CSV file: {output_file}...')
    try:
        export_dict_to_csv(results, output_file) # Uses default header: ["Path", "Duration", "MOS"]
        print(f"Similarity calculation and CSV export complete. Output saved to '{output_file}'.")
    except Exception as e:
        print(f"Error exporting results to CSV '{output_file}': {e}")


def calculate_similarity_for_speaker_directory_structure(
    data_dir: str, 
    ref_dir: str, 
    output_csv_file: str, 
    output_log_file: str, 
    output_model_stats_file: str, 
    output_speaker_stats_file: str, 
    speaker_encoder_model: VoiceEncoder
    ) -> None:
    """Calculates and exports speaker similarity for a structured directory of audio files.

    This function is designed for scenarios where audio files are organized by
    model and speaker (e.g., for evaluating Text-to-Speech models). It expects
    synthesized/cloned audio files in `data_dir` with a structure like
    `data_dir/model_name/speaker_name/audio_file.wav`. Reference audio files
    for each speaker are expected in `ref_dir` with a structure like
    `ref_dir/speaker_name/reference_audio.wav`.

    The function performs the following:
    1.  Iterates through each model and speaker in `data_dir`.
    2.  For each synthesized/cloned audio file, it attempts to find a corresponding
        reference audio file in `ref_dir` based on speaker name and filename parsing
        (heuristics from the original script are maintained for filename matching).
    3.  Calculates cosine similarity and Euclidean distance between the synthesized
        and reference audio embeddings using the provided `speaker_encoder_model`.
    4.  Writes detailed results (model, speaker, file paths, similarity scores)
        to `output_csv_file`.
    5.  Calculates and logs statistics (mean cosine similarity, 95% confidence
        intervals using bootstrapping) per speaker and per model to `output_log_file`.
    6.  Writes aggregated statistics per speaker to `output_speaker_stats_file`.
    7.  Writes aggregated statistics per model to `output_model_stats_file`.

    Args:
        data_dir (str): Path to the directory containing synthesized/cloned audio
                        files, structured as `model_name/speaker_name/`.
        ref_dir (str): Path to the directory containing reference audio files,
                       structured as `speaker_name/`.
        output_csv_file (str): Path to save the main CSV file with detailed
                               similarity results for every file pair.
        output_log_file (str): Path to save a log file containing detailed
                               statistical analysis (means, std, confidence intervals)
                               for each speaker and model.
        output_model_stats_file (str): Path to save a CSV file with aggregated
                                       mean cosine similarity statistics per model.
        output_speaker_stats_file (str): Path to save a CSV file with aggregated
                                         mean cosine similarity statistics per speaker
                                         (for each model).
        speaker_encoder_model (VoiceEncoder): The Resemblyzer VoiceEncoder instance
                                              to use for generating embeddings.

    Returns:
        None. All results are written to the specified output files.

    Raises:
        FileNotFoundError: If `data_dir` or `ref_dir` do not exist.
        OSError: For issues related to file I/O when creating output files.
        Exception: Catches general exceptions during processing of individual files
                   or during statistical calculations, printing error messages to
                   console and log file where appropriate. Processing attempts to
                   continue for other files/speakers/models if an error occurs.
    """
    # Ensure output directories exist for the stat files if they are in subdirs (though typically not)
    # For simplicity, assuming output files are in writable locations.
    # More robust error handling for output file paths could be added if needed.

    # Open all output files at once to ensure they are writable at the start.
    # Using newline='' for CSV writers to prevent blank rows on Windows.
    # Using utf-8 encoding for broader compatibility.
    try:
        with open(output_csv_file, 'w', newline='', encoding='utf-8') as outfi, \
             open(output_log_file, 'w', encoding='utf-8') as log_outfi, \
             open(output_speaker_stats_file, 'w', newline='', encoding='utf-8') as stats_speaker_outfi, \
             open(output_model_stats_file, 'w', newline='', encoding='utf-8') as stats_model_outfi:
            
            # Setup for the main detailed CSV output
            detailed_fieldnames = ['model', 'speaker', 'cloned_wav', 'ref', 'cosine_similarity', 'euclidean_similarity']
            detailed_writer = csv.DictWriter(outfi, fieldnames=detailed_fieldnames)
            detailed_writer.writeheader()

            # Dictionaries to store lists of scores for statistical calculations
            # model_scores_for_stats: {model_name: [score1, score2, ...]}
            # speaker_scores_for_stats: {model_name: {speaker_name: [score1, score2, ...]}}
            speaker_scores_for_stats = {} 
            model_scores_for_stats = {}

            # Iterate through models in the data directory
            for model_name in os.listdir(data_dir):
                path_model_name = os.path.join(data_dir, model_name)
                if not os.path.isdir(path_model_name):
                    log_outfi.write(f"Skipping '{model_name}': not a directory.\n")
                    continue
                
                log_outfi.write(f"Processing Model: {model_name}\n")
                log_outfi.write("=" * (len(model_name) + 20) + "\n") # Header for model in log
                
                # Initialize lists for current model's scores
                model_scores_for_stats[model_name] = []
                speaker_scores_for_stats[model_name] = {}

                # Iterate through speakers for the current model
                for speaker in os.listdir(path_model_name):
                    path_model_speaker = os.path.join(path_model_name, speaker)
                    if not os.path.isdir(path_model_speaker):
                        log_outfi.write(f"  Skipping Speaker '{speaker}' under model '{model_name}': not a directory.\n")
                        continue

                    log_outfi.write(f"  Processing Speaker: {speaker}\n")
                    speaker_scores_for_stats[model_name][speaker] = []
                    
                    # Iterate through WAV files for the current speaker and model
                    for cloned_wav_filename in os.listdir(path_model_speaker):
                        if not cloned_wav_filename.lower().endswith(".wav"):
                            continue # Process only .wav files

                        cloned_wav_path = os.path.join(path_model_speaker, cloned_wav_filename)
                        
                        # Heuristic for finding the corresponding reference file name
                        # (from original script: utils/resemblyzer_inference_with_different_speakers.py)
                        speaker_underscore = speaker + "_"
                        if speaker_underscore in cloned_wav_filename:
                            sample_raw_name = cloned_wav_filename.split(speaker_underscore)[-1]
                        else:
                            sample_raw_name = cloned_wav_filename # Fallback if speaker_ is not in filename
                        sample_raw_name = sample_raw_name.split("_synthesis")[0] + ".wav" 
                        
                        ref_speaker_dir = os.path.join(ref_dir, speaker)
                        ref_wav_path = os.path.join(ref_speaker_dir, sample_raw_name)

                        if not os.path.exists(ref_wav_path):
                            warning_msg = f"    WARNING: Reference file not found: '{ref_wav_path}' for cloned file: '{cloned_wav_path}'. Skipping.\n"
                            print(warning_msg.strip()) # Also print to console for immediate visibility
                            log_outfi.write(warning_msg)
                            continue

                        # Compute similarity
                        cosine_sim_score, euclidean_dist_score = "Error", "Error" # Default if calculation fails
                        try:
                            similarity_scores = compute_similarity(cloned_wav_path, ref_wav_path, speaker_encoder_model)
                            if similarity_scores:
                                cosine_sim_score, euclidean_dist_score = similarity_scores
                                # Store valid cosine similarity for statistics
                                if isinstance(cosine_sim_score, (float, int)):
                                    speaker_scores_for_stats[model_name][speaker].append(cosine_sim_score)
                                    model_scores_for_stats[model_name].append(cosine_sim_score)
                            # If similarity_scores is None, error message already printed by compute_similarity
                        
                        except Exception as e: # Catch any unexpected error from compute_similarity itself
                            error_msg = f"    ERROR: Exception during similarity computation for '{cloned_wav_path}' and '{ref_wav_path}': {e}\n"
                            print(error_msg.strip())
                            log_outfi.write(error_msg)
                        
                        # Write detailed result to the main CSV
                        detailed_writer.writerow({
                            'model': model_name, 
                            'speaker': speaker, 
                            'cloned_wav': cloned_wav_path, 
                            'ref': ref_wav_path, 
                            'cosine_similarity': cosine_sim_score, 
                            'euclidean_similarity': euclidean_dist_score
                        })
                    
                    # Calculate and log statistics for the current speaker
                    current_speaker_scores = speaker_scores_for_stats[model_name][speaker]
                    if current_speaker_scores:
                        mean_cos_sim = np.mean(current_speaker_scores)
                        std_cos_sim = np.std(current_speaker_scores)
                        # Calculate 95% confidence interval for the mean using bootstrapping
                        # np.random.default_rng() is preferred for newer NumPy versions for random state
                        confidence_interval = bootstrap((current_speaker_scores,), np.mean, confidence_level=0.95, random_state=np.random.default_rng())
                        ci_low = confidence_interval.confidence_interval.low
                        ci_high = confidence_interval.confidence_interval.high
                        ci_range_half = (ci_high - ci_low) / 2
                        
                        log_outfi.write(f"    Speaker '{speaker}' Mean Cosine Similarity: {mean_cos_sim:.4f}\n")
                        log_outfi.write(f"    Speaker '{speaker}' Std Dev Cosine Similarity: {std_cos_sim:.4f}\n")
                        log_outfi.write(f"    Speaker '{speaker}' 95% CI for Mean: [{ci_low:.4f}, {ci_high:.4f}] (mean +/- {ci_range_half:.4f})\n")
                        # Store for speaker stats CSV
                        speaker_scores_for_stats[model_name][speaker] = f"{mean_cos_sim:.4f} +/- {ci_range_half:.4f}"
                    else:
                        log_outfi.write(f"    Speaker '{speaker}': No valid similarity scores to calculate statistics.\n")
                        speaker_scores_for_stats[model_name][speaker] = "N/A" # Mark as N/A for stats CSV
                    log_outfi.write("-" * 30 + "\n") # Separator in log

                # Calculate and log statistics for the current model
                current_model_scores = model_scores_for_stats[model_name]
                if current_model_scores:
                    mean_cos_sim_model = np.mean(current_model_scores)
                    std_cos_sim_model = np.std(current_model_scores)
                    confidence_interval_model = bootstrap((current_model_scores,), np.mean, confidence_level=0.95, random_state=np.random.default_rng())
                    ci_low_model = confidence_interval_model.confidence_interval.low
                    ci_high_model = confidence_interval_model.confidence_interval.high
                    ci_range_half_model = (ci_high_model - ci_low_model) / 2

                    log_outfi.write(f"  Model '{model_name}' Overall Mean Cosine Similarity: {mean_cos_sim_model:.4f}\n")
                    log_outfi.write(f"  Model '{model_name}' Overall Std Dev Cosine Similarity: {std_cos_sim_model:.4f}\n")
                    log_outfi.write(f"  Model '{model_name}' Overall 95% CI for Mean: [{ci_low_model:.4f}, {ci_high_model:.4f}] (mean +/- {ci_range_half_model:.4f})\n")
                    # Store for model stats CSV
                    model_scores_for_stats[model_name] = f"{mean_cos_sim_model:.4f} +/- {ci_range_half_model:.4f}"
                else:
                    log_outfi.write(f"  Model '{model_name}': No valid similarity scores to calculate overall statistics.\n")
                    model_scores_for_stats[model_name] = "N/A"
                log_outfi.write("=" * (len(model_name) + 20) + "\n\n") # End of model section in log

            # Write aggregated speaker statistics to CSV
            speaker_stats_fieldnames = ['Model', 'Speaker', 'Mean Cosine Similarity (+/- 95% CI range/2)']
            speaker_stats_writer = csv.DictWriter(stats_speaker_outfi, fieldnames=speaker_stats_fieldnames)
            speaker_stats_writer.writeheader()
            for model_name_key, speakers_data in speaker_scores_for_stats.items():
                for speaker_key, stats_string in speakers_data.items():
                    speaker_stats_writer.writerow({
                        'Model': model_name_key, 
                        'Speaker': speaker_key, 
                        'Mean Cosine Similarity (+/- 95% CI range/2)': stats_string
                    })

            # Write aggregated model statistics to CSV
            model_stats_fieldnames = ['Model', 'Mean Cosine Similarity (+/- 95% CI range/2)']
            model_stats_writer = csv.DictWriter(stats_model_outfi, fieldnames=model_stats_fieldnames)
            model_stats_writer.writeheader()
            for model_name_key, stats_string in model_scores_for_stats.items():
                model_stats_writer.writerow({
                    'Model': model_name_key, 
                    'Mean Cosine Similarity (+/- 95% CI range/2)': stats_string
                })
            
            print("Similarity calculation with speaker directory structure complete. Check log and output CSV files.")

    except FileNotFoundError as e:
        # This would typically be for one of the output files if paths are complex,
        # or if data_dir/ref_dir are not found at the very start (though os.listdir would catch this first).
        print(f"Error: A specified file or directory was not found: {e.filename}")
        # Potentially re-raise or handle as per library design
        raise
    except IOError as e: # Catch broader I/O errors (permissions, disk full, etc.)
        print(f"Error during file I/O operation: {e}")
        raise
    except Exception as e: # Catch-all for other unexpected errors
        print(f"An unexpected error occurred during the advanced similarity calculation: {e}")
        raise # Re-raise to make it visible to the caller
        
    # Original script has: print("My job here is done.") - replaced with more informative message.

# Note: The original script's main execution block (if __name__ == "__main__":) with argparse
# is handled by the CLI module now.
        writer.writeheader()

        dict_stats_speaker = {}
        dict_stats_model = {}
        
        for model_name in os.listdir(data_dir):
            if not os.path.isdir(os.path.join(data_dir, model_name)):
                continue
