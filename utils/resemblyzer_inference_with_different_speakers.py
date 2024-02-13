import os
from resemblyzer import VoiceEncoder, preprocess_wav
from scipy.spatial.distance import cosine, euclidean
import argparse
import numpy as np
import csv
from scipy.stats import bootstrap

# This script computes Cosinus Similarity for a given set of audio files

# Example of use: 
# python resemblyzer_inference_with_different_speakers.py --data_dir /vrac/dguennec/dev/espnet/egs/ssw12/tts1/synthesis/ --ref_dir /vrac/dguennec/dev/espnet/egs/ssw12/tts1/downloads/test_natural_set/test_natural/ --output_csv_file resemblyzer_foobar.csv --output_log_file resemblyzer_log.log --output_model_stats resemblyzer_models.csv --output_speaker_stats resemblyzer_speaker.csv

def get_embedding(filepath, model):
    ref_wav = preprocess_wav(filepath)
    ref_embed = model.embed_utterance(ref_wav)
    
    return ref_embed


def compute_cosine_similarity(x, y):
    return 1 - cosine(x, y)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', help='Directory of wav files to test (structure : data_dir/model_name/speaker_name/wav_files)')
    parser.add_argument('--ref_dir', help='Directory of reference wav files (structure : ref_dir/speaker_name/wav_files)')
    parser.add_argument('--output_csv_file', help='Where to save the results of the objective evaluation')
    parser.add_argument('--output_log_file', help='Where to save the means/std of the objective evaluation')
    parser.add_argument('--output_model_stats', help='Where to save the means/std of the objective evaluation')
    parser.add_argument('--output_speaker_stats', help='Where to save the means/std of the objective evaluation')

    args = parser.parse_args()
    data_dir = args.data_dir
    ref_dir = args.ref_dir
    output_csv_file = args.output_csv_file
    output_log_file = args.output_log_file
    output_speaker_stats = args.output_speaker_stats
    output_model_stats = args.output_model_stats

    speaker_encoder = VoiceEncoder()
    
    with open(output_csv_file, 'w') as outfi, open(output_log_file, 'w') as log_outfi, open(output_speaker_stats, 'w') as stats_speaker_outfi, open(output_model_stats, 'w') as stats_model_outfi:
        fieldnames = ['model', 'speaker', 'cloned_wav','ref', 'cosine_similarity', 'euclidean_similarity']
        writer = csv.DictWriter(outfi, fieldnames=fieldnames)
        writer.writeheader()

        # Prepare stats
        dict_stats_speaker = {}
        dict_stats_model = {}
        for model_name in os.listdir(data_dir):
            dict_stats_speaker[model_name] = {}

        # Compute speaker similarity
        for model_name in os.listdir(data_dir):
            path_model_name = os.path.join(data_dir, model_name)
            log_outfi.write("Model : " + str(model_name) + "\n")
            mod_cosine_similarity_list, mod_euclidean_distance_list = [], []
            
            for speaker in os.listdir(path_model_name):
                log_outfi.write("Speaker : " + str(speaker) + "\n")
                path_model_speaker = os.path.join(path_model_name, speaker)
                sp_cosine_similarity_list, sp_euclidean_distance_list = [], []

                for cloned_wav in os.listdir(path_model_speaker):
                    if os.path.splitext(cloned_wav)[1]=='.wav':
                        cloned_wav_path = os.path.join(path_model_speaker, cloned_wav)

                        # Parse to find corresponding ref_file
                        speaker_underscore = speaker + "_"
                        sample_raw_name = cloned_wav.split(speaker_underscore)[-1]
                        sample_raw_name = sample_raw_name.split("_synthesis")[0] + ".wav"
                        ref_path = os.path.join(ref_dir, speaker)
                        ref_path = os.path.join(ref_path, sample_raw_name)

                        try:
                            # Compute speaker similarity
                            ref_embed = get_embedding(ref_path, speaker_encoder)
                            cloned_embed = get_embedding(cloned_wav_path, speaker_encoder)
                            cosine_similarity = compute_cosine_similarity(ref_embed, cloned_embed)
                            euclidean_distance = euclidean(ref_embed, cloned_embed)
                        except Exception as e:
                            print(e)
                        
                        # Write results in outfi
                        writer.writerow({'model': str(model_name), 'speaker': str(speaker), 'cloned_wav': str(cloned_wav_path), 'ref': str(ref_path), 'cosine_similarity': str(cosine_similarity), 'euclidean_similarity': str(euclidean_distance)})

                        # Store for mean/std computation
                        sp_cosine_similarity_list.append(cosine_similarity)
                        # sp_euclidean_distance_list.append(euclidean_distance)
                        mod_cosine_similarity_list.append(cosine_similarity)
                        # mod_euclidean_distance_list.append(euclidean_distance)
                
                # Compute the mean and standard deviation or accuracy
                cosine_similarity_mean, cosine_similarity_std = np.mean(sp_cosine_similarity_list), np.std(sp_cosine_similarity_list)
                # euclidean_distance_mean, euclidean_distance_std = np.mean(sp_euclidean_distance_list), np.std(sp_euclidean_distance_list)
                log_outfi.write("Speaker mean cosine similarity = " + str(cosine_similarity_mean) + "\n")
                # log_outfi.write("Speaker std cosine similarity = " + str(cosine_similarity_std))
                # log_outfi.write("Speaker mean euclidean distance = " + str(cosine_similarity_mean))
                # log_outfi.write("Speaker std euclidean distance = " + str(cosine_similarity_std))
                confidence_interval = bootstrap((sp_cosine_similarity_list,), np.mean, confidence_level=0.95,random_state=np.random.default_rng())
                log_outfi.write("Lower confidence interval = " + str(confidence_interval.confidence_interval.low) + "\n")
                log_outfi.write("Higher confidence interval = " + str(confidence_interval.confidence_interval.high) + "\n")
                log_outfi.write("Confidence +- = " + str((confidence_interval.confidence_interval.high - confidence_interval.confidence_interval.low)/2) + "\n")
                result_for_stats = str(cosine_similarity_mean) + " +/- " + str((confidence_interval.confidence_interval.high - confidence_interval.confidence_interval.low)/2)
                dict_stats_speaker[model_name][speaker] = result_for_stats
            
            # Compute the mean and standard deviation or accuracy
            cosine_similarity_mean, cosine_similarity_std = np.mean(mod_cosine_similarity_list), np.std(mod_cosine_similarity_list)
            # euclidean_distance_mean, euclidean_distance_std = np.mean(mod_euclidean_distance_list), np.std(mod_euclidean_distance_list)
            log_outfi.write("Model mean cosine similarity = " + str(cosine_similarity_mean) + "\n")
            # log_outfi.write("Model std cosine similarity = " + str(cosine_similarity_std))
            # log_outfi.write("Model mean euclidean distance = " + str(cosine_similarity_mean))
            # log_outfi.write("Model std euclidean distance = " + str(cosine_similarity_std))
            confidence_interval = bootstrap((mod_cosine_similarity_list,), np.mean, confidence_level=0.95,random_state=np.random.default_rng())
            log_outfi.write("Lower confidence interval = " + str(confidence_interval.confidence_interval.low) + "\n")
            log_outfi.write("Higher confidence interval = " + str(confidence_interval.confidence_interval.high) + "\n")
            log_outfi.write("Confidence +- = " + str((confidence_interval.confidence_interval.high - confidence_interval.confidence_interval.low)/2) + "\n")
            result_for_stats = str(cosine_similarity_mean) + " +/- " + str((confidence_interval.confidence_interval.high - confidence_interval.confidence_interval.low)/2)
            dict_stats_model[model_name] = result_for_stats
        
        fieldnames_speaker = ['Model', 'Speaker', 'Mean Cosine Similarity']
        writer_stats_speaker = csv.DictWriter(stats_speaker_outfi, fieldnames=fieldnames_speaker)
        writer_stats_speaker.writeheader()
        for model_name in dict_stats_speaker.keys():
            for speaker in dict_stats_speaker[model_name].keys():
                writer_stats_speaker.writerow({'Model': str(model_name), 'Speaker': str(speaker), 'Mean Cosine Similarity': dict_stats_speaker[model_name][speaker]})

        fieldnames_model = ['Model', 'Mean Cosine Similarity']
        writer_stats_model = csv.DictWriter(stats_model_outfi, fieldnames=fieldnames_model)
        writer_stats_model.writeheader()
        for model_name in dict_stats_model.keys():
            writer_stats_model.writerow({'Model': str(model_name), 'Mean Cosine Similarity': dict_stats_model[model_name]})


    print("My job here is done.")
