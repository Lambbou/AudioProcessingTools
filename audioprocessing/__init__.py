"""Initializes the audioprocessing package.

This package provides a collection of tools for audio processing, 
particularly focused on preparing datasets for machine learning tasks.
It includes modules for normalization, similarity calculation, selection (MOS calculation and filtering),
transcription (phonetization), resampling, silence trimming, and various I/O utilities.

The main entry point for command-line operations is the `audiotools` script,
which is defined in `audioprocessing.cli`.
"""

# This __init__.py file is intentionally kept minimal.
# Specific functionalities should be imported directly from their respective modules, for example:
# from audioprocessing.normalization import process_directory_for_normalization
# from audioprocessing.resampling import resample_corpus_to_output_dir
#
# This approach helps in keeping the package namespace clean and promotes clarity
# on where each function or class originates.
#
# For a list of available command-line tools, run:
# audiotools --help
