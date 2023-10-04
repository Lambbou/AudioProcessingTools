#!/bin/bash

# Check if an argument is provided
if [ $# -ne 2 ]; then
  echo "Usage: $0 input_directory_path output_directory_path"
  exit 1
fi

# Get the directory path from the command line argument
in_dir_path="$1"
out_dir_path="$1"

# Check if the provided path is a valid directory
if [ ! -d "$in_dir_path" ]; then
  echo "Error: '$in_dir_path' is not a valid directory."
  exit 1
fi

# Creating the output directory in case it wasn't there already
mkdir -p "$out_dir_path"

# Check if the provided path is a valid directory and was created successfully
if [ $? -eq 0 ]; then
  echo "Error: '$out_dir_path' is not a valid directory."
  exit 1
fi

# Get the directory name from the provided path
dir_name=$(basename "$in_dir_path")

# Step 1
# Resample the audio
mkdir -p "$out_dir_path"/resample
python ../utils/resample_corpus.py "$in_dir_path" "$out_dir_path"/resample

# Step 1 - wrap-up
if [ $? -eq 0 ]; then
    echo "Resampling completed successfully."
    # WARNING! Never do a rm after the first task as you would DELETE THE ORIGINAL DATA
else
    echo "Error while resampling the data."
    exit 1
fi

# Step 2
# Normalize the audio
mkdir -p "$out_dir_path"/resample
python ../utils/normalize_corpus.py "$out_dir_path"/resample "$out_dir_path"/norm

# Step 1 - wrap-up
if [ $? -eq 0 ]; then
    echo "Normalization completed successfully."
    rm -rf "$out_dir_path"/resample
    echo "Data from step 1 (resampling) was deleted."
else
    echo "Error while normalizing the data."
    exit 1
fi

# Trim the audio
mkdir -p "$out_dir_path"/resample
python ../utils/trim_silence.py "$out_dir_path"/norm "$out_dir_path"/trim

# Step 2 - wrap-up
if [ $? -eq 0 ]; then
    echo "Trimming completed successfully."
    rm -rf "$out_dir_path"/norm
    echo "Data from step 2 (normalization) was deleted."
else
    echo "Error while trimming the data."
    exit 1
fi

# Final wrap-up
mv "$out_dir_path"/trim/* "$out_dir_path"/
rmdir "$out_dir_path"/trim/

# Check if the symbolic link was created successfully
if [ $? -eq 0 ]; then
  echo "Data was successfully processed and placed in directory '$out_dir_path'."
else
  echo "Error while deleting the final stage of intermediary data. Please check manually what happened."
  exit 1
fi