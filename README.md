# AudioProcessingTools

AudioProcessingTools is a collection of tools to prepare a dataset for machine learning tasks. Each tool is completely standalone and correspond to a specific task commonly used to prepare dataset. As this toolkit is supposed to be as generic as possible, only tasks commonly used in 

## Installation

### System Packages

#### GNU/Linux

You need to install the following packages in order to use the toolkit.

* ffmpeg
* espeak
* festival

On Debian/Ubuntu/Mint, you can do so using the following command:

`sudo apt install ffmpeg festival espeak-ng mbrola`

#### Other OS

This is not tested yet so sorry but you are on your own. That should be possible though as the tools used in the toolkit are fairly standard so far.

### Python packages

Once Python is installed on your system, simply run the following from root of the cloned directory: 

`pip install -r requirements.txt`

## Features

### Utils

This is the directory containing all the standalone tools that can be used straight away. All tools are written in Python.

### Pipelines

This directory contains shell scripts that use the tools in the utils/ directory to perform more advanced processing of a corpus.

## Disclamer 

Generative AI models (such as LLMs) was used as a helper in order in the development process to make these tools.