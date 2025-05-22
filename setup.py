"""Setup script for the audioprocessing package.

This script is used by setuptools to build and install the audioprocessing package.
It defines package metadata, dependencies, entry points (for command-line tools),
and classifiers.
"""
from setuptools import setup, find_packages

# Read the contents of requirements.txt to populate the install_requires list.
# This ensures that dependencies specified in requirements.txt are automatically
# installed when the package is installed via pip.
with open('requirements.txt') as f:
    required = f.read().splitlines()

setup(
    name='audioprocessing',  # The name of the package as it will be listed on PyPI.
    version='0.1.0',        # The current version of the package. Placeholder, should be updated.
    
    # find_packages() automatically discovers and includes all packages (directories with __init__.py)
    # in the project. The 'include' argument specifies which packages to include explicitly.
    # Here, it includes the main 'audioprocessing' package and any sub-packages.
    packages=find_packages(include=['audioprocessing', 'audioprocessing.*']),
    
    author='David Guennec',
    author_email='not@available.com',
    description='A library for audio processing tasks related to ML dataset preparation.', # Short description.
    
    # long_description is typically read from README.md to provide a detailed description on PyPI.
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown', # Specifies the format of the long description.
    
    url='https_path_to_your_repo', # Placeholder for the project's URL.
    
    # install_requires lists the dependencies that will be installed by pip alongside this package.
    # Populated from requirements.txt.
    install_requires=required,
    
    python_requires='>=3.8', # Specifies the minimum Python version required.
    
    # entry_points are used to create command-line scripts.
    # 'console_scripts' defines scripts that can be run from the terminal.
    # 'audiotools=audioprocessing.cli:main_cli' means:
    #   - Create a script named 'audiotools'.
    #   - When 'audiotools' is run, execute the 'main_cli' function
    #     found in the 'audioprocessing.cli' module.
    entry_points={
        'console_scripts': [
            'audiotools=audioprocessing.cli:main_cli',
        ],
    },
    
    # classifiers provide metadata to PyPI to help users find the package.
    # They should conform to the list of trove classifiers on PyPI.
    # Examples:
    #   - Programming Language: Specifies compatible Python versions.
    #   - License: Indicates the license under which the package is distributed.
    #   - Operating System: States OS compatibility.
    #   - Development Status: Current stage of development (e.g., Alpha, Beta, Production/Stable).
    #   - Intended Audience: Describes the target users.
    #   - Topic: Categorizes the package by its subject area.
    classifiers=[
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'License :: OSI Approved :: MIT License', # Assuming MIT, adjust if different.
        'Operating System :: OS Independent',      # Package is not OS-specific.
        'Development Status :: 3 - Alpha',       # Indicates initial development phase.
        'Intended Audience :: Science/Research',
        'Intended Audience :: Developers',
        'Topic :: Multimedia :: Sound/Audio :: Analysis',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],
)
