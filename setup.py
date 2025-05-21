from setuptools import setup, find_packages

# Read requirements from requirements.txt
with open('requirements.txt') as f:
    required = f.read().splitlines()

setup(
    name='audioprocessing',
    version='0.1.0', # YOUR_VERSION
    packages=find_packages(include=['audioprocessing', 'audioprocessing.*']),
    author='AudioProcessingTools Contributor', # YOUR_NAME
    author_email='user@example.com', # YOUR_EMAIL
    description='A library for audio processing tasks related to ML dataset preparation.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https_path_to_your_repo', # Replace with your repo URL if available
    install_requires=required,
    python_requires='>=3.8', # Based on README, Python 3.10 tested, 3.6+ should work. Let's go with 3.8+
    entry_points={
        'console_scripts': [
            'audiotools=audioprocessing.cli:main_cli',
        ],
    },
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License', # Assuming MIT based on typical open source projects, adjust if different
        'Operating System :: OS Independent',
        'Development Status :: 3 - Alpha', # Initial development
        'Intended Audience :: Science/Research',
        'Topic :: Multimedia :: Sound/Audio :: Analysis',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
)
