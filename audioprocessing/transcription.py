import pandas as pd
"""Text phonetization utilities using the phonemizer library.

This module provides tools for converting text into its phonemic representation
(IPA - International Phonetic Alphabet) using the `phonemizer` library, which
interfaces with backend engines like espeak. It supports multiple languages
and is primarily designed for processing text data from CSV files, often used
in preparing datasets for speech synthesis or other speech-related machine
learning tasks.
"""
from phonemizer import phonemize # Although EspeakBackend is used directly, phonemize itself is a high-level option.
from phonemizer.backend import EspeakBackend
from phonemizer.backend.espeak.exceptions import EspeakError # For specific phonemizer exceptions
import pandas as pd # Explicitly import pandas for type hinting and clarity
from tqdm import tqdm
import os
import csv # For quoting in pandas output

# Defines a mapping from common language names to language codes recognized by phonemizer (espeak).
# This map is used by `get_language_code` to resolve language inputs.
LANGUAGE_CODE_MAP = {
    "english": "en-us",
    "french": "fr-fr",
    "german": "de",
    "portuguese": "pt",
    "polish": "pl",
    "dutch": "nl",
    "spanish": "es",
    "italian": "it"
}

# Store initialized backends to avoid re-initialization for each text
_initialized_backends = {} # type: dict[str, EspeakBackend | None]

def get_language_code(language_name_or_code: str) -> str | None:
    """Validates and returns a language code suitable for phonemizer.

    This function attempts to map common language names (e.g., "english") or
    provided language codes (e.g., "en-us") to a canonical language code
    defined in `LANGUAGE_CODE_MAP`. If a direct match or a value match is found,
    the corresponding code is returned.

    If the input `language_name_or_code` is not found in the map, a warning
    is printed, and the original input string is returned. This allows for
    the possibility that the user has provided a valid espeak language code
    that is not explicitly listed in `LANGUAGE_CODE_MAP` but might still be
    supported by the phonemizer backend.

    Args:
        language_name_or_code (str): The language name (e.g., "french") or
            language code (e.g., "fr-fr") to validate. Comparison is
            case-insensitive.

    Returns:
        str | None: A validated language code string if found or the original
            input string if not found in the map (after printing a warning).
            Returns the lowercased version of the code if it's a direct value match.
            Technically, this could return None if the input is None, but type
            hint is `str` assuming valid string input.
    """
    if language_name_or_code.lower() in LANGUAGE_CODE_MAP:
        return LANGUAGE_CODE_MAP[language_name_or_code.lower()]
    elif language_name_or_code.lower() in LANGUAGE_CODE_MAP.values():
        return language_name_or_code.lower()
    print(f"Warning: Language '{language_name_or_code}' not directly supported or mapped. Trying to use as is.")
    # Attempt to use the code directly if it's not in the map (e.g. 'en' if phonemizer supports it)
    return language_name_or_code 


def initialize_phonemizer_backend(language_code: str) -> EspeakBackend | None:
    """Initializes and returns an EspeakBackend for the given language code.

    This function manages a cache (`_initialized_backends`) of EspeakBackend
    instances. If a backend for the specified `language_code` has already been
    initialized, it's returned from the cache. Otherwise, a new backend is
    initialized, stored in the cache, and then returned.

    The backend is configured to preserve punctuation and include stress markers,
    which are common requirements for TTS applications.

    Args:
        language_code (str): The language code for which to initialize the backend
                             (e.g., "en-us", "fr-fr").

    Returns:
        EspeakBackend | None: An instance of `phonemizer.backend.EspeakBackend`
            if initialization is successful. Returns `None` if an error occurs
            during backend initialization (e.g., language not supported by espeak,
            espeak not installed correctly).

    Raises:
        This function catches exceptions during `EspeakBackend` initialization
        (like `EspeakError` or other `phonemizer` specific errors if the language
        is unsupported or espeak is not found/configured correctly) and prints
        an error message, returning `None` instead of re-raising directly.
        However, underlying issues like a missing espeak installation might
        still lead to program termination depending on `phonemizer`'s behavior.
    """
    if language_code not in _initialized_backends:
        try:
            print(f"Initializing EspeakBackend for language: {language_code}")
            # preserve_punctuation and with_stress were not in original but are common
            _initialized_backends[language_code] = EspeakBackend(
                language_code, 
                preserve_punctuation=True,
                with_stress=True # Common for TTS, can be made optional
            )
        except Exception as e:
            print(f"Error initializing EspeakBackend for {language_code}: {e}")
            _initialized_backends[language_code] = None # Mark as failed
    return _initialized_backends[language_code]

def phonetize_text(text: str, language_code: str) -> str | None:
    """Phonetizes a single text string using a pre-initialized EspeakBackend.

    This function retrieves an initialized `EspeakBackend` for the given
    `language_code` (initializing it if not already cached) and then uses it
    to phonetize the input `text`.

    Args:
        text (str): The text string to be phonetized. The string is stripped
                    of leading/trailing whitespace before phonetization.
        language_code (str): The language code for phonetization (e.g., "en-us").

    Returns:
        str | None: The phonetized string (IPA representation) if successful.
            Returns `None` if the backend for the language is not available or
            if an error occurs during phonetization.

    Raises:
        This function catches exceptions from `backend.phonemize` (e.g.,
        `EspeakError` if espeak fails for a specific utterance) and prints
        an error message, returning `None`.
    """
    backend = initialize_phonemizer_backend(language_code)
    if backend:
        try:
            phonetized = backend.phonemize([text.strip()]) # Expects a list
            return phonetized[0] if phonetized else None
        except Exception as e:
            print(f"Error during phonemization for text '{text[:50]}...' with lang '{language_code}': {e}")
            return None
    return None

def process_csv_for_phonetization(
    input_csv_path: str, 
    output_csv_path: str, 
    transcript_column: str, 
    basename_column: str, # This is often a file path or ID
    language_column: str = None, # Column name in CSV for language
    forced_language_code: str = None, # If all texts are of one language
    input_separator: str = '\t', 
    input_quotechar: str = '|', # Original used quotechar, pandas uses quotechar
    output_separator: str = '\t',
    output_quotechar: str = '|'
    ) -> None:
    """Processes a CSV file to phonetize text in a specified column.

    This function reads an input CSV file using pandas, identifies text data
    from a `transcript_column`, determines the language for phonetization
    (either from a `language_column` or a `forced_language_code`), phonetizes
    the text using the appropriate Espeak backend via `phonetize_text`, and
    writes the results (including a basename/ID, original transcript,
    phonetized string, and language code used) to a new output CSV file.

    It handles pre-initialization of phonemizer backends for efficiency if
    multiple languages are present in the CSV or if a single language is forced.

    Args:
        input_csv_path (str): Path to the input CSV file.
        output_csv_path (str): Path to save the output CSV file containing
                               phonetizations.
        transcript_column (str): The name of the column in the input CSV that
                                 contains the text transcripts to be phonetized.
        basename_column (str): The name of the column in the input CSV that
                               serves as an identifier for each row (e.g., a
                               file path or unique ID). This is carried over to
                               the output CSV.
        language_column (str, optional): The name of the column in the input
                                         CSV that specifies the language for each
                                         row. If `None`, `forced_language_code`
                                         must be provided. Defaults to `None`.
        forced_language_code (str, optional): If provided, this language code
                                              is used for all rows, overriding
                                              any `language_column`. Defaults to `None`.
        input_separator (str): The delimiter used in the input CSV file.
                               Defaults to tab (`'\\t'`).
        input_quotechar (str): The quote character used in the input CSV file.
                               Defaults to pipe (`'|'`).
        output_separator (str): The delimiter to use for the output CSV file.
                                Defaults to tab (`'\\t'`).
        output_quotechar (str): The quote character to use for the output CSV file.
                                Defaults to pipe (`'|'`).

    Returns:
        None. Results are written to `output_csv_path`. Console messages
        indicate progress and any errors.

    Raises:
        pd.errors.EmptyDataError: If the input CSV file is empty.
        FileNotFoundError: If the `input_csv_path` does not exist.
        ValueError: If essential columns (`transcript_column`, `basename_column`,
                    or `language_column` if used and not `forced_language_code`)
                    are not found in the CSV header. Also, if `forced_language_code`
                    is invalid and cannot be used to initialize a backend.
        Exception: Catches and prints other general exceptions from pandas CSV
                   reading or writing, or unexpected issues during processing.
                   The function will typically print an error and return without
                   completing if such issues occur early (e.g., reading CSV).
                   Individual row processing errors (like phonetization failure for
                   a specific text) are logged, and an error marker is placed in
                   the output for that row, allowing the batch job to continue.
    """
    # Validate input_csv_path before attempting to read
    if not os.path.exists(input_csv_path):
        raise FileNotFoundError(f"Error: Input CSV file not found at '{input_csv_path}'")
    if os.path.getsize(input_csv_path) == 0:
        # pandas.errors.EmptyDataError is more specific if pandas is definitely used.
        # Raising a general ValueError if it's empty before pandas parsing.
        raise ValueError(f"Error: Input CSV file '{input_csv_path}' is empty.")

    try:
        # Use keep_default_na=False and na_filter=False to prevent pandas from interpreting "NA" or empty strings as NaN
        # This ensures that all original text values are preserved unless explicitly empty.
        df = pd.read_csv(
            input_csv_path, 
            sep=input_separator, 
            quotechar=input_quotechar, 
            keep_default_na=False, # Important for not converting 'NA' or other strings to NaN
            na_filter=False,       # Important for not converting empty strings to NaN
            dtype=str              # Read all columns as strings initially to preserve data
        )
    except pd.errors.EmptyDataError: # Raised if file is empty or only has headers with no data
        print(f"Error: Input CSV file '{input_csv_path}' is empty or contains no data rows.")
        raise # Re-raise for CLI to handle or for library user
    except Exception as e: # Catch other pandas parsing errors or general file errors
        print(f"Error reading input CSV '{input_csv_path}': {e}")
        raise ValueError(f"Pandas parsing error for {input_csv_path}: {e}")


    # Validate required column names
    if transcript_column not in df.columns:
        raise ValueError(f"Error: Transcript column '{transcript_column}' not found in CSV. Available columns: {df.columns.tolist()}")
    if basename_column not in df.columns:
        raise ValueError(f"Error: Basename column '{basename_column}' not found in CSV. Available columns: {df.columns.tolist()}")
    
    # Validate language column if it's going to be used
    final_forced_lang_code = None # Stores the validated forced language code
    if forced_language_code:
        final_forced_lang_code = get_language_code(forced_language_code)
        if not final_forced_lang_code or not initialize_phonemizer_backend(final_forced_lang_code):
            # Error already printed by initialize_phonemizer_backend if it fails
            raise ValueError(f"Error: Could not initialize phonemizer for forced language: '{forced_language_code}'. Cannot proceed.")
    elif language_column: # Only check language_column if no valid forced_language_code is set
        if language_column not in df.columns:
            raise ValueError(f"Error: Language column '{language_column}' not found and no valid forced language provided. Available columns: {df.columns.tolist()}")
        # Pre-initialize backends for all unique languages found in the specified language column
        unique_langs_in_csv = df[language_column].astype(str).unique() # Ensure they are strings
        for lang_name_or_code_csv in unique_langs_in_csv:
            lang_code_to_init = get_language_code(lang_name_or_code_csv)
            if lang_code_to_init:
                 if not initialize_phonemizer_backend(lang_code_to_init):
                    print(f"Warning: Failed to initialize backend for language '{lang_name_or_code_csv}' (resolved to '{lang_code_to_init}') from CSV. Phonetization for this language may fail.")
            else: # get_language_code already prints a warning if it can't resolve
                print(f"Warning: Could not determine a valid language code for '{lang_name_or_code_csv}' found in language column '{language_column}'.")
    else: # Neither forced_language_code nor language_column provided
        raise ValueError("Error: Must provide either a `forced_language_code` or a valid `language_column` name.")


    phonetized_results = [] # Store results as lists for DataFrame creation
    
    print(f"Processing {len(df)} rows from '{input_csv_path}'...")
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Phonetizing rows"):
        basename = str(row[basename_column]) # Ensure string type
        transcript = str(row[transcript_column]).strip() # Ensure string and strip whitespace
        
        actual_lang_code_used = "N/A" # Default if language cannot be determined
        
        if final_forced_lang_code: # A valid forced language code is available
            actual_lang_code_used = final_forced_lang_code
        elif language_column and language_column in df.columns: # language_column must exist due to checks above
            lang_from_csv = str(row[language_column]).strip()
            resolved_lang_code = get_language_code(lang_from_csv)
            if resolved_lang_code:
                actual_lang_code_used = resolved_lang_code
            else: # Should not happen if pre-initialization logic is sound, but as a fallback
                print(f"Warning: Unresolved language '{lang_from_csv}' for basename '{basename}'. Skipping phonetization for this row.")
                phonetized_results.append([basename, transcript, "Error: Unresolved language", lang_from_csv])
                continue
        else: # Should be caught by initial checks, but as a safeguard
            print(f"Critical Error: No language determination strategy for basename '{basename}'. This should not happen.")
            phonetized_results.append([basename, transcript, "Error: No language specified", "N/A"])
            continue

        ipa_string = None
        if actual_lang_code_used != "N/A":
            # Attempt phonetization only if a valid language code was determined
            ipa_string = phonetize_text(transcript, actual_lang_code_used)
        
        if not ipa_string: # Handles None or empty string from phonetize_text
            # Warning already printed by phonetize_text if it failed
            ipa_string = "Error: Phonetization failed" 
        
        phonetized_results.append([basename, transcript, ipa_string, actual_lang_code_used])

    # Create a DataFrame from the results
    output_df = pd.DataFrame(phonetized_results, columns=['Basename', 'Transcript', 'Phonetization', 'Language'])
    
    # Write the DataFrame to the output CSV
    try:
        # Use csv.QUOTE_NONNUMERIC if a quote character is specified, otherwise QUOTE_MINIMAL.
        # This matches the original script's apparent intent with pandas.
        quoting_style = csv.QUOTE_MINIMAL
        if output_quotechar: # Only use QUOTE_NONNUMERIC if a quotechar is actually defined.
            quoting_style = csv.QUOTE_NONNUMERIC

        output_df.to_csv(
            output_csv_path, 
            sep=output_separator, 
            quotechar=output_quotechar if output_quotechar else '"', # pandas to_csv needs a quotechar if quoting is not QUOTE_NONE
            index=False, 
            quoting=quoting_style
        )
        print(f"\nPhonetization complete. Output saved to '{output_csv_path}'. Processed {len(df)} rows.")
    except Exception as e:
        print(f"Error writing output CSV to '{output_csv_path}': {e}")
        # Consider re-raising for CLI to catch or for library user to handle
        raise

if __name__ == '__main__':
    # Example usage (for testing purposes, not part of the library structure)
    # Create a dummy CSV for testing
    data = {
        'AudioFile': ['file1.wav', 'file2.wav', 'file3.wav', 'file4.wav'],
        'Text': ['Hello world', 'Bonjour le monde', 'Hallo Welt', 'Invalid Language Test'],
        'LanguageID': ['english', 'french', 'german', 'gibrish']
    }
    dummy_df = pd.DataFrame(data)
    dummy_input_csv = 'dummy_input_phon.csv'
    dummy_output_csv = 'dummy_output_phon.csv'
    dummy_df.to_csv(dummy_input_csv, sep='\t', quotechar='|', index=False)

    print(f"Running test with forced language 'english'...")
    process_csv_for_phonetization(
        input_csv_path=dummy_input_csv,
        output_csv_path=dummy_output_csv.replace(".csv", "_en.csv"),
        transcript_column='Text',
        basename_column='AudioFile',
        forced_language_code='en-us'
    )
    
    print(f"\nRunning test with language column 'LanguageID'...")
    process_csv_for_phonetization(
        input_csv_path=dummy_input_csv,
        output_csv_path=dummy_output_csv,
        transcript_column='Text',
        basename_column='AudioFile',
        language_column='LanguageID'
    )
    
    # Clean up dummy files
    # os.remove(dummy_input_csv)
    # os.remove(dummy_output_csv.replace(".csv", "_en.csv"))
    # os.remove(dummy_output_csv)
    print("\nNote: Dummy files created. For CLI testing, use `audiotools phonetize-csv ...`")
