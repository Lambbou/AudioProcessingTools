import pandas as pd
from phonemizer import phonemize # This is a high-level API, consider if direct backend use is better
from phonemizer.backend import EspeakBackend
from tqdm import tqdm
import os

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
_initialized_backends = {}

def get_language_code(language_name_or_code: str) -> str | None:
    """
    Validates and returns a supported language code.
    Accepts full names (e.g., 'english') or codes (e.g., 'en-us').
    """
    if language_name_or_code.lower() in LANGUAGE_CODE_MAP:
        return LANGUAGE_CODE_MAP[language_name_or_code.lower()]
    elif language_name_or_code.lower() in LANGUAGE_CODE_MAP.values():
        return language_name_or_code.lower()
    print(f"Warning: Language '{language_name_or_code}' not directly supported or mapped. Trying to use as is.")
    # Attempt to use the code directly if it's not in the map (e.g. 'en' if phonemizer supports it)
    return language_name_or_code 


def initialize_phonemizer_backend(language_code: str) -> EspeakBackend | None:
    """Initializes and returns an EspeakBackend for the given language code."""
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
    """
    Phonetizes a single text string using a pre-initialized EspeakBackend.
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
    ):
    """
    Reads a CSV, phonetizes a specified text column, and writes results to a new CSV.
    """
    try:
        df = pd.read_csv(input_csv_path, sep=input_separator, quotechar=input_quotechar, keep_default_na=False, na_filter=False)
    except Exception as e:
        print(f"Error reading input CSV '{input_csv_path}': {e}")
        return

    if transcript_column not in df.columns:
        print(f"Error: Transcript column '{transcript_column}' not found in CSV. Available columns: {df.columns.tolist()}")
        return
    if basename_column not in df.columns:
        print(f"Error: Basename column '{basename_column}' not found in CSV. Available columns: {df.columns.tolist()}")
        return
    if language_column and language_column not in df.columns and not forced_language_code:
        print(f"Error: Language column '{language_column}' not found and no forced language provided. Available columns: {df.columns.tolist()}")
        return

    phonetized_results = []
    
    # Pre-initialize backends if a language column exists and many languages are expected
    # Or if a single forced_language_code is given
    if forced_language_code:
        valid_forced_lang_code = get_language_code(forced_language_code)
        if not valid_forced_lang_code or not initialize_phonemizer_backend(valid_forced_lang_code):
            print(f"Error: Could not initialize phonemizer for forced language: {forced_language_code}")
            return
    elif language_column and language_column in df.columns:
        unique_langs = df[language_column].unique()
        for lang_name_or_code in unique_langs:
            lang_code = get_language_code(str(lang_name_or_code))
            if lang_code:
                 initialize_phonemizer_backend(lang_code) # Pre-initialize
            else:
                print(f"Warning: Could not determine a valid language code for '{lang_name_or_code}' in language column.")


    for _, row in tqdm(df.iterrows(), total=len(df), desc="Phonetizing rows"):
        basename = row[basename_column]
        transcript = str(row[transcript_column]).strip() # Ensure transcript is string
        
        current_lang_code = None
        if forced_language_code:
            current_lang_code = valid_forced_lang_code
        elif language_column and language_column in df.columns:
            lang_from_csv = str(row[language_column]).strip()
            current_lang_code = get_language_code(lang_from_csv)
            if not current_lang_code:
                print(f"Skipping row due to unresolvable language: {lang_from_csv} for basename {basename}")
                phonetized_results.append([basename, transcript, "Error: Unresolved language", lang_from_csv])
                continue
        else: # Should not happen if checks above are correct
            print("Error: No language information found for phonetization.")
            phonetized_results.append([basename, transcript, "Error: No language specified", "N/A"])
            continue

        ipa_string = None
        if current_lang_code:
            ipa_string = phonetize_text(transcript, current_lang_code)
        
        if not ipa_string:
            print(f"Warning: Phonetization failed for: '{transcript[:50]}...' (lang: {current_lang_code}). IPA will be empty or error marker.")
            ipa_string = ipa_string or "Error: Phonetization failed" # Keep previous error or set new one

        phonetized_results.append([basename, transcript, ipa_string, current_lang_code or "N/A"])

    phonetized_df = pd.DataFrame(phonetized_results, columns=['Basename', 'Transcript', 'Phonetization', 'Language'])
    try:
        phonetized_df.to_csv(output_csv_path, sep=output_separator, quotechar=output_quotechar, index=False, quoting=csv.QUOTE_NONNUMERIC if output_quotechar else csv.QUOTE_MINIMAL)
        print(f"Phonetization complete. Output saved to '{output_csv_path}'")
    except Exception as e:
        print(f"Error writing output CSV '{output_csv_path}': {e}")

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
