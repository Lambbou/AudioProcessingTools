import click
import pandas as pd
from phonemizer import phonemize
from phonemizer.backend import EspeakBackend
from tqdm import tqdm

CHOICES = {"english":"en-us", "french":"fr-fr", "german":"de", "portuguese":"pt", "polish":"pl", "dutch":"nl", "spanish":"es", "italian":"it"}

def get_lang_code(text:str) -> str:
    for choice, lg in CHOICES.items():
        if choice in text:
            return lg
    exit(f"ERROR: String {text} must match one of the following choices: {CHOICES}")
    
@click.command()
@click.argument('input_csv_file', type=click.File('r'))
@click.argument('output_csv_file', type=click.File('w'))
@click.option('-l', '--lang', type=click.STRING, default='', help='Use this option to force the language if you don\'t have a language row in your csv.')
@click.option('--basename-row', type=click.STRING, default="Basename", help='The name (in the header) of the row corresponding the the path to the wav file.') # "clips_with_full_path"
@click.option('--transcript-row', type=click.STRING, default="Basename", help='The desired sampling rate.') # ""
@click.option('-s','--separator', type=click.STRING, default="\t", help='The desired sampling rate.') # "|"
@click.option('-q', '--quotechar', type=click.STRING, default='|', help='The desired sampling rate.') # '"'
def phonetize_sentences(input_csv_file, output_csv_file, lang, basename_row, separator, quotechar):
    # Read the CSV file into a DataFrame
    df = pd.read_csv(input_csv_file, sep=separator, quotechar=quotechar)

    # Create an empty list to store the matched lines
    phonetized_lines = []

    # Importing backends for code efficiency
    backend_en = EspeakBackend('en-us', preserve_punctuation=True)
    backend_fr = EspeakBackend('fr-fr', preserve_punctuation=True)
    backend_de = EspeakBackend('de', preserve_punctuation=True)
    backend_pt = EspeakBackend('pt', preserve_punctuation=True)
    backend_pl = EspeakBackend('pl', preserve_punctuation=True)
    backend_nl = EspeakBackend('nl', preserve_punctuation=True)
    backend_es = EspeakBackend('es', preserve_punctuation=True)
    backend_it = EspeakBackend('it', preserve_punctuation=True)

    # Match each line of the CSV with the associated line in the WAV list with tqdm progress bar
    for _, row in tqdm(df.iterrows(), total=len(df)):
        # Language        Basename        Transcript -> header
        # clip_id|text_aligned|clips_with_full_path|text_phonetized
        if lang: 
            language = lang
        else:
            language = get_lang_code(row["Language"])
        basename = row[basename_row"Basename"]
        transcript = row["Transcript"].strip()

        ipa_string = ""
        if language == "en-us":
            ipa_string = backend_en.phonemize([transcript])[0]
        elif language == "fr-fr":
            ipa_string = backend_fr.phonemize([transcript])[0]
        elif language == "de":
            ipa_string = backend_de.phonemize([transcript])[0]
        elif language == "pt":
            ipa_string = backend_pt.phonemize([transcript])[0]
        elif language == "pl":
            ipa_string = backend_pl.phonemize([transcript])[0]
        elif language == "nl":
            ipa_string = backend_nl.phonemize([transcript])[0]
        elif language == "es":
            ipa_string = backend_es.phonemize([transcript])[0]
        elif language == "it":
            ipa_string = backend_it.phonemize([transcript])[0]
        else:
            exit(f"Got an unexpected language code: {language}")
        #print(f"Text phonetized in {row['Language']} : {ipa_string}")

        if len(ipa_string) <= 0:
            print(f"That does not seem normal {transcript=} |||||| {ipa_string=}")

        phonetized_lines.append([basename, transcript, ipa_string, language])

    # Create a DataFrame with the matched lines and export it to a new CSV file
    phonetized_df = pd.DataFrame(phonetized_lines, columns=['Basename', 'Transcript', 'Phonetization', 'Language'])
    phonetized_df.to_csv(output_csv_file, sep='\t', quotechar='|', index=False)

    print("Phonetization completed successfully!")

if __name__ == '__main__':
    phonetize_sentences()