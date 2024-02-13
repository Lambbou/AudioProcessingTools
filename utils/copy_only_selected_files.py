import csv
import os
import click
import shutil

def copy_files_from_tsv(tsv_file, output_directory):
    try:
        with open(tsv_file, 'r', newline='') as tsvfile:
            reader = csv.DictReader(tsvfile, delimiter='\t', quotechar='|')
            for row in reader:
                # Get the path from the "Basename" column
                path = row.get('Basename')
                if path:
                    # Check if the source file exists
                    if os.path.exists(path):
                        # Copy the file to the specified output directory
                        shutil.copy(path, os.path.join(output_directory, os.path.basename(path)))
                        print(f"Copied: {path} to {output_directory}")

    except Exception as e:
        print(f"Error: {str(e)}")

@click.command()
@click.argument('tsv_file', type=click.Path(exists=True))
@click.argument('output_directory', type=click.Path())
def process_tsv(tsv_file, output_directory):
    copy_files_from_tsv(tsv_file, output_directory)
    print("File copy completed.")

if __name__ == '__main__':
    process_tsv()
