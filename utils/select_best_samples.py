import csv
import click

def filter_and_save_csv(input_csv, output_csv, size_threshold=600000):
    # Initialize lists to store rows that meet the size threshold
    selected_rows = []
    total_size = 0

    with open(input_csv, 'r') as csvfile:
        reader = csv.DictReader(csvfile, delimiter="\t", quotechar='|')
        fieldnames = reader.fieldnames

        # Sort rows by score in descending order
        sorted_rows = sorted(reader, key=lambda row: float(row['MOS']), reverse=True)

        for row in sorted_rows:
            # Check if adding the row would exceed the size threshold
            if total_size + float(row['Duration']) <= size_threshold:
                selected_rows.append(row)
                total_size += float(row['Duration'])

    with open(output_csv, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames, delimiter="\t", quotechar='|')
        writer.writeheader()
        writer.writerows(selected_rows)

@click.command()
@click.argument('input_csv', type=click.Path(exists=True))
@click.argument('output_csv', type=click.Path())
@click.option('--size-threshold', default=600000, help='Size threshold for retaining rows')
def process_csv(input_csv, output_csv, size_threshold):
    filter_and_save_csv(input_csv, output_csv, size_threshold)
    print(f'Selected rows saved to {output_csv}')

if __name__ == '__main__':
    process_csv()
