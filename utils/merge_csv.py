import csv
import click

def resolve_header(colonne_cle:str, header1:str, header2:str) -> tuple[int, int]:
    h1 = h2 = -1
    for i, h in enumerate(header1):
        if h == colonne_cle:
            h1 = i
    if h1 == -1:
        exit(f"Error: couldn't find column \"{colonne_cle}\" in the first CSV. Please check.")
    for i, h in enumerate(header2):
        if h == colonne_cle:
            h2 = i
    if h1 == -1:
        exit(f"Error: couldn't find column \"{colonne_cle}\" in the second CSV. Please check.")
        
    return h1,h2

@click.command()
@click.argument("fichier_csv1", type=click.Path(exists=True))
@click.argument("fichier_csv2", type=click.Path(exists=True))
@click.argument("colonne_cle", type=str)
@click.option("-o", "--output", type=str, default="merged_data.csv", help="The name of the merged file if you want to give it a specific name.")
def main(fichier_csv1, fichier_csv2, colonne_cle, output):
    """
    Fonction qui fusionne deux CSV en se basant sur une colonne commune.

    Args:
        fichier_csv1: Le chemin du premier fichier CSV en entrée.
        fichier_csv2: Le chemin du deuxième fichier CSV en entrée.
        colonne_cle: Le nom de la colonne commune entre les deux fichiers.
    """

    with open(fichier_csv1, "r") as f1:
        reader1 = csv.reader(f1, delimiter="\t", quotechar='|')
        lignes1 = list(reader1)

    with open(fichier_csv2, "r") as f2:
        reader2 = csv.reader(f2, delimiter="\t", quotechar='|')
        lignes2 = list(reader2)
    
    key1, key2 = resolve_header(colonne_cle, lignes1[0], lignes2[0])

    # Crée un dictionnaire qui mappe les lignes du premier fichier CSV sur les lignes du deuxième fichier CSV
    # en se basant sur la colonne clé
    resulting_data = []
    match = 0
    for ligne1 in lignes1:
        for ligne2 in lignes2:            
            if ligne1[key1] == ligne2[key2]:
                print(f"Found a match: {ligne1} --- \t --- {ligne2}")
                match+=1
                tmp = ligne2
                del tmp[key2]
                resulting_data.append(ligne1 + tmp)
                break
            
    if match != len(lignes2):
        print(f"Found {match} matching lines but {fichier_csv2} is {len(lignes2)} lines long.")

    # Écrit les lignes fusionnées dans un nouveau fichier CSV
    with open(output, "w") as f:
        writer = csv.writer(f, delimiter="\t", quotechar='|')
        writer.writerows(resulting_data)


if __name__ == "__main__":
    main()
