import csv

def read_csv_and_add_png(csv_path, column_name="basename"):
    result = set()
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if column_name in row:
                result.add(row[column_name] + ".png")
    return result


