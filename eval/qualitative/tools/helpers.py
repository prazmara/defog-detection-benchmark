import csv
import pandas as pd
import os

def read_csv_and_add_png(csv_path, column_name="basename"):
    result = set()
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if column_name in row:
                result.add(row[column_name] + ".png")
    return result

def read_missing_csv_and_find_file_name(csv_path, image_folder_path, model):
    #loop through image folder and find paths that match the 
    df = pd.read_csv(csv_path)
    files = os.listdir(image_folder_path)
    print(f"Total files in folder: {len(files)}")
    file_dict = {}
    for filename in files:
        base, ext = os.path.splitext(filename)
        parts = base.split("_")
        if len(parts) < 3:
            raise ValueError(f"Unexpected cand_name format: {filename}")

        city, seq, frame = parts[0], parts[1], parts[2]
        print(f"Mapping key: {(city, seq, frame,model)} to file: {filename}")
        file_dict[(str(city), int(seq), int(frame), model)] = filename
    print(f"Total files mapped by key: {len(file_dict)}")
    
    basenames = set()
    for _, row in df.iterrows():
        city = str(row["city"])
        seq  = 000000 + int(row["seq_from_basename"])
        frame = 000000 + int(row["frame_from_basename"])
        #print(f"Looking for key: {(city, seq, frame,row["model"])}")

        key = (city, int(seq), int(frame),row["model"])
        if key in file_dict:
            basenames.add(file_dict[key])
    return basenames
