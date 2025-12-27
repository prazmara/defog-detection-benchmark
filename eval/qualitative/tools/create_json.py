import os, json
import argparse

parser = argparse.ArgumentParser(description="Process result directory and output path.")
parser.add_argument(
    "--input_dir",
    type=str,
    nargs='+',  # or use '*' if you want it to be optional
    required=True,
    help="List of paths to result_txt_dir"
)
parser.add_argument("--output_json", 
                    type=str, 
                    required=True, 
                    help="Path to save the output JSON")
parser.add_argument("--test", 
                    action="store_true", 
                    default=False, 
                    help="Test mode to skip certain cities")
parser.add_argument("--train", 
                    action="store_true", 
                    default=False, 
                    help="TRAIN MODE to skip certain cities")
parser.add_argument("--val",
                    action="store_true", 
                    default=False, 
                    help="VAL MODE to skip certain cities")

args = parser.parse_args()
output_json_path = args.output_json
results = []

if args.test:
    cities = [name for name in os.listdir("images/test/")]
if args.train:
    cities = [name for name in os.listdir("images/train/train")]
if args.val: 
    cities = [name for name in os.listdir("images/val/")]


if cities is None:
    print("No cities found in the specified directory.")
    exit(1)

for result_txt_dir in args.input_dir:

    for filename in os.listdir(result_txt_dir):
        full = os.path.join(result_txt_dir, filename)

        # Determine “base” (strip prompt‑extension) and source filename
        if filename.endswith(".png.json"):
            base = filename[:-9]                  # remove “.png.json”
            source_filename = base + ".png"
        elif filename.endswith(".txt"):
            base = filename[:-4]                  # remove “.txt”
            source_filename = base + ".png"
        elif filename.endswith(".json"):
            base = filename[:-5]                  # remove “.json”
            source_filename = base + ".png"
        else:
            continue

        parts = base.split("_")
        city = parts[0]
        if city not in cities:
            print(f"Skipping {filename} as it is not in the specified cities.")
            continue
        # Extract beta if present
        beta = None
        if "_beta_" in base:
            try:
                beta = float(base.split("_beta_")[1])
            except ValueError:
                pass

        # Build target filename (strip foggy suffix if present)
        if "_foggy_" in base:
            target_base = base.split("_foggy_")[0]
        else:
            target_base = base
        target_filename = target_base + ".png"

        source_path = os.path.join("images", "source", city, source_filename)
        target_path = os.path.join("images", "target", city, target_filename)

        if any(d.get("source") == source_path for d in results) and any(d.get("target") == target_path for d in results) :
            #print(f"Skipping duplicate entry for {source_path}")
            continue
        # Load prompt JSON
        with open(full, 'r') as f:
            try:
                prompt = json.load(f)
            except json.JSONDecodeError as e:
                try:
                    #read text file
                    with open(full, 'r', encoding='utf-8') as f:
                        prompt = f.read()
                except Exception as e:
                    print(f"Error reading {filename}: {e}")
                    print(f"Skipping {filename}: {e}")
                    continue
                
                
        
        results.append({
            "source": source_path,
            "target": target_path,
            "prompt": str(prompt),
            "beta": beta
        })


with open(output_json_path, 'w') as out:
    json.dump(results, out, indent=4)

print(f"Saved {len(results)} entries to {output_json_path}")
