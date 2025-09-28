from datasets import Dataset, DatasetDict, Features, Image, Value
import os

def frank_dataset():


# Paths to your image directories

     cond_image_dir = "/path/to/condition_images"
     image_dir = "/path/to/images"

# Collect file paths

     image_files = sorted([os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith(".png")])
     cond_image_files = sorted([os.path.join(cond_image_dir, f) for f in os.listdir(cond_image_dir) if f.endswith(".png")])


     assert len(image_files) == len(cond_image_files), "Mismatch between image and conditioning image counts"

# Create dataset dict
     data = {
     "image": image_files,
     "conditioning_image": cond_image_files,
     "text": ["Need to remove fog so the objects should be clearly visible."] * len(image_files),
     "name": image_files,
     }

# Define features explicitly
     features = Features({
     "image": Image(),
     "conditioning_image": Image(),
     "text": Value("string"),
     "name": Value("string"),
     })

    # Build dataset
     train_dataset = Dataset.from_dict(data, features=features)

    # Wrap in DatasetDict
     dataset = DatasetDict({
     "train": train_dataset
     })

     return dataset
