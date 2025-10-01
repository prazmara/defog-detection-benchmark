
# VLM JUDGE PIPELINE

### Introduction
**VLM Judge**: An automated evaluation pipeline that uses Vision-Language Models to score defogged images against foggy inputs and ground truth, providing rubric-based, human-like qualitative metrics at scale.


### Enviroment Set up
**SETUP**
enviroment setup:
```
conda env create -n <name> -f requirments.yaml  
```

In this paper we are using azure openai APIs, set up can be found here https://ai.azure.com/doc/?tid=608b3d1d-5200-4549-8210-f8b4312acabf 

Please create a .env (saved locally) with strcutre below
.env :
```
AZURE_API_KEY = <APIKEY>
AZURE_API_VERSION = <API VERSION>
AZURE_API_BASE = <API BASE NAME>
```
## 📂 Default Folder Structure

Please make sure your dataset is organized in the following structure.  
Both **foggy images** and **ground truth images** must follow this convention.


```
├──cityscape
    ├── defogged_models
    │   ├── dehazeformer
    │   │   └── val
    │   │       ├── frankfurt
    │   │       ├── lindau
    │   │       └── munster
    ├── foggy
    │   └── val
    │       ├── frankfurt
    │       ├── lindau
    │       └── munster
    └── ground_truth
        └── val
            ├── frankfurt
            ├── lindau
            └── munster
```


## VLM Judge 
Entry point to VLM judge pipeline is vlm_judge_pipeline

```
python vlm_judge_pipeline.py --model <model_name> 
```

## 🧩 Model Choices

Use the `--model` argument to specify which model’s outputs you want to evaluate.  
Available options are:

- `dehazeformer`  
- `focalnet`  
- `mitdense`  
- `mitnh`  
- `fluxnet`  
- `nanobanana`  
- `b01_dhft`  
- `b01_dhf`  
- `b01`  
- `flux_non_cot`  
- `flux_cot`  
- `flux_split`  
- `flux_split_cot`  
- `flux_split_non_cot`  
- `GT` *(ground truth baseline)*  

## ⚙️ Command-Line Arguments

| Argument                | Default                                | Description                                                                 |
|--------------------------|----------------------------------------|-----------------------------------------------------------------------------|
| `--image-folder`         | *(none)*                               | Path to a folder of candidate images to process (only required if the images are not in the default location).                                      |
| `--num-samples`          | `50`                                   | Number of samples to process                                                |
| `--foggy-image-folder`   | `cityscape/foggy/val`                  | Path to foggy input images                                                  |
| `--gt-image-folder`      | `cityscape/ground_truth/val`           | Path to ground-truth (clear) images                                         |
| `--output-csv`           | `DB/combined_results.csv`              | Output CSV file to store results                                            |
| `--skip-csv`             | *(none)*                               | Path to a CSV file with basenames to skip                                   |
| `--fixed`                | `fixed_basenames/candidates_paths.txt` | Path to a text file with fixed basenames (candidates) to process, used to ensure same images are used for each model           |


## 📑 Output CSV Format

After running the pipeline, results are saved to a CSV file (default: `DB/combined_results.csv`).  
Each row corresponds to one evaluated image triplet (Foggy → Candidate → Ground Truth).

The CSV contains the following columns:

| Column                   | Description                                                                 |
|---------------------------|-----------------------------------------------------------------------------|
| `model`                  | Name of the evaluated model (e.g., `dehazeformer`, `focalnet`, etc.)        |
| `city`                   | City name (e.g., `frankfurt`, `lindau`, `munster`)                         |
| `basename`               | Base filename of the image                                                  |
| `foggy_path`             | Path to the foggy input image                                               |
| `cand_path`              | Path to the candidate (dehazed) image                                       |
| `gt_path`                | Path to the ground-truth (clear) image                                      |
| `visibility_restoration` | Score (0–5) for mid/far/background visibility restoration                   |
| `visual_distortion`      | Score (0–5) for artifacts, color shifts, halos, etc.                        |
| `boundary_clarity`       | Score (0–5) for contour and edge sharpness                                  |
| `scene_consistency`      | Score (0–5) for preservation of global layout and geometry                  |
| `object_consistency`     | Score (0–5) for correct presence/placement of objects (cars, people, signs) |
| `perceived_detectability`| Score (0–5) for how detectable objects are in the candidate                  |
| `relation_consistency`   | Score (0–5) for spatial relations between objects (diagnostic only)         |
| `explanation`            | Short text explanation from the VLM judge                                   |





### Future Extensions
- Support for multiple VLM backends (OpenAI, Google GenAI, Anthropic, etc.)
- Configurable prompt templates and rubrics
- Automatic evaluation across additional datasets