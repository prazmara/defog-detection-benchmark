# VLM Judge Pipeline

A modular pipeline for evaluating defogged images using Vision Language Models.

## Setup 

## Installation

Ensure you have the required dependencies:

```bash

conda env create -f requirments.yaml -n <name>
```

## Pipeline Structure

```
pipeline/
├── __init__.py          # Package initialization
├── config.py            # Configuration constants
├── cli.py               # Command-line interface
├── azure_setup.py       # Azure OpenAI client setup
├── data_loader.py       # Data loading utilities
└── image_processor.py   # Image processing and evaluation
```

## Usage

### Basic Command

```bash
python vlm_judge_pipeline.py --model <model_name> --output-csv <output_file>
```

### Example Commands

1. **Process a specific model with default Cityscape dataset:**
```bash
python vlm_judge_pipeline.py \
    --model dehazeformer \
    --output-csv results/dehazeformer_results.csv
```

2. **Process images from a specific folder:**
```bash
python vlm_judge_pipeline.py \
    --model focalnet \
    --image-folder cityscape/defogged_models/focalnet/val/frankfurt \
    --output-csv results/focalnet_results.csv
```

3. **Process with a skip list (to avoid reprocessing):**
```bash
python vlm_judge_pipeline.py \
    --model mitdense \
    --skip-csv results/already_processed.csv \
    --output-csv results/mitdense_results.csv
```

4. **Process missing samples from a CSV:**
```bash
python vlm_judge_pipeline.py \
    --model fluxnet \
    --missing-csv results/missing_samples.csv \
    --image-folder cityscape/defogged_models/fluxnet/val \
    --output-csv results/fluxnet_results.csv
```

5. **Reuse existing sample file:**
```bash
python vlm_judge_pipeline.py \
    --model nanobanana \
    --reuse-sample \
    --sample-file samples/selected_images.csv \
    --output-csv results/nanobanana_results.csv
```

6. **Process ACDC dataset:**
```bash
python vlm_judge_pipeline.py \
    --model GT \
    --dataset acdc \
    --image-folder acdc/acdc-fog-val \
    --output-csv results/acdc_gt_results.csv
```

### Command-Line Arguments

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `--model` | Yes | - | Model name (dehazeformer, focalnet, mitdense, etc.) |
| `--output-csv` | Yes | - | Output CSV file path |
| `--dataset` | No | cityscape | Dataset type: `cityscape` or `acdc` |
| `--image-folder` | No | - | Path to folder of images to process |
| `--reuse-sample` | No | False | Reuse existing sample file |
| `--sample-file` | No | - | Path to sample file (used with `--reuse-sample`) |
| `--skip-csv` | No | - | CSV file with basenames to skip |
| `--missing-csv` | No | - | CSV with missing samples to process |

### Supported Models

- dehazeformer
- focalnet
- mitdense
- mitnh
- fluxnet
- nanobanana
- b01_dhft
- flux_non_cot
- flux_cot
- b01_dhf
- flux_split
- flux_split_cot
- flux_split_non_cot
- b01
- GT

## Environment Setup

Ensure you have the following environment variables set in your `.env` file:

```bash
GPT5_CHAT=your_azure_api_key
```

## Output

The pipeline generates:
- CSV file with evaluation results at the specified `--output-csv` path
- JSONL file at `outputs/combined_results.jsonl` for backup

## Module Descriptions

### config.py
Contains all configuration constants including supported models, Azure settings, and default parameters.

### cli.py
Handles command-line argument parsing and validation.

### azure_setup.py
Manages Azure OpenAI client initialization and output directory setup.

### data_loader.py
Provides functions for loading image datasets from various sources (folders, CSV files, default locations).

### image_processor.py
Handles image path resolution, VLM scoring, and result persistence.


# Image Defogging Pipeline

A CLI tool for defogging images using Google Gemini AI.

## Features

- Process images from any input directory
- Configurable output directory
- Support for API key via CLI argument or environment variable
- Customizable model, prompt, and image dimensions
- Filter images by filename substring
- Skip already processed images
- Comprehensive progress tracking and error reporting


## Usage

### Basic Usage

```bash
python nano_bananna.py \
  --input-dir /path/to/foggy/images \
  --output-dir /path/to/output \
  --api-key YOUR_GEMINI_API_KEY
```

### Using Environment Variable for API Key

```bash
export GEMINI_API_KEY="your_api_key_here"

python nano_bananna.py \
  --input-dir cityscape/foggy/val/frankfurt \
  --output-dir nano_foggy_results
```

### Advanced Usage

Process only images with "0.01" in filename, skip existing outputs:

```bash
python nano_bananna.py \
  --input-dir cityscape/foggy/val/frankfurt \
  --output-dir nano_foggy_results \
  --filter "0.01" \
  --skip-existing
```

Customize model and dimensions:

```bash
python nano_bananna.py \
  --input-dir input_images \
  --output-dir output_images \
  --model gemini-2.5-flash-image-preview \
  --width 2048 \
  --height 1024 \
  --temperature 0.0 \
  --prompt "Remove fog and enhance visibility"
```

## Command Line Arguments

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `--input-dir` | Yes | - | Input directory containing foggy images |
| `--output-dir` | Yes | - | Output directory for defogged images |
| `--api-key` | No | - | Google Gemini API key (or use `GEMINI_API_KEY` env var) |
| `--model` | No | `gemini-2.5-flash-image-preview` | Gemini model to use |
| `--prompt` | No | `Need to remove fog...` | Prompt for defogging task |
| `--width` | No | `2048` | Target width for output images |
| `--height` | No | `1024` | Target height for output images |
| `--temperature` | No | `0.0` | Temperature for model generation |
| `--filter` | No | `None` | Filter images by substring in filename |
| `--skip-existing` | No | `False` | Skip images that already have output files |

## Output

- Defogged images are saved with `_defogged.png` suffix
- Progress is shown for each image
- Summary statistics are displayed at the end
- Output directory is created automatically if it doesn't exist

## Example Output

```
Initializing Gemini client with model: gemini-2.5-flash-image-preview
Scanning input directory: cityscape/foggy/val/frankfurt
Applied filter '0.01': 50 images match
Found 50 images to process

[1/50] Processing: frankfurt_000000_000294_leftImg8bit_foggy_beta_0.01.png
Processing: cityscape/foggy/val/frankfurt/frankfurt_000000_000294_leftImg8bit_foggy_beta_0.01.png
  Saved to: nano_foggy_results/frankfurt_000000_000294_leftImg8bit_foggy_beta_0.01_defogged.png

...

============================================================
Processing Summary:
  Total images: 50
  Successfully processed: 48
  Skipped (existing): 0
  Errors: 2
============================================================
```

# Post Process Pipeline

# Post-Processing Pipeline Runner

This script provides a command-line interface (CLI) for running the rubric post-processing pipeline on one or more CSV files. It supports processing a single CSV, multiple CSVs, or all CSV files within a directory, and consolidates results into structured outputs.

The script acts as the **entry point** for batch and automated analysis workflows.

---

## Overview

The pipeline runner wraps the core post-processing logic and adds:

- Flexible input handling (single file, multiple files, or folder)
- Batch processing support
- Automatic output directory management
- Optional aggregation of results across multiple datasets
- CLI flags for verbosity and saving behavior

It is intended for large-scale or repeated analysis of rubric-based evaluation data.

---

## Features

- Process a single CSV file
- Process multiple CSV files in one command
- Process all CSV files within a directory
- Automatically creates per-file output subdirectories
- Supports custom rubric column selection
- Optional verbose logging
- Optional saving of intermediate and final CSV results
- Generates combined summary CSVs when processing multiple files

---

## Usage

### Process a Single CSV

```bash
python run_post_process_pipeline.py \
  --input_csv db/clean_up/dataset_no_duplicate_rows.csv \
  --output_dir results/single

```
### Process Multiple CSV Files
```bash
python run_post_process_pipeline.py \
  --input_csv file1.csv file2.csv file3.csv\
  --output_dir results/batch

```
### Process All CSVs in a Folder
```bash
python run_post_process_pipeline.py \
  --input_folder db/clean_up \
  --output_dir results/folder_batch

```

### CLI 
| Argument         | Required | Default                 | Description                    |
| ---------------- | -------- | ----------------------- | ------------------------------ |
| `--input_csv`    | Yes*     | –                       | One or more input CSV files    |
| `--input_folder` | Yes*     | –                       | Folder containing CSV files    |
| `--output_dir`   | Yes      | –                       | Output directory for results   |
| `--rubric_cols`  | No       | Default pipeline rubric | Rubric column names to analyze |
| `--no-verbose`   | No       | False                   | Suppress verbose logging       |
| `--no-save`      | No       | False                   | Do not save output CSV files   |


### Specify Custom Rubric Columns
```bash

python run_post_process_pipeline.py \
  --input_folder db/clean_up \
  --output_dir results/custom_rubric \
  --rubric_cols visibility_restoration boundary_clarity

```

### Output Structure
When processing multiple CSV files, the output directory is organized as follows:

```bash
output_dir/
├── csv_file_1/
│   ├── summary_statistics.csv
│   ├── rubric_statistics.csv
│   └── ...
├── csv_file_2/
│   ├── summary_statistics.csv
│   ├── rubric_statistics.csv
│   └── ...
├── combined_model_stats.csv
└── combined_model_rubric_stats.csv


```


# Human Survey Post-Processing

This script post-processes human judge survey scores for defogging models, aggregates results across rubric categories, and generates statistical summaries, visualizations, and a comprehensive report.

It is intended for analyzing subjective evaluation data collected via CSV files (e.g., Google Forms or Qualtrics) and producing publication-ready tables and figures.

---

## Features

- Parses human-rated scores across multiple models and rubric categories
- Validates and clips scores to the expected 0–5 range
- Computes mean, standard deviation (sample), and standard error of the mean (SEM)
- Aggregates both per-category and total scores
- Automatically identifies models and images via column-name patterns
- Generates multiple visualizations:
  - Total score bar charts
  - Grouped category comparison bars
  - Heatmaps
  - Radar charts
- Exports CSV summaries and a structured text report

---

## Input Format

The script expects a CSV file where:

- Each row corresponds to a human respondent
- Each column corresponds to a model–image–category score
- Column names contain:
  - A category name (e.g., `Visibility Restoration`)
  - A model identifier (e.g., `dehazformer`, `dehazformer trained`, `defoggedflux best`)

Example column name:


Scores are assumed to be on a 0–5 Likert scale.

---

## Evaluated Categories

The following rubric categories are analyzed (configurable in the script):

- Visibility Restoration
- Boundary Clarity
- Object Consistency

---

## Usage

```bash
python survey_postprocess.py \
  --input_csv path/to/human_judgments.csv \
  --output_dir humanjudge

```

