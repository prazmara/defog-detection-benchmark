# VLM Judge Pipeline

A modular pipeline for evaluating defogged images using Vision Language Models.

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
