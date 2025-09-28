# Cityscapes Dehazing Evaluation Project

This project evaluates dehazing models on object detection downstream tasks using the Cityscapes dataset. The project compares object detection performance on ground truth images versus foggy images with different intensities.

## Dataset Structure

The project uses four subdatasets from Cityscapes:

1. **ground_truth** (`leftImg8bit`): Original clear images without fog
2. **foggy_beta_0.02**: Foggy images with beta=0.02 (highest fog intensity)
3. **foggy_beta_0.01**: Foggy images with beta=0.01 (medium fog intensity)
4. **foggy_beta_0.005**: Foggy images with beta=0.005 (lowest fog intensity)

## Project Structure

```
WACV/
├── citytococo/
│   └── data/
│       └── cityscapes/
│           ├── leftImg8bit/          # Ground truth images
│           ├── foggy_beta_0.02/      # High fog intensity
│           ├── foggy_beta_0.01/      # Medium fog intensity
│           ├── foggy_beta_0.005/     # Low fog intensity
│           ├── gtfine/               # Ground truth annotations
│           └── annotations/          # COCO format annotations
├── models/                           # YOLO models storage
├── dataset_analysis.py              # Main analysis script
└── README.md
```

## Installation

This project uses `uv` for dependency management. Make sure you have `uv` installed and the virtual environment activated:

```bash
# Activate virtual environment
source .venv/bin/activate

# Install dependencies
uv sync
```

## Usage

### Running the Complete Analysis

The main script `dataset_analysis.py` performs the following steps:

1. **Load Datasets**: Loads all four Cityscapes subdatasets into FiftyOne
2. **Download YOLO Models**: Downloads YOLOv8n, YOLOv8s, and YOLOv8m models
3. **Visualize Datasets**: Optionally launches FiftyOne App for dataset visualization
4. **Object Detection**: Runs YOLO models on all datasets
5. **Analyze Results**: Compares performance metrics across datasets
6. **Create Visualizations**: Generates performance comparison charts

```bash
# Run the complete analysis
uv run dataset_analysis.py
```
