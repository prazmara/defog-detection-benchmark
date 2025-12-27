"""Configuration constants for VLM Judge Pipeline."""

# Supported defogging models
SUPPORTED_MODELS = [
    "dehazeformer", "focalnet", "mitdense", "mitnh", "fluxnet", "nanobanana",
    "b01_dhft", "flux_non_cot", "flux_cot", "b01_dhf", "flux_split",
    "flux_split_cot", "flux_split_non_cot", "b01", "GT"
]

# CSV output fields
DEFAULT_FIELDS = [
    "model", "city", "basename",
    "foggy_path", "cand_path", "gt_path",
    "visibility_restoration", "visual_distortion", "boundary_clarity",
    "scene_consistency", "object_consistency", "perceived_detectability",
    "relation_consistency", "explanation"
]

# Cityscape dataset configuration
CITYSCAPE_CITIES = ["frankfurt", "lindau", "munster"]
DEFAULT_SAMPLE_SIZE = 50

# Azure OpenAI configuration
AZURE_API_VERSION = "2024-12-01-preview"
AZURE_ENDPOINT = "https://aminm-m4mh1l2m-eastus2.cognitiveservices.azure.com/"
DEPLOYMENT_NAME = "gpt-5-chat"

# File paths
CANDIDATE_PATHS_FILE = "candidate_paths.txt"
JSONL_OUTPUT_PATH = "outputs/combined_results.jsonl"
