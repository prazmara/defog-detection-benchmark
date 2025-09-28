# Zero-shot defogging with FLUX

This subdirectory contains code to run zero-shot defogging using the FLUX model, which leverages Visual-Language Models (VLMs) for prompt-driven image restoration.

Due to resolution limitations of FLUX, we provide two modes to handle high-resolution images, as described in our paper:

1. **Resize Mode**: Downscales input images, defogs to small image, and then uses upscaling to restore the original resolution.

2. **Split Mode**: Splits input images into overlapping 1024×1024 tiles for higher resolution processing.

We also implement a Chain-of-Thought (CoT) prompting strategy to enhance defogging performance. This involves using a sequence of prompts to guide the model through the defogging process.

## How to run

```bash
# Resize path, CoT prompt (+ negative prompt)
python defog_main.py \
  --image-folder /path/to/images \
  --out-folder /path/to/out \
  --mode resize \
  --cot \
  --glob "*0.01.png"

# Split path (two 1024×1024 tiles), without CoT 
python defog_main.py \
  --image-folder /path/to/images \
  --out-folder /path/to/out \
  --mode split \
  --no-cot
```