export MODEL_DIR="stabilityai/stable-diffusion-xl-base-1.0"
export OUTPUT_DIR="path/to/save/checkpoints"

accelerate launch train_controlnet.py \
 --pretrained_model_name_or_path=$MODEL_DIR \
 --output_dir=$OUTPUT_DIR \
 --dataset_name=fusing/fill50k \
 --mixed_precision="fp16" \
 --resolution=1024 \
 --learning_rate=1e-5 \
 --num_train_epochs=4 \
 --train_batch_size=1 \
 --gradient_accumulation_steps=4 \
 --seed=42 \
 --checkpoints_total_limit 2 \

