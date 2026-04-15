# SG-Diff

This repository provides the official implementation of **SG-Diff: Semantic–Geometric Prior Guided Diffusion for Personalized Dental Cone-Beam CT Artifact Reduction**. The source code and datasets are being prepared for open-source release.

## Environment Setup

We recommend using `conda` or standard Python virtual environments. This project relies on `torch` and HuggingFace ecosystems such as `diffusers` and `accelerate`.

```bash
# Recommended: Create a new internal environment
# conda create -n sgdiff python=3.10
# conda activate sgdiff

# Install core dependencies:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install diffusers transformers accelerate datasets
pip install opencv-python scikit-image tqdm
pip install xformers
```

## Data Preparation & Splitting

Before training, generate the 5-fold cross-validation splits. This splits the raw data randomly into reproducible training and testing text files.

```bash
python datasets/make_5fold_splits.py --root ./datasets
python datasets/prepare_hf_dataset.py
```

## Stage 1: Prior-Incorporated Root Canal Morphology Identification

This module is responsible for:

1. Extracting the **Semantic Prior** and **Geometric Prior** from artifact-affected images.
2. Performing **Morphology Identification** to output morphology class identifiers.

The output from this module is necessary for guiding the Stage 2 Mixture-of-Experts (MoE) artifact reduction diffusion model.

### 1. Train the Prior Model

Navigate to the `Prior_Extractor` directory and execute the training script.

```bash
cd Prior_Extractor
bash run_train.sh
```

### 2. Extracting Priors and Identification

```bash
# Execute from within Prior_Extractor
python extract_priors.py --fold_idx 0 --split_mode test
cd ..
```

## Stage 2: Prior-Guided Artifact Reduction and Anatomy Restoration

This generative module is the core of our framework, responsible for:

1. Accepting the **Semantic Prior** and **Geometric Prior** from Stage 1 to securely lock in anatomical fidelity.
2. Utilizing a novel **Morphology-Aware Mixture-of-Experts (MoE) Diffusion Mechanism**. 
3. Synthesizing high-fidelity, artifact-free Dental CBCT images from the heavily degraded inputs.

### 1. Base Model Warm-up

Navigate to the `MoE_Diffusion` directory. First, fine-tune the base U-Net to accommodate the 14-channel input (8-channel Latents + 6-channel Semantic Prior):

```bash
cd MoE_Diffusion

accelerate launch --num_processes=1 --mixed_precision="fp16" \
    train.py \
    --pretrained_model_name_or_path="path/to/instruct-pix2pix-model-8ch" \
    --cache_dir="path/to/hf_cache" \
    --dataset_name="path/to/datasets-hf/fold0/train" \
    --original_image_column="file_name" \
    --edited_image_column="edited_image" \
    --edit_prompt_column="edit_prompt" \
    --prior_dir="path/to/dataset_priors_oof/seg_probs" \
    --resolution=256 --random_flip \
    --train_batch_size=16 \
    --gradient_accumulation_steps=4 \
    --gradient_checkpointing \
    --max_train_steps=1500 \
    --checkpointing_steps=500 \
    --checkpoints_total_limit=1 \
    --learning_rate=1e-5 \
    --freeze_backbone \
    --max_grad_norm=1 \
    --lr_warmup_steps=1000 \
    --conditioning_dropout_prob=0.05 \
    --seed=42 \
    --report_to=tensorboard \
    --output_dir="sgdiff-base-model"
```

### 2. MoE & ControlNet Joint Training

Freeze the U-Net backbone, inject the Geometric Prior (SDF) into the newly initialized ControlNet, and train the Morphology-Aware Mixture-of-Experts (MoE) heads using the `class_json` from Stage 1:

```bash
accelerate launch --num_processes=1 train-controlnet.py \
    --pretrained_model_name_or_path="path/to/instruct-pix2pix-model-8ch" \
    --unet_model_name_or_path="path/to/sgdiff-base-model" \
    --resolution=256 --random_flip \
    --train_batch_size=8 \
    --gradient_accumulation_steps=2 \
    --gradient_checkpointing \
    --max_train_steps=1500 \
    --checkpointing_steps=500 \
    --learning_rate=5e-05 \
    --max_grad_norm=1 \
    --lr_warmup_steps=0 \
    --dataset_name="path/to/dental-hf/fold0/train" \
    --original_image_column="file_name" \
    --edited_image_column="edited_image" \
    --edit_prompt_column="edit_prompt" \
    --prior_dir="path/to/dataset_priors_oof/seg_probs" \
    --sdf_dir="path/to/dataset_priors_oof/sdf_preds" \
    --output_dir="sgdiff-moe-model" \
    --report_to="tensorboard"
```

### 3. Inference

Run the end-to-end inference on the test set. The model will dynamically load the Morphology Class IDs to route experts, and compile a final evaluation report mapping PSNR and SSIM:

```bash
python test-controlnet.py \
    --model_path="path/to/instruct-pix2pix-model-8ch" \
    --unet_path="path/to/sgdiff-base-model" \
    --controlnet_path="path/to/sgdiff-moe-model" \
    --data_root="path/to/dental-hf/fold0/test" \
    --prior_dir="path/to/dataset_priors_oof/seg_probs" \
    --sdf_dir="path/to/dataset_priors_oof/sdf_preds" \
    --class_json_path="path/to/dataset_priors_oof/cls_preds" \
    --save_dir="path/to/test_outputs"
```
