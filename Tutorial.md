# SG-Diff



This repository provides the official implementation of **SG-Diff: Semantic–Geometric Prior Guided Diffusion for Personalized Dental Cone-Beam CT Artifact Reduction**. The source code and datasets are being prepared for open-source release.



**Stage 1: Prior-Incorporated Root Canal Morphology Identification.** 

This module is responsible for:
1. Extracting the **Semantic Prior** and **Geometric Prior** from artifact-affected images.
3. Performing **Morphology Identification** to output morphology class identifiers.

The output from this module is necessary for guiding the Stage 2 Mixture-of-Experts (MoE) artifact reduction diffusion model.



**Stage 2: Prior-Guided Artifact Reduction and Anatomy Restoration.**

This generative module is the core of our framework, responsible for:

1. Accepting the **Semantic Prior** and **Geometric Prior** from Stage 1 to securely lock in anatomical fidelity.
2. Utilizing a novel **Morphology-Aware Mixture-of-Experts (MoE) Diffusion Mechanism**. 
3. Synthesizing high-fidelity, artifact-free Dental CBCT images from the heavily degraded inputs.



**Execution Guide**

1. **Data Preparation & Splitting**
Before training, generate the 5-fold cross-validation splits. This splits the raw data randomly into reproducible training and testing text files.
```bash
python datasets/make_5fold_splits.py --root ./datasets
python datasets/prepare_hf_dataset.py
```

2. **Train the Prior Model**
Navigate to the `PriorExtractor` directory and execute the training script.
```bash
cd PriorExtractor
bash run_train.sh
```
3. **Extracting Priors and Identification**
```bash
cd PriorExtractor
python extract_priors.py --fold_idx 0 --split_mode test
```



