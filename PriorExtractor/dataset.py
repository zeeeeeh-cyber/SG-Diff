import os
import glob
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class DentalDataset(Dataset):
    def __init__(self, img_dir, mask_dir, sdf_dir, json_dir=None, transform=None, allowed_prefixes=None):

        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.sdf_dir = sdf_dir
        self.json_dir = json_dir
        self.transform = transform
        self.allowed_prefixes = None

        if allowed_prefixes is not None:
            self.allowed_prefixes = set(str(x).strip() for x in allowed_prefixes if str(x).strip())
        
        self.valid_samples = []
        
        self.artifact_files = sorted(glob.glob(os.path.join(img_dir, '*.png')))

        if self.allowed_prefixes is not None:
            self.artifact_files = [
                f for f in self.artifact_files
                if os.path.basename(f).split('_')[0].replace('.png', '') in self.allowed_prefixes
            ]
        

        for i, img_path in enumerate(self.artifact_files):
            basename = os.path.basename(img_path)
            
            prefix = basename.split('_')[0].replace('.png', '')
            
            mask_candidates = glob.glob(os.path.join(self.mask_dir, f"{prefix}_*_mask.png"))
            if not mask_candidates:
                mask_candidates = glob.glob(os.path.join(self.mask_dir, f"{prefix}_*_mask.npy"))
            
            sdf_candidates = glob.glob(os.path.join(self.sdf_dir, f"{prefix}_*_sdf.npy"))
            
            if mask_candidates and sdf_candidates:
                mask_path = mask_candidates[0]
                sdf_path = sdf_candidates[0]
                
                json_path = None
                if self.json_dir is not None:
                    json_candidates = glob.glob(os.path.join(self.json_dir, f"{prefix}_*_id*.json"))
                    if json_candidates:
                        json_path = json_candidates[0]
                
                self.valid_samples.append({
                    'img_path': img_path,
                    'mask_path': mask_path,
                    'sdf_path': sdf_path,
                    'json_path': json_path
                })
        if len(self.valid_samples) == 0:
             print(f"Warning: No valid samples found in {img_dir} with corresponding masks/sdfs in {mask_dir}/{sdf_dir}")
        else:
            print(f"Found {len(self.valid_samples)} valid samples out of {len(self.artifact_files)} 'artifactIma' images.")

    def __len__(self):
        return len(self.valid_samples)

    def __getitem__(self, idx):
        sample_info = self.valid_samples[idx]
        img_path = sample_info['img_path']
        mask_path = sample_info['mask_path']
        sdf_path = sample_info['sdf_path']
        json_path = sample_info['json_path']

        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(f"Image not found: {img_path}")

        if mask_path.endswith('.png'):
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        else:
            mask = np.load(mask_path)

        if mask is None:
             raise ValueError(f"Failed to load mask: {mask_path}")

        sdf = np.load(sdf_path)

        class_label = 0
        if json_path is not None and os.path.exists(json_path):
            import json
            with open(json_path, 'r', encoding='utf-8') as jf:
                label_data = json.load(jf)
                if "shapes" in label_data and len(label_data["shapes"]) > 0:
                    class_label = int(label_data["shapes"][0]["label"])

        image = cv2.resize(image, (256, 256), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, (256, 256), interpolation=cv2.INTER_NEAREST)
        sdf = cv2.resize(sdf, (256, 256), interpolation=cv2.INTER_LINEAR)

        image = image.astype(np.float32) / 255.0
        image = torch.from_numpy(image).unsqueeze(0) # (1, H, W)

        mask = torch.from_numpy(mask).long()

        sdf = sdf.astype(np.float32)
        sdf = sdf / 100.0
        sdf = np.clip(sdf, -1.0, 1.0)
        sdf = torch.from_numpy(sdf).unsqueeze(0)
        
        sample = {'image': image, 'mask': mask, 'sdf': sdf, 'class_label': class_label, 'img_path': img_path}

        if self.transform:
            sample = self.transform(sample)

        return sample

if __name__ == "__main__":
    import sys
    
    img_dir = "datasets/artifactIma"
    mask_dir = "dataProcessing/dataset_mask"
    sdf_dir = "dataProcessing/dataset_sdf"
    
    if os.path.exists(img_dir):
        ds = DentalDataset(img_dir, mask_dir, sdf_dir)
        print(f"Dataset size: {len(ds)}")
        if len(ds) > 0:
            sample = ds[0]
            print(f"Image shape: {sample['image'].shape}, dtype: {sample['image'].dtype}")
            print(f"Mask shape: {sample['mask'].shape}, dtype: {sample['mask'].dtype}")
            print(f"SDF shape: {sample['sdf'].shape}, dtype: {sample['sdf'].dtype}")
            print("Dataset test passed!")
    else:
        print("Dataset dirs do not exist, skipping test.")
