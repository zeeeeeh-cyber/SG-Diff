import os
import argparse
import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

from model import MultiTaskUNet
from dataset import DentalDataset

def load_prefix_list(txt_path):
    with open(txt_path, "r") as f:
        xs = [line.strip() for line in f.readlines()]
    return [x for x in xs if x]

def extract_priors(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = MultiTaskUNet(n_channels=1, n_classes=6).to(device)

    checkpoint_path = os.path.join(args.checkpoint_dir, f"fold{args.fold_idx}", 'best_model.pth')
    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint not found at {checkpoint_path}")
        return
        
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    print(f"Loaded model from {checkpoint_path}")
    
    allowed_prefixes = None
    if args.split_dir and args.split_mode:
        txt_name = f"fold{args.fold_idx}_{args.split_mode}.txt"
        txt_path = os.path.join(args.split_dir, txt_name)
        if os.path.exists(txt_path):
            allowed_prefixes = load_prefix_list(txt_path)
            print(f"Loaded {len(allowed_prefixes)} prefixes from {txt_path}")
        else:
            print(f"Error: Split file not found at {txt_path}")
            return
    else:
        print("Extracting priors for ALL images in directory (No split specified)")

    dataset = DentalDataset(
        img_dir=args.img_dir,
        mask_dir=args.mask_dir,
        sdf_dir=args.sdf_dir,
        allowed_prefixes=allowed_prefixes
    )
    
    batch_size = 16
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    seg_dir = os.path.join(args.output_dir, 'seg_probs')
    sdf_dir = os.path.join(args.output_dir, 'sdf_preds')
    cls_dir = os.path.join(args.output_dir, 'cls_preds')
    os.makedirs(seg_dir, exist_ok=True)
    os.makedirs(sdf_dir, exist_ok=True)
    os.makedirs(cls_dir, exist_ok=True)
    
    print(f"Starting extraction for {len(dataset)} samples (Fold: {args.fold_idx})...")
    
    processed = 0
    with torch.no_grad():
        for batch in tqdm(data_loader):
            images = batch['image'].to(device)
            img_paths = batch['img_path'] 

            seg_logits, sdf_pred, cls_logits = model(images)
            
            for j in range(len(images)):
                img_path = img_paths[j]
                basename = os.path.basename(img_path)
                prefix = basename.split('_')[0]

                probs = torch.softmax(seg_logits[j:j+1], dim=1)
                probs_np = probs.squeeze(0).cpu().numpy().astype(np.float16)
                
                sdf_np = sdf_pred[j].cpu().numpy().astype(np.float16)

                cls_id = torch.argmax(cls_logits[j]).item()
                
                np.save(os.path.join(seg_dir, f"{prefix}_seg.npy"), probs_np)
                np.save(os.path.join(sdf_dir, f"{prefix}_sdf.npy"), sdf_np)

                import json
                with open(os.path.join(cls_dir, f"{prefix}_cls.json"), 'w') as f:
                    json.dump({"predicted_class": cls_id, "morphology_label": str(cls_id)}, f)
                
                processed += 1
                if args.num_samples and processed >= args.num_samples:
                    print(f"Reached limit of {args.num_samples} samples.")
                    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract priors")
    
    parser.add_argument('--img_dir', type=str, default='datasets/artifactIma')
    parser.add_argument('--mask_dir', type=str, default='dataProcessing/dataset_mask')
    parser.add_argument('--sdf_dir', type=str, default='dataProcessing/dataset_sdf')
    
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints_dental365')
    parser.add_argument('--split_dir', type=str, default='datasets/splits')
    parser.add_argument('--fold_idx', type=int, required=True, help='Which fold to use')
    parser.add_argument('--split_mode', type=str, choices=['train', 'test', 'none'], default='test', 
                        help='Which split to extract')
    
    parser.add_argument('--output_dir', type=str, default='./dataset_priors_oof')
    parser.add_argument('--num_samples', type=int, default=None)
    
    args = parser.parse_args()
    
    if args.split_mode == 'none':
        args.split_mode = None
        
    extract_priors(args)
