import os
import argparse
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from model import MultiTaskUNet
from dataset import DentalDataset
from loss import JointLoss

def load_prefix_list(txt_path):
    with open(txt_path, "r") as f:
        xs = [line.strip() for line in f.readlines()]
    return [x for x in xs if x]

def train_model(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    train_txt = os.path.join(args.split_dir, f"fold{args.fold_idx}_train.txt")
    train_prefixes = load_prefix_list(train_txt)

    dataset = DentalDataset(
        img_dir=args.img_dir,
        mask_dir=args.mask_dir,
        sdf_dir=args.sdf_dir,
        json_dir=args.json_dir,
        allowed_prefixes=train_prefixes
    )

    val_percent = 0.1
    n_val = max(1, int(len(dataset) * val_percent))
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0))

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True, drop_last=False)
    
    print(f"Train size: {n_train}, Val size: {n_val}")

    model = MultiTaskUNet(n_channels=1, n_classes=6).to(device)

    class_weights = torch.tensor([1.0] + [10.0] * 5, dtype=torch.float32, device=device)
    criterion = JointLoss(num_classes=6, lambda_sdf=args.lambda_sdf, class_weights=class_weights).to(device)
    optimizer = optim.Adam(list(model.parameters()) + list(criterion.parameters()), lr=args.lr)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5)

    best_val_loss = float('inf')
    
    fold_ckpt_dir = os.path.join(args.checkpoint_dir, f"fold{args.fold_idx}")
    os.makedirs(fold_ckpt_dir, exist_ok=True)
    
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0
        epoch_seg_loss = 0
        epoch_sdf_loss = 0
        epoch_cls_loss = 0
        
        with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{args.epochs}', unit='img') as pbar:
            for batch in train_loader:
                images = batch['image'].to(device)
                masks = batch['mask'].to(device)
                sdfs = batch['sdf'].to(device)
                labels = batch['class_label'].to(device)

                seg_pred, sdf_pred, cls_pred = model(images)

                loss, seg_loss, sdf_loss, cls_loss = criterion(seg_pred, masks, sdf_pred, sdfs, cls_pred, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                pbar.set_postfix(**{'loss': loss.item(), 'seg': seg_loss.item(), 'sdf': sdf_loss.item(), 'cls': cls_loss.item()})
                pbar.update(images.shape[0])
                
                epoch_loss += loss.item() * images.shape[0]
                epoch_seg_loss += seg_loss.item() * images.shape[0]
                epoch_sdf_loss += sdf_loss.item() * images.shape[0]
                epoch_cls_loss += cls_loss.item() * images.shape[0]

        train_loss = epoch_loss / n_train
        print(f"Train Loss: {train_loss:.4f} (Seg: {epoch_seg_loss/n_train:.4f}, SDF: {epoch_sdf_loss/n_train:.4f}, Cls: {epoch_cls_loss/n_train:.4f})")

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                images = batch['image'].to(device)
                masks = batch['mask'].to(device)
                sdfs = batch['sdf'].to(device)
                labels = batch['class_label'].to(device)
                
                seg_pred, sdf_pred, cls_pred = model(images)
                loss, _, _, _ = criterion(seg_pred, masks, sdf_pred, sdfs, cls_pred, labels)
                val_loss += loss.item() * images.shape[0]

        val_loss /= n_val
        print(f"Val Loss: {val_loss:.4f}")
        
        scheduler.step(val_loss)

        # Save checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(fold_ckpt_dir, "best_model.pth"))
            print(f"Saved best model to {fold_ckpt_dir}")
    torch.save(model.state_dict(), os.path.join(fold_ckpt_dir, "last_model.pth"))
    print(f"Saved last model to {fold_ckpt_dir}")
    print("Training finished.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Model")
    parser.add_argument('--img_dir', type=str, default='datasets/artifactIma', help='Path to images')
    parser.add_argument('--mask_dir', type=str, default='dataProcessing/dataset_mask', help='Path to masks')
    parser.add_argument('--sdf_dir', type=str, default='dataProcessing/dataset_sdf', help='Path to SDFs')
    parser.add_argument('--json_dir', type=str, default='dataProcessing/dataset_json', help='Path to JSON files for classification label')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--lambda_sdf', type=float, default=1.0, help='Weight for SDF loss')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints', help='Directory to save checkpoints')
    parser.add_argument('--split_dir', type=str, required=True)
    parser.add_argument('--fold_idx', type=int, required=True)

    args = parser.parse_args()

    if not os.path.exists(args.checkpoint_dir):
        os.makedirs(args.checkpoint_dir)
        
    train_model(args)
