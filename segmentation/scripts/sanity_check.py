import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import sys
import os

# This adds the project root to python path so we can import 'src' and 'config'
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import config
from src.dataset import CaravanDataset
from src.model import Unet
from src.transforms import get_train_transforms

def test_model_architecture():
    print("\n[1/2] Testing Model Architecture (The Plumbing)...")
    model = Unet(in_channels=3, out_channels=1).to(config.DEVICE)
    
    # Create a dummy tensor matching the config size
    # Shape: (Batch_Size, C, H, W)
    x = torch.randn((2, 3, config.IMAGE_HEIGHT, config.IMAGE_WIDTH)).to(config.DEVICE)
    
    try:
        preds = model(x)
        print(f"   Input Shape:  {x.shape}")
        print(f"   Output Shape: {preds.shape}")
        
        # Check matching spatial dimensions
        assert preds.shape[2] == x.shape[2] and preds.shape[3] == x.shape[3]
        print(" SUCCESS: Output spatial dimensions match input!")
        
    except Exception as e:
        print(f" FAILED: Model crashed. Error: {e}")
        sys.exit(1)

def test_data_augmentation():
    print("\n[2/2] Testing Data Integrity (Visual Check)...")
    
    #  Define Aggressive Transforms (Rotation) to verify mask moves with image
    test_transform = get_train_transforms(config.IMAGE_HEIGHT, config.IMAGE_WIDTH)

    # Load Dataset
    dataset = CaravanDataset(
        image_dir=config.TRAIN_IMG_DIR,
        mask_dir=config.TRAIN_MASK_DIR,
        transform=test_transform,
    )

    if len(dataset) == 0:
        print(" FAILED: No images found. Check paths in config.py")
        sys.exit(1)

    loader = DataLoader(dataset, batch_size=4, shuffle=True)
    images, masks = next(iter(loader))

    fig, ax = plt.subplots(4, 2, figsize=(10, 15))
    for i in range(4):
        # Permute: (C, H, W) -> (H, W, C) for Matplotlib
        img_np = images[i].permute(1, 2, 0).cpu().numpy()
        mask_np = masks[i].cpu().numpy()

        ax[i, 0].imshow(img_np)
        ax[i, 0].set_title("Rotated Image")
        ax[i, 0].axis("off")
        
        ax[i, 1].imshow(mask_np, cmap="gray")
        ax[i, 1].set_title("Rotated Mask")
        ax[i, 1].axis("off")

    output_file = "sanity_check_result.png"
    plt.tight_layout()
    plt.savefig(output_file)
    print(f" SUCCESS: saved visualization to '{output_file}'")
    print(" ACTION REQUIRED: Open this image. Ensure the white mask shape matches the car rotation.")

if __name__ == "__main__":
    if not os.path.exists(config.TRAIN_IMG_DIR):
        print(f" Error: Directory not found: {config.TRAIN_IMG_DIR}")
    else:
        test_model_architecture()
        test_data_augmentation()