import torch
import pandas as pd
import numpy as np
from PIL import Image
import os
import sys
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import config
from src.dataset import CaravanDataset
from src.model import Unet
from src.transforms import get_val_transforms

def save_failure_visualizations(model, filenames, output_dir="error_analysis"):
    """
    Saves a 3-panel image (Original | Ground Truth | Prediction) for inspection.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print(f"\nGeneratings visual reports for {len(filenames)} worst failures...")
    
    # Transform for inference (Resize -> Tensor)
    transform = A.Compose([
        A.Resize(height=config.IMAGE_HEIGHT, width=config.IMAGE_WIDTH),
        A.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0], max_pixel_value=255.0),
        ToTensorV2(),
    ])

    model.eval()
    
    for img_name in filenames:
        # Paths
        img_path = os.path.join(config.TRAIN_IMG_DIR, img_name)
        mask_path = os.path.join(config.TRAIN_MASK_DIR, img_name.replace(".jpg", "_mask.gif"))
        
        # load Data
        original_img = np.array(Image.open(img_path).convert("RGB"))
        original_mask = np.array(Image.open(mask_path).convert("L"))
        
        #preprocess for Model
        augmented = transform(image=original_img)
        img_tensor = augmented["image"].unsqueeze(0).to(config.DEVICE)
        
        #predict
        with torch.no_grad():
            preds = torch.sigmoid(model(img_tensor))
            pred_mask = (preds > 0.5).float().cpu().squeeze().numpy()
        
        #visualization Prep
        #resize masks back to original image size for clear comparison
        h, w, _ = original_img.shape
        pred_mask_img = Image.fromarray((pred_mask * 255).astype(np.uint8)).resize((w, h), Image.NEAREST)
        
        # ccreate a blank canvas: Width = w*3 (Original + GT + Pred), Height = h
        combined_img = Image.new('RGB', (w * 3, h))
        
        # paste Original
        combined_img.paste(Image.fromarray(original_img), (0, 0))
        
        # paste Ground Truth (White)
        combined_img.paste(Image.fromarray(original_mask).convert("RGB"), (w, 0))
        
        # paste Prediction (White)
        combined_img.paste(pred_mask_img.convert("RGB"), (w * 2, 0))
        
        # save
        save_path = os.path.join(output_dir, f"fail_{img_name}")
        combined_img.save(save_path)
        print(f"Saved: {save_path}")


def analyze_failures(model_path):
    # RECREATE THE SPLIT
    all_images = os.listdir(config.TRAIN_IMG_DIR)
    _, val_images = train_test_split(all_images, test_size=0.2, random_state=42)
    
    print(f"Analyzing {len(val_images)} validation images...")

    # Setup
    model = Unet(in_channels=3, out_channels=1).to(config.DEVICE)
    checkpoint = torch.load(model_path, map_location=config.DEVICE)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    val_transform = get_val_transforms(config.IMAGE_HEIGHT, config.IMAGE_WIDTH)
    val_ds = CaravanDataset(
        image_dir=config.TRAIN_IMG_DIR,
        mask_dir=config.TRAIN_MASK_DIR,
        images_list=val_images,
        transform=val_transform
    )
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False)

    # Collect Scores
    results = []
    
    with torch.no_grad():
        for idx, (x, y) in enumerate(tqdm(val_loader)):
            x = x.to(config.DEVICE)
            y = y.to(config.DEVICE).unsqueeze(1)
            
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
            
            dice = (2 * (preds * y).sum()) / ((preds + y).sum() + 1e-8)
            
            results.append({
                "image": val_images[idx],
                "dice": dice.item()
            })

    # sort & Report
    df = pd.DataFrame(results)
    df = df.sort_values(by="dice", ascending=True)
    
    print("\nTOP 10 WORST FAILURES:")
    print(df.head(10))
    df.to_csv("failure_analysis.csv", index=False)
    
    # GENERATE IMAGES (The New Part)
    worst_files = df.head(20)["image"].tolist()
    save_failure_visualizations(model, worst_files)

if __name__ == "__main__":
    analyze_failures("checkpoints/checkpoint_v3.pth.tar")