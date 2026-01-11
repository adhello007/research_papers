import torch
import numpy as np
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import sys
import os
import random

# Path Setup
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import config
from src.model import Unet

# --- CONFIGURATION ---
TEST_IMG_DIR = "segmentation/data/test_images/" 
OUTPUT_DIR = "segmentation/test_predictions/"    
MODEL_PATH = "segmentation/checkpoints/checkpoint_v1.pth.tar"  
NUM_IMAGES_TO_PREDICT = 10          
OVERLAY_COLOR = (0, 255, 0)         
OVERLAY_ALPHA = 0.4                 

def run_test_inference():
    # 1. Setup
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    device = config.DEVICE
    model = Unet(in_channels=3, out_channels=1).to(device)
    
    # Load Weights
    print(f"Loading model from {MODEL_PATH}...")
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    # Get List of Images
    all_images = os.listdir(TEST_IMG_DIR)
    # Randomly select N images
    selected_images = random.sample(all_images, min(len(all_images), NUM_IMAGES_TO_PREDICT))

    #Transform (Resize Only)
    transform = A.Compose([
        A.Resize(height=config.IMAGE_HEIGHT, width=config.IMAGE_WIDTH),
        A.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0], max_pixel_value=255.0),
        ToTensorV2(),
    ])

    print(f"Running inference on {len(selected_images)} images...")

    #Inference Loop
    for img_name in selected_images:
        img_path = os.path.join(TEST_IMG_DIR, img_name)
        
        # Load Original High-Res Image
        original_pil = Image.open(img_path).convert("RGB")
        orig_w, orig_h = original_pil.size
        
        # Preprocess
        original_np = np.array(original_pil)
        transformed = transform(image=original_np)
        img_tensor = transformed["image"].unsqueeze(0).to(device)

        # Predict
        with torch.no_grad():
            preds = torch.sigmoid(model(img_tensor))
            mask = (preds > 0.5).float()

        # Post-Process (Resize Mask to Original Size)
        mask_np = mask.cpu().squeeze().numpy()
        mask_pil = Image.fromarray((mask_np * 255).astype(np.uint8))
        mask_resized = mask_pil.resize((orig_w, orig_h), resample=Image.NEAREST)

        # 5. Create Overlay
        color_layer = Image.new("RGB", (orig_w, orig_h), OVERLAY_COLOR)
        
        mask_resized_l = mask_resized.convert("L") # Ensure grayscale
        overlay_img = Image.composite(color_layer, original_pil, mask_resized_l)
        final_output = Image.blend(original_pil, overlay_img, OVERLAY_ALPHA)

        save_path = os.path.join(OUTPUT_DIR, f"pred_{img_name}")
        final_output.save(save_path)
        print(f"Saved: {save_path}")

if __name__ == "__main__":
    run_test_inference()