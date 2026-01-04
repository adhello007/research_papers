import torch
import numpy as np
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import sys
import os
import matplotlib.pyplot as plt

# Setup paths
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import config
from src.model import Unet

def predict_single_image(image_path, model_path, output_path="output.png"):
    # 1. Load the High-Res Image
    original_image = np.array(Image.open(image_path).convert("RGB"))
    orig_height, orig_width = original_image.shape[:2]

    # 2. Preprocess (Resize to Model Size)
    transform = A.Compose([
        A.Resize(height=config.IMAGE_HEIGHT, width=config.IMAGE_WIDTH),
        A.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0], max_pixel_value=255.0),
        ToTensorV2(),
    ])
    
    transformed = transform(image=original_image)
    img_tensor = transformed["image"].unsqueeze(0).to(config.DEVICE) # Add batch dim

    # 3. Load Model
    model = Unet(in_channels=3, out_channels=1).to(config.DEVICE)
    checkpoint = torch.load(model_path, map_location=config.DEVICE)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    # 4. Inference
    print(f"Running inference on {image_path}...")
    with torch.no_grad():
        logits = model(img_tensor)
        probs = torch.sigmoid(logits)
        # Binarize: 0 or 1
        mask = (probs > 0.5).float()

    # 5. Post-process (Upscale Mask to Original Resolution)
    # We move to CPU and convert to numpy
    mask_np = mask.squeeze().cpu().numpy() # Shape: (160, 240)
    
    # Resize mask back to Original Size (Use Nearest Neighbor to keep edges sharp)
    mask_img = Image.fromarray((mask_np * 255).astype(np.uint8))
    mask_resized = mask_img.resize((orig_width, orig_height), resample=Image.NEAREST)
    
    # 6. Overlay and Save
    # Create a nice visualization: Original Image + Green Overlay
    mask_rgb = np.array(mask_resized)
    
    # Create an overlay (Green tint)
    overlay = original_image.copy()
    # Where mask is white (255), we add green color
    # Green channel boost
    overlay[mask_rgb == 255, 1] = 255 
    
    # Blend: 70% Original + 30% Green Overlay
    blended = Image.fromarray(original_image).convert("RGBA")
    overlay_img = Image.fromarray(overlay).convert("RGBA")
    final_result = Image.blend(blended, overlay_img, alpha=0.3)
    
    final_result.save(output_path)
    print(f"Saved result to {output_path}")

if __name__ == "__main__":
    # Example Usage
    # Grab a random test image or define a specific path
    TEST_IMG = "data/train_images/00087a6bd4dc_01.jpg" # Replace with a real path
    CHECKPOINT = "my_checkpoint.pth.tar"
    
    predict_single_image(TEST_IMG, CHECKPOINT)