import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter 
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..')) # Fix import path
from src.transforms import get_train_transforms, get_val_transforms
# Import your modules

import config
from src.model import Unet
from src.dataset import CaravanDataset
from src.utils import (
    save_checkpoint,
    load_checkpoint,
    check_accuracy,
    save_predictions_as_imgs,
)

def train_fn(loader, model, optimizer, loss_fn, scaler, writer, step):
    loop = tqdm(loader)

    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(device=config.DEVICE)
        targets = targets.float().unsqueeze(1).to(device=config.DEVICE)

        # Forward
        with torch.cuda.amp.autocast():
            predictions = model(data)
            loss = loss_fn(predictions, targets)

        # Backward
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # Update tqdm loop
        loop.set_postfix(loss=loss.item())
        
        # Log to TensorBoard every batch
        writer.add_scalar('Training Loss', loss.item(), global_step=step)
        step += 1
        
    return step

def main():
    #1. Get Transforms (Clean & Modular)
    train_transform = get_train_transforms(config.IMAGE_HEIGHT, config.IMAGE_WIDTH)
    val_transform = get_val_transforms(config.IMAGE_HEIGHT, config.IMAGE_WIDTH)

    # 2. Model & Training Setup
    model = Unet(in_channels=3, out_channels=1).to(config.DEVICE)
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    scaler = torch.cuda.amp.GradScaler()
    
    # TensorBoard Writer
    writer = SummaryWriter(log_dir="runs/Carvana_Experiment_1")
    step = 0

    # 3. Loaders
    train_ds = CaravanDataset(
        image_dir=config.TRAIN_IMG_DIR,
        mask_dir=config.TRAIN_MASK_DIR,
        transform=train_transform,
    )
    val_ds = CaravanDataset(
        image_dir=config.TRAIN_IMG_DIR,
        mask_dir=config.TRAIN_MASK_DIR,
        transform=val_transform,
    )

    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
        shuffle=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
        shuffle=False,
    )

    if config.LOAD_MODEL:
        load_checkpoint(torch.load("my_checkpoint.pth.tar"), model)


    # 4. The Grand Loop
    for epoch in range(config.NUM_EPOCHS):
        print(f"\n--- Epoch {epoch+1}/{config.NUM_EPOCHS} ---")
        
        # A. Train
        step = train_fn(train_loader, model, optimizer, loss_fn, scaler, writer, step)

        # B. Save Model
        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        }
        save_checkpoint(checkpoint)

        # C. Check Accuracy
        check_accuracy(val_loader, model, device=config.DEVICE)

        # D. Print Examples to Folder
        save_predictions_as_imgs(
            val_loader, model, folder="saved_images/", device=config.DEVICE
        )

if __name__ == "__main__":
    main()