import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import sys
import os
from sklearn.model_selection import train_test_split # New import

# Setup Path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import config
from src.model import Unet
from src.dataset import CaravanDataset
from src.transforms import get_train_transforms, get_val_transforms
from src.utils import (
    save_checkpoint,
    load_checkpoint,
    get_val_metrics,
    log_to_csv
)

def train_epoch(loader, model, optimizer, loss_fn, scaler):
    loop = tqdm(loader, leave=False)
    total_loss = 0.0

    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(device=config.DEVICE)
        targets = targets.float().unsqueeze(1).to(device=config.DEVICE)

        with torch.amp.autocast('cuda'):
            predictions = model(data)
            loss = loss_fn(predictions, targets)

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        loop.set_postfix(loss=loss.item())
        
    return total_loss / len(loader)

def main():
    # setup Transforms
    train_transform = get_train_transforms(config.IMAGE_HEIGHT, config.IMAGE_WIDTH)
    val_transform = get_val_transforms(config.IMAGE_HEIGHT, config.IMAGE_WIDTH)
    
    # setup Model & Training Components
    model = Unet(in_channels=3, out_channels=1).to(config.DEVICE)
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    scaler = torch.amp.GradScaler('cuda')

    # learning Rate Scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.1, patience=2
    )

    all_images = os.listdir(config.TRAIN_IMG_DIR)
    train_images, val_images = train_test_split(all_images, test_size=0.2, random_state=42)

    print(f"Total Images: {len(all_images)}")
    print(f"Training on:  {len(train_images)}")
    print(f"Validating on: {len(val_images)}")

    
    train_ds = CaravanDataset(
        image_dir=config.TRAIN_IMG_DIR,
        mask_dir=config.TRAIN_MASK_DIR,
        images_list=train_images,
        transform=train_transform,
    )
    val_ds = CaravanDataset(
        image_dir=config.TRAIN_IMG_DIR,
        mask_dir=config.TRAIN_MASK_DIR,
        images_list=val_images,  
        transform=val_transform,
    )

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=config.BATCH_SIZE, num_workers=config.NUM_WORKERS, 
        pin_memory=config.PIN_MEMORY, shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=config.BATCH_SIZE, num_workers=config.NUM_WORKERS, 
        pin_memory=config.PIN_MEMORY, shuffle=False
    )

    best_dice = 0.0
    
    if config.LOAD_MODEL:
        load_checkpoint(torch.load("my_checkpoint.pth.tar"), model)

    #Training Loop
    print(f"\n{'Epoch':<5} | {'Train Loss':<10} | {'Val Loss':<10} | {'Val Dice':<10} | {'Best Dice':<10} | {'LR':<10}")
    print("-" * 75)

    for epoch in range(config.NUM_EPOCHS):
        avg_train_loss = train_epoch(train_loader, model, optimizer, loss_fn, scaler)

        avg_val_loss, avg_val_dice = get_val_metrics(val_loader, model, loss_fn, device=config.DEVICE)
        
        scheduler.step(avg_val_loss)
        
        current_lr = optimizer.param_groups[0]['lr']

        print(f"{epoch+1:<5} | {avg_train_loss:.5f}    | {avg_val_loss:.5f}    | {avg_val_dice:.5f}     | {best_dice:.5f}     | {current_lr:.1e}")

        if avg_val_dice > best_dice:
            best_dice = avg_val_dice
            save_checkpoint({"state_dict": model.state_dict(), "optimizer": optimizer.state_dict()}, filename="best_model.pth.tar")

        log_to_csv(epoch+1, avg_train_loss, avg_val_loss, avg_val_dice, best_dice)

if __name__ == "__main__":
    main()