import torch 
from torch.utils.data import DataLoader
import albumentations as A 
from albumentations.pytorch import ToTensorV2
from model import UNET
from dataset import CarvanaDataset
from loss import DiceLoss
from tqdm import tqdm
DEVICE = "cuda" if torch.cuda.is_available() else "cpu" 
IMAGE_HEIGHT = 160
IMAGE_WIDTH = 240
LEARNING_RATE=3e-3
TRAIN_DIR = "data/train_images"
TRAIN_MASKS_DIR = "data/train_masks"
BATCH_SIZE = 16
EPOCHS=10

def train_fn(loader, model, optimizer, loss_fn, scaler): 
    """
    only one epoch of training 
    """
    loop = tqdm(loader)
    
    for batch_idx, (data, targets) in enumerate(loop): 
        data = data.to(DEVICE)
        targets = targets.float().unsqueeze(1).to(DEVICE)  #add channel dimension 
        # B , H, w --> B, 1, H, W
        with torch.amp.autocast(device_type=DEVICE, dtype=torch.float16): #for mixed precision
            predictions = model(data) 
            loss = loss_fn(predictions, targets)

        optimizer.zero_grad() 
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update() 

        loop.set_postfix(loss=f"{loss.item():.4f}") #updating tqdm progres bar


def main(): 

    train_transform = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH), 
            A.Rotate(limit=35, p=0.3), 
            A.HorizontalFlip(p=0.5), 
            A.VerticalFlip(p=0.1), 
            A.Normalize(
                mean = [0.0,0.0,0.0], 
                std=[1.0,1.0,1.0], 
                max_pixel_value=255.0, 
            ), 
            ToTensorV2(), 
        ]
    )

    model = UNET(in_channels=3, out_channels=1).to(DEVICE)

    loss_fn = DiceLoss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    scaler = torch.amp.GradScaler() #multiplies float 0.00001 gradients which are zerod by float16 precision on loss
    #and grad calculation. 

    train_ds = CarvanaDataset(
        TRAIN_DIR, 
        TRAIN_MASKS_DIR, 
        train_transform,
    )

    train_loader = DataLoader( 
        train_ds, 
        batch_size=BATCH_SIZE, 
        num_workers=2, 
        pin_memory=True, 
        shuffle=True
    )

    for epoch in range(EPOCHS): 
        print(f"Epoch {epoch+1}/{EPOCHS}")
        train_fn(train_loader, model, optimizer, loss_fn, scaler)

        checkpoint = { 
            "state_dict" : model.state_dict(), 
            "optimizer" : optimizer.state_dict(), 
        }
        
    torch.save(checkpoint, "mycheckpoint_v1.tar")
    print("Model Saved")


if __name__=="__main__": 
    main()