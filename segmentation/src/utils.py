import torch
import torchvision
import pandas as pd 
import os 
def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)

def load_checkpoint(checkpoint, model):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])

def log_to_csv(epoch, train_loss, val_loss, val_dice, best_dice, filename="training_log.csv"):
    # Updated to include Validation Loss column
    data = {
        "Epoch": [epoch],
        "Train Loss": [f"{train_loss:.5f}"],
        "Val Loss": [f"{val_loss:.5f}"], 
        "Val Dice": [f"{val_dice:.5f}"],
        "Best Dice": [f"{best_dice:.5f}"]
    }
    df = pd.DataFrame(data)
    if not os.path.isfile(filename):
        df.to_csv(filename, index=False)
    else:
        df.to_csv(filename, mode='a', header=False, index=False)

def get_val_metrics(loader, model, loss_fn, device="cuda"):
    """
    Calculates BOTH Validation Loss and Dice Score.
    """
    dice_score = 0
    total_val_loss = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device).unsqueeze(1) # Shape: (N, 1, H, W)
            
            # 'amp' for consistency, though strictly not needed for eval
            with torch.amp.autocast('cuda'): 
                logits = model(x)
                # Calculate Validation Loss
                loss = loss_fn(logits, y.float())
            
            total_val_loss += loss.item()

            # 2. Dice Calculation
            preds = torch.sigmoid(logits)
            preds = (preds > 0.5).float()
            dice_score += (2 * (preds * y).sum()) / ((preds + y).sum() + 1e-8)

    avg_val_loss = total_val_loss / len(loader)
    avg_dice = dice_score / len(loader)
    
    model.train()
    return avg_val_loss, avg_dice.item()

def save_predictions_as_imgs(loader, model, folder="saved_images/", device="cuda"):
    model.eval()
    for idx, (x, y) in enumerate(loader):
        x = x.to(device=device)
        with torch.no_grad():
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
        
        # Save the images
        torchvision.utils.save_image(preds, f"{folder}/pred_{idx}.png")
        torchvision.utils.save_image(y.unsqueeze(1), f"{folder}/correct_{idx}.png")
        
        # Stop after saving one batch (we don't need 5000 images)
        if idx == 0:
            break

    model.train()