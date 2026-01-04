import albumentations as A
from albumentations.pytorch import ToTensorV2

def get_train_transforms(height, width):
    """
    Returns the training transforms with data augmentation.
    """
    return A.Compose(
        [
            A.Resize(height=height, width=width),
            A.Rotate(limit=35, p=1.0),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.1),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )

def get_val_transforms(height, width):
    """
    Returns the validation transforms (No augmentation, just resizing/normalization).
    """
    return A.Compose(
        [
            A.Resize(height=height, width=width),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )