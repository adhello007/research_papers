import torch 
from PIL import Image
from model import UNET
import numpy as np 
import albumentations as A 
from albumentations.pytorch import ToTensorV2
input_image = "data/test_images/7b807c756d15_12.jpg"
model_checkpoint = "/home/akshat/lab/deep_learning/research_papers/segmentation/mycheckpoint_v1.tar"
IMAGE_HEIGHT= 160 
IMAGE_WIDTH = 240
image = np.array(Image.open(input_image).convert('RGB'))
test_transform = A.Compose([
    A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH), 
    A.Normalize(
        mean=[0.0,0.0,0.0], 
        std=[1.0,1.0,1.0], 
        max_pixel_value=255.0
    ), 
    ToTensorV2()
])
image_tensor = test_transform(image=image)
image_tensor = image_tensor["image"]

image_tensor = image_tensor.unsqueeze(0)
image_tensor = image_tensor.to("cuda")
checkpoint = torch.load('mycheckpoint_v1.tar', map_location="cuda")

model = UNET()
model.load_state_dict(checkpoint['state_dict']) 
model.to('cuda')
model.eval() 

with torch.no_grad(): 
    prediction = model(image_tensor)

    prediction = torch.sigmoid(prediction)

    prediction = (prediction > 0.5).float()

output_mask = prediction.squeeze().cpu().numpy()

output_mask = (output_mask * 255.0).astype(np.uint8)

final_image = Image.fromarray(output_mask)
final_image.save('output_mask1.jpg')



