import torch
from PIL import Image 
from model import UNet
import sys
import torchvision.transforms as transforms
import numpy as np
from albumentations.pytorch import ToTensorV2
import albumentations as A 

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model_path = sys.argv[1]

# Load model to device
checkpoint = torch.load(model_path, map_location=device)
model = UNet(n_channels=3, n_classes=1)
model.load_state_dict(checkpoint['state_dict'])
model.to(device)
model.eval()

# Resize image and normalise
transform = transforms.Compose([
    transforms.Resize((512,512)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.5,0.5,0.5],
        std=[0.5,0.5,0.5],
    ),
])

# Transform image and put to GPU
input_image = Image.open(sys.argv[2])
input_tensor = transform(input_image)
input_tensor = input_tensor.unsqueeze(0).to(device)

# Get the model output for the input imag.e
with torch.no_grad():
    output = model(input_tensor)
    binary_mask = (output > 0.5).float()

# Save Image
numpy_mask = binary_mask.cpu().numpy()
numpy_mask = np.squeeze(numpy_mask).astype(np.uint8) * 255
segmented_mask = Image.fromarray(numpy_mask)
segmented_mask.save('./predicted.jpg')
print("Image Saved")
