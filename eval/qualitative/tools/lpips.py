import lpips
import torch
from PIL import Image
import torchvision.transforms as transforms

# Load LPIPS model (AlexNet backbone by default)
loss_fn = lpips.LPIPS(net='vgg')  # options: 'alex', 'vgg', 'squeeze'

# Preprocessing
to_tensor = transforms.Compose([
    transforms.Resize((256,256)),   # optional: resize to standard size
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])  # scale to [-1,1]
])


def compute_lpips(img1_path, img2_path):
    # Load images
    img1 = Image.open(img1_path).convert('RGB')
    img2 = Image.open(img2_path).convert('RGB')

    # Preprocess images
    img1_tensor = to_tensor(img1).unsqueeze(0)  # add batch dimension
    img2_tensor = to_tensor(img2).unsqueeze(0)

    # Compute LPIPS
    with torch.no_grad():
        lpips_value = loss_fn(img1_tensor, img2_tensor)

    return lpips_value.item()