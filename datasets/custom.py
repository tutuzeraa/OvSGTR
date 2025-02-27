import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision.transforms import Compose, Resize, ToTensor, Normalize

class CustomDataset(Dataset):
    """A dataset for loading custom images without annotations for inference."""
    def __init__(self, args, transform, image_set='val'):
        self.image_dir = args.data_path
        self.image_files = [f for f in os.listdir(self.image_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
        if not self.image_files:
            raise ValueError(f"No images found in {self.image_dir}")
        
        self.transform = transform 

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_files[idx])
        image = Image.open(img_path).convert('RGB')
        image = self.transform(image)
        # Dummy target for compatibility; assumes model can handle empty annotations
        target = {
            'image_id': torch.tensor([idx], dtype=torch.int64),
            'annotations': []
        }
        return image, target