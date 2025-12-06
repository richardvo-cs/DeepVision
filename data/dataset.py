"""
GoPro Dataset loader for motion deblurring.

The GoPro dataset contains pairs of blurred and sharp images captured
with a high-speed camera. Blurred images are created by averaging 
consecutive frames.

Dataset structure:
    GOPRO_Large/
    ├── train/
    │   ├── GOPR0372_07_00/
    │   │   ├── blur/
    │   │   │   ├── 000001.png
    │   │   │   └── ...
    │   │   └── sharp/
    │   │       ├── 000001.png
    │   │       └── ...
    │   └── ...
    └── test/
        └── ... (same structure)
"""
import os
import random
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image


class GoproDataset(Dataset):
    """
    GoPro Dataset for motion deblurring.
    
    Args:
        root_dir: Path to train or test directory
        img_size: Target image size (H, W) for resizing
        augment: Whether to apply data augmentation
        mode: 'train' or 'test'
    """
    
    def __init__(self, root_dir, img_size=(256, 256), augment=True, mode='train'):
        super().__init__()
        
        self.root_dir = Path(root_dir)
        self.img_size = img_size
        self.augment = augment and mode == 'train'
        self.mode = mode
        
        # Collect all image pairs
        self.image_pairs = self._load_image_pairs()
        
        print(f"Loaded {len(self.image_pairs)} image pairs from {root_dir}")
        
        # Define transforms
        self.resize = transforms.Resize(img_size, transforms.InterpolationMode.BILINEAR)
        
        # Normalization to [-1, 1] for GAN training
        self.normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
    
    def _load_image_pairs(self):
        """Load all blur/sharp image pairs from the dataset."""
        pairs = []
        
        # Iterate through scene directories
        for scene_dir in self.root_dir.iterdir():
            if not scene_dir.is_dir():
                continue
            
            blur_dir = scene_dir / 'blur'
            sharp_dir = scene_dir / 'sharp'
            
            if not blur_dir.exists() or not sharp_dir.exists():
                continue
            
            # Get all blur images
            blur_images = sorted(blur_dir.glob('*.png'))
            
            for blur_path in blur_images:
                sharp_path = sharp_dir / blur_path.name
                
                if sharp_path.exists():
                    pairs.append({
                        'blur': str(blur_path),
                        'sharp': str(sharp_path),
                        'scene': scene_dir.name,
                        'frame': blur_path.stem
                    })
        
        return pairs
    
    def _apply_augmentation(self, blur_img, sharp_img):
        """
        Apply synchronized augmentation to both images.
        
        Augmentations:
        - Random horizontal flip
        - Random vertical flip
        - Random rotation (90 degrees)
        - Random crop (then resize back)
        """
        # Random horizontal flip
        if random.random() > 0.5:
            blur_img = transforms.functional.hflip(blur_img)
            sharp_img = transforms.functional.hflip(sharp_img)
        
        # Random vertical flip
        if random.random() > 0.5:
            blur_img = transforms.functional.vflip(blur_img)
            sharp_img = transforms.functional.vflip(sharp_img)
        
        # Random rotation (0, 90, 180, 270 degrees)
        angle = random.choice([0, 90, 180, 270])
        if angle > 0:
            blur_img = transforms.functional.rotate(blur_img, angle)
            sharp_img = transforms.functional.rotate(sharp_img, angle)
        
        return blur_img, sharp_img
    
    def __len__(self):
        return len(self.image_pairs)
    
    def __getitem__(self, idx):
        """
        Get a blur/sharp image pair.
        
        Returns:
            dict with keys:
                - blur: Blurred image tensor [C, H, W]
                - sharp: Sharp image tensor [C, H, W]
                - path: Original blur image path
        """
        pair = self.image_pairs[idx]
        
        # Load images
        blur_img = Image.open(pair['blur']).convert('RGB')
        sharp_img = Image.open(pair['sharp']).convert('RGB')
        
        # Resize
        blur_img = self.resize(blur_img)
        sharp_img = self.resize(sharp_img)
        
        # Apply augmentation
        if self.augment:
            blur_img, sharp_img = self._apply_augmentation(blur_img, sharp_img)
        
        # Convert to tensors and normalize
        blur_tensor = self.normalize(blur_img)
        sharp_tensor = self.normalize(sharp_img)
        
        return {
            'blur': blur_tensor,
            'sharp': sharp_tensor,
            'path': pair['blur']
        }


class GoproRandomCropDataset(GoproDataset):
    """
    GoPro Dataset with random crop augmentation.
    
    Instead of resizing, this crops random patches from full-resolution images.
    This preserves natural blur characteristics better.
    """
    
    def __init__(self, root_dir, crop_size=(256, 256), augment=True, mode='train'):
        super().__init__(root_dir, img_size=crop_size, augment=augment, mode=mode)
        self.crop_size = crop_size
    
    def __getitem__(self, idx):
        pair = self.image_pairs[idx]
        
        # Load full-resolution images
        blur_img = Image.open(pair['blur']).convert('RGB')
        sharp_img = Image.open(pair['sharp']).convert('RGB')
        
        w, h = blur_img.size
        crop_h, crop_w = self.crop_size
        
        # Random crop position (same for both images)
        if self.mode == 'train':
            top = random.randint(0, max(0, h - crop_h))
            left = random.randint(0, max(0, w - crop_w))
        else:
            # Center crop for testing
            top = (h - crop_h) // 2
            left = (w - crop_w) // 2
        
        blur_img = transforms.functional.crop(blur_img, top, left, crop_h, crop_w)
        sharp_img = transforms.functional.crop(sharp_img, top, left, crop_h, crop_w)
        
        # Apply augmentation
        if self.augment:
            blur_img, sharp_img = self._apply_augmentation(blur_img, sharp_img)
        
        # Convert to tensors
        blur_tensor = self.normalize(blur_img)
        sharp_tensor = self.normalize(sharp_img)
        
        return {
            'blur': blur_tensor,
            'sharp': sharp_tensor,
            'path': pair['blur']
        }


def get_dataloaders(train_dir, test_dir, batch_size=4, img_size=(256, 256),
                    num_workers=4, use_random_crop=False):
    """
    Create training and testing dataloaders.
    
    Args:
        train_dir: Path to training data
        test_dir: Path to test data
        batch_size: Batch size
        img_size: Target image size
        num_workers: Number of data loading workers
        use_random_crop: Whether to use random crop instead of resize
        
    Returns:
        train_loader, test_loader
    """
    if use_random_crop:
        DatasetClass = GoproRandomCropDataset
        train_dataset = DatasetClass(train_dir, crop_size=img_size, augment=True, mode='train')
        test_dataset = DatasetClass(test_dir, crop_size=img_size, augment=False, mode='test')
    else:
        train_dataset = GoproDataset(train_dir, img_size=img_size, augment=True, mode='train')
        test_dataset = GoproDataset(test_dir, img_size=img_size, augment=False, mode='test')
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, test_loader


if __name__ == "__main__":
    # Test dataset loading
    import matplotlib.pyplot as plt
    
    # Update this path to your dataset location
    train_dir = "./GOPRO_Large/train"
    
    if os.path.exists(train_dir):
        dataset = GoproDataset(train_dir, img_size=(256, 256), augment=True)
        
        # Get a sample
        sample = dataset[0]
        
        print(f"Blur shape: {sample['blur'].shape}")
        print(f"Sharp shape: {sample['sharp'].shape}")
        print(f"Blur range: [{sample['blur'].min():.2f}, {sample['blur'].max():.2f}]")
        
        # Visualize
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        
        # Denormalize for visualization
        blur_vis = (sample['blur'].permute(1, 2, 0).numpy() + 1) / 2
        sharp_vis = (sample['sharp'].permute(1, 2, 0).numpy() + 1) / 2
        
        axes[0].imshow(blur_vis)
        axes[0].set_title('Blurred')
        axes[0].axis('off')
        
        axes[1].imshow(sharp_vis)
        axes[1].set_title('Sharp')
        axes[1].axis('off')
        
        plt.tight_layout()
        plt.savefig('dataset_sample.png')
        print("Saved sample visualization to dataset_sample.png")
    else:
        print(f"Dataset not found at {train_dir}")
        print("Please update the path or extract the GOPRO_Large dataset first.")

