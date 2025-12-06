"""
Testing and inference script for DeblurGAN.

Supports:
- Testing on GoPro test set with metrics (PSNR, SSIM)
- Single image deblurring
- Batch inference on directory of images
"""
import os
import argparse
from pathlib import Path
import time

import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image
from tqdm import tqdm
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

from config import Config
from models import UNetGenerator
from data import get_dataloaders


class Tester:
    """DeblurGAN Tester and Inference."""
    
    def __init__(self, checkpoint_path, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Load model
        self.generator = UNetGenerator(
            in_channels=3,
            out_channels=3,
            ngf=Config.NGF
        ).to(self.device)
        
        self._load_checkpoint(checkpoint_path)
        self.generator.eval()
        
        # Transform for preprocessing
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
    
    def _load_checkpoint(self, path):
        """Load generator weights from checkpoint."""
        print(f"Loading checkpoint: {path}")
        checkpoint = torch.load(path, map_location=self.device)
        
        if 'generator_state_dict' in checkpoint:
            self.generator.load_state_dict(checkpoint['generator_state_dict'])
            print(f"Loaded from epoch {checkpoint.get('epoch', 'unknown')}")
        else:
            # Assume it's just the state dict
            self.generator.load_state_dict(checkpoint)
        
        print("Model loaded successfully")
    
    def _preprocess(self, image):
        """Preprocess image for inference."""
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')
        
        # Store original size
        original_size = image.size  # (W, H)
        
        # Resize to multiple of 16 for U-Net
        w, h = original_size
        new_w = (w // 16) * 16
        new_h = (h // 16) * 16
        
        if new_w != w or new_h != h:
            image = image.resize((new_w, new_h), Image.BILINEAR)
        
        # Transform
        tensor = self.transform(image).unsqueeze(0)
        
        return tensor, original_size
    
    def _postprocess(self, tensor, original_size=None):
        """Postprocess output tensor to image."""
        # Denormalize
        tensor = (tensor.squeeze(0) + 1) / 2
        tensor = tensor.clamp(0, 1)
        
        # Convert to PIL
        image = transforms.ToPILImage()(tensor.cpu())
        
        # Resize back to original if needed
        if original_size is not None:
            image = image.resize(original_size, Image.BILINEAR)
        
        return image
    
    @torch.no_grad()
    def deblur_image(self, image_path, output_path=None):
        """
        Deblur a single image.
        
        Args:
            image_path: Path to blurred image
            output_path: Path to save deblurred image (optional)
            
        Returns:
            Deblurred PIL Image
        """
        # Preprocess
        tensor, original_size = self._preprocess(image_path)
        tensor = tensor.to(self.device)
        
        # Inference
        start_time = time.time()
        output = self.generator(tensor)
        inference_time = time.time() - start_time
        
        # Postprocess
        result = self._postprocess(output, original_size)
        
        print(f"Inference time: {inference_time*1000:.1f}ms")
        
        # Save if output path provided
        if output_path:
            result.save(output_path)
            print(f"Saved to: {output_path}")
        
        return result
    
    @torch.no_grad()
    def deblur_directory(self, input_dir, output_dir):
        """
        Deblur all images in a directory.
        
        Args:
            input_dir: Directory with blurred images
            output_dir: Directory to save deblurred images
        """
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Find all images
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        image_files = [f for f in input_dir.iterdir() 
                       if f.suffix.lower() in image_extensions]
        
        print(f"Found {len(image_files)} images")
        
        total_time = 0
        for img_path in tqdm(image_files, desc="Deblurring"):
            output_path = output_dir / f"deblurred_{img_path.name}"
            
            # Preprocess
            tensor, original_size = self._preprocess(str(img_path))
            tensor = tensor.to(self.device)
            
            # Inference
            start_time = time.time()
            output = self.generator(tensor)
            total_time += time.time() - start_time
            
            # Postprocess and save
            result = self._postprocess(output, original_size)
            result.save(output_path)
        
        avg_time = total_time / len(image_files) * 1000
        print(f"\nAverage inference time: {avg_time:.1f}ms per image")
        print(f"Results saved to: {output_dir}")
    
    @torch.no_grad()
    def test_gopro(self, test_dir):
        """
        Test on GoPro test set and compute metrics.
        
        Args:
            test_dir: Path to GoPro test directory
            
        Returns:
            Dictionary with average PSNR and SSIM
        """
        _, test_loader = get_dataloaders(
            train_dir=test_dir,  # Not used
            test_dir=test_dir,
            batch_size=1,
            img_size=(720, 1280),  # Full resolution
            num_workers=0,
            use_random_crop=False
        )
        
        psnr_values = []
        ssim_values = []
        
        print(f"Testing on {len(test_loader)} images...")
        
        for batch in tqdm(test_loader, desc="Testing"):
            blur = batch['blur'].to(self.device)
            sharp = batch['sharp']
            
            # Generate
            fake = self.generator(blur)
            
            # Convert to numpy for metrics
            fake_np = ((fake.squeeze(0).cpu().numpy().transpose(1, 2, 0) + 1) / 2 * 255).astype(np.uint8)
            sharp_np = ((sharp.squeeze(0).numpy().transpose(1, 2, 0) + 1) / 2 * 255).astype(np.uint8)
            
            # Calculate metrics
            psnr_val = psnr(sharp_np, fake_np)
            ssim_val = ssim(sharp_np, fake_np, channel_axis=2)
            
            psnr_values.append(psnr_val)
            ssim_values.append(ssim_val)
        
        results = {
            'psnr': np.mean(psnr_values),
            'psnr_std': np.std(psnr_values),
            'ssim': np.mean(ssim_values),
            'ssim_std': np.std(ssim_values)
        }
        
        print(f"\n{'='*40}")
        print("Test Results:")
        print(f"  PSNR: {results['psnr']:.2f} ± {results['psnr_std']:.2f} dB")
        print(f"  SSIM: {results['ssim']:.4f} ± {results['ssim_std']:.4f}")
        print(f"{'='*40}")
        
        return results
    
    @torch.no_grad()
    def visualize_comparison(self, blur_path, sharp_path, output_path):
        """
        Create a side-by-side comparison image.
        
        Args:
            blur_path: Path to blurred image
            sharp_path: Path to ground truth sharp image
            output_path: Path to save comparison
        """
        # Load and process blur
        blur_tensor, _ = self._preprocess(blur_path)
        blur_tensor = blur_tensor.to(self.device)
        
        # Generate deblurred
        fake = self.generator(blur_tensor)
        
        # Load sharp
        sharp = Image.open(sharp_path).convert('RGB')
        sharp_tensor = self.transform(sharp).unsqueeze(0)
        
        # Resize all to same size
        h, w = blur_tensor.shape[2:]
        sharp_tensor = transforms.functional.resize(sharp_tensor, (h, w))
        
        # Denormalize
        blur_vis = (blur_tensor + 1) / 2
        fake_vis = (fake + 1) / 2
        sharp_vis = (sharp_tensor + 1) / 2
        
        # Concatenate: blur | deblurred | sharp
        comparison = torch.cat([blur_vis, fake_vis, sharp_vis.to(self.device)], dim=3)
        
        # Save
        save_image(comparison, output_path)
        print(f"Saved comparison to: {output_path}")
        
        # Calculate metrics
        fake_np = (fake_vis.squeeze(0).cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
        sharp_np = (sharp_vis.squeeze(0).numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
        
        psnr_val = psnr(sharp_np, fake_np)
        ssim_val = ssim(sharp_np, fake_np, channel_axis=2)
        
        print(f"PSNR: {psnr_val:.2f} dB, SSIM: {ssim_val:.4f}")


def main():
    parser = argparse.ArgumentParser(description='Test DeblurGAN')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--mode', type=str, default='single',
                        choices=['single', 'directory', 'gopro', 'compare'],
                        help='Inference mode')
    parser.add_argument('--input', type=str, required=True,
                        help='Input image/directory path')
    parser.add_argument('--output', type=str, default='./output',
                        help='Output path')
    parser.add_argument('--sharp', type=str, default=None,
                        help='Ground truth sharp image (for compare mode)')
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'cpu'], help='Device to use')
    
    args = parser.parse_args()
    
    # Create tester
    tester = Tester(args.checkpoint, args.device)
    
    if args.mode == 'single':
        # Single image deblurring
        output_path = args.output if args.output.endswith('.png') or args.output.endswith('.jpg') \
                      else os.path.join(args.output, 'deblurred.png')
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        tester.deblur_image(args.input, output_path)
        
    elif args.mode == 'directory':
        # Batch inference
        tester.deblur_directory(args.input, args.output)
        
    elif args.mode == 'gopro':
        # Test on GoPro dataset
        tester.test_gopro(args.input)
        
    elif args.mode == 'compare':
        # Create comparison
        if args.sharp is None:
            raise ValueError("--sharp argument required for compare mode")
        tester.visualize_comparison(args.input, args.sharp, args.output)


if __name__ == "__main__":
    main()

