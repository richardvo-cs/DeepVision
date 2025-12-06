"""
Visualization utilities for DeblurGAN.
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision.utils import make_grid, save_image


def tensor_to_image(tensor, denormalize=True):
    """
    Convert a tensor to a PIL Image.
    
    Args:
        tensor: Image tensor [C, H, W] or [B, C, H, W]
        denormalize: Whether to denormalize from [-1, 1] to [0, 1]
        
    Returns:
        PIL Image
    """
    if tensor.dim() == 4:
        tensor = tensor[0]  # Take first image from batch
    
    if denormalize:
        tensor = (tensor + 1) / 2
    
    tensor = tensor.clamp(0, 1)
    
    # Convert to numpy
    img = tensor.cpu().numpy().transpose(1, 2, 0)
    img = (img * 255).astype(np.uint8)
    
    return Image.fromarray(img)


def save_comparison(blur, sharp, fake, save_path, denormalize=True):
    """
    Save a side-by-side comparison of blur, generated, and sharp images.
    
    Args:
        blur: Blurred image tensor
        sharp: Sharp ground truth tensor
        fake: Generated deblurred tensor
        save_path: Path to save the comparison
        denormalize: Whether to denormalize tensors
    """
    if denormalize:
        blur = (blur + 1) / 2
        sharp = (sharp + 1) / 2
        fake = (fake + 1) / 2
    
    # Create comparison grid
    comparison = torch.cat([blur, fake, sharp], dim=3)
    grid = make_grid(comparison, nrow=1, normalize=False)
    
    save_image(grid, save_path)


def plot_training_curves(log_file, save_path=None):
    """
    Plot training curves from log file.
    
    Args:
        log_file: Path to training log
        save_path: Path to save the plot
    """
    # This would parse TensorBoard logs or a custom log file
    # Placeholder implementation
    pass


def visualize_features(model, image, layer_names=None):
    """
    Visualize intermediate feature maps from the model.
    
    Args:
        model: The neural network model
        image: Input image tensor
        layer_names: List of layer names to visualize
        
    Returns:
        Dictionary of feature visualizations
    """
    features = {}
    hooks = []
    
    def get_hook(name):
        def hook(module, input, output):
            features[name] = output.detach()
        return hook
    
    # Register hooks
    for name, module in model.named_modules():
        if layer_names is None or name in layer_names:
            hooks.append(module.register_forward_hook(get_hook(name)))
    
    # Forward pass
    with torch.no_grad():
        _ = model(image)
    
    # Remove hooks
    for hook in hooks:
        hook.remove()
    
    return features


def create_animation(frames, output_path, fps=10):
    """
    Create a GIF animation from a list of frames.
    
    Args:
        frames: List of PIL Images or tensors
        output_path: Path to save the GIF
        fps: Frames per second
    """
    if not frames:
        return
    
    # Convert tensors to PIL Images
    pil_frames = []
    for frame in frames:
        if torch.is_tensor(frame):
            frame = tensor_to_image(frame)
        pil_frames.append(frame)
    
    # Save as GIF
    pil_frames[0].save(
        output_path,
        save_all=True,
        append_images=pil_frames[1:],
        duration=1000 // fps,
        loop=0
    )

