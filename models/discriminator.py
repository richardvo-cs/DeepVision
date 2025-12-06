"""
PatchGAN Discriminator for DeblurGAN.

PatchGAN classifies whether overlapping patches of the image are real or fake,
rather than classifying the entire image. This encourages high-frequency local
realism and is more effective for image-to-image translation tasks.
"""
import torch
import torch.nn as nn


class PatchGANDiscriminator(nn.Module):
    """
    PatchGAN Discriminator.
    
    Outputs a matrix of predictions, where each element corresponds to a 
    receptive field patch in the input image. This enforces local structure
    and high-frequency details.
    
    Args:
        in_channels: Number of input channels (3 for RGB, 6 for conditional input)
        ndf: Number of filters in first conv layer
        n_layers: Number of conv layers (determines receptive field size)
    """
    
    def __init__(self, in_channels=6, ndf=64, n_layers=3):
        """
        Initialize PatchGAN discriminator.
        
        For conditional GAN: input is concatenation of blurred and (sharp or generated) image.
        So in_channels = 6 for RGB images.
        """
        super().__init__()
        
        # First layer: no normalization
        sequence = [
            nn.Conv2d(in_channels, ndf, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        ]
        
        # Intermediate layers with increasing filter count
        nf_mult = 1
        nf_mult_prev = 1
        
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, 
                         kernel_size=4, stride=2, padding=1, bias=False),
                nn.InstanceNorm2d(ndf * nf_mult),
                nn.LeakyReLU(0.2, inplace=True)
            ]
        
        # Second to last layer
        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                     kernel_size=4, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(ndf * nf_mult),
            nn.LeakyReLU(0.2, inplace=True)
        ]
        
        # Final layer: output 1 channel prediction map
        sequence += [
            nn.Conv2d(ndf * nf_mult, 1, kernel_size=4, stride=1, padding=1)
        ]
        
        self.model = nn.Sequential(*sequence)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        """Initialize network weights."""
        if isinstance(m, nn.Conv2d):
            nn.init.normal_(m.weight, 0.0, 0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.InstanceNorm2d):
            if m.weight is not None:
                nn.init.normal_(m.weight, 1.0, 0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
    
    def forward(self, blur_img, sharp_img):
        """
        Forward pass.
        
        Args:
            blur_img: Blurred input image [B, C, H, W]
            sharp_img: Sharp image (real or generated) [B, C, H, W]
            
        Returns:
            Patch predictions [B, 1, H', W'] where H', W' depend on n_layers
        """
        # Concatenate along channel dimension for conditional discrimination
        x = torch.cat([blur_img, sharp_img], dim=1)
        return self.model(x)


class MultiscaleDiscriminator(nn.Module):
    """
    Multi-scale PatchGAN discriminator.
    
    Uses multiple discriminators at different scales to capture both
    global structure and fine details.
    """
    
    def __init__(self, in_channels=6, ndf=64, n_layers=3, num_D=3):
        super().__init__()
        
        self.num_D = num_D
        
        # Create multiple discriminators
        self.discriminators = nn.ModuleList()
        for i in range(num_D):
            self.discriminators.append(
                PatchGANDiscriminator(in_channels, ndf, n_layers)
            )
        
        # Downsampling layer for multi-scale
        self.downsample = nn.AvgPool2d(3, stride=2, padding=1, count_include_pad=False)
    
    def forward(self, blur_img, sharp_img):
        """
        Forward pass through all scales.
        
        Returns list of discriminator outputs at each scale.
        """
        outputs = []
        
        for i, D in enumerate(self.discriminators):
            outputs.append(D(blur_img, sharp_img))
            
            # Downsample for next scale (except last)
            if i < self.num_D - 1:
                blur_img = self.downsample(blur_img)
                sharp_img = self.downsample(sharp_img)
        
        return outputs


class SpectralNormDiscriminator(nn.Module):
    """
    PatchGAN with Spectral Normalization for training stability.
    """
    
    def __init__(self, in_channels=6, ndf=64, n_layers=3):
        super().__init__()
        
        # First layer
        sequence = [
            nn.utils.spectral_norm(
                nn.Conv2d(in_channels, ndf, kernel_size=4, stride=2, padding=1)
            ),
            nn.LeakyReLU(0.2, inplace=True)
        ]
        
        # Intermediate layers
        nf_mult = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            
            sequence += [
                nn.utils.spectral_norm(
                    nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                             kernel_size=4, stride=2, padding=1)
                ),
                nn.LeakyReLU(0.2, inplace=True)
            ]
        
        # Second to last
        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        
        sequence += [
            nn.utils.spectral_norm(
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                         kernel_size=4, stride=1, padding=1)
            ),
            nn.LeakyReLU(0.2, inplace=True)
        ]
        
        # Final
        sequence += [
            nn.utils.spectral_norm(
                nn.Conv2d(ndf * nf_mult, 1, kernel_size=4, stride=1, padding=1)
            )
        ]
        
        self.model = nn.Sequential(*sequence)
    
    def forward(self, blur_img, sharp_img):
        x = torch.cat([blur_img, sharp_img], dim=1)
        return self.model(x)


if __name__ == "__main__":
    # Test the discriminator
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create model
    model = PatchGANDiscriminator(in_channels=6, ndf=64, n_layers=3).to(device)
    
    # Test forward pass
    blur = torch.randn(1, 3, 256, 256).to(device)
    sharp = torch.randn(1, 3, 256, 256).to(device)
    
    output = model(blur, sharp)
    
    print(f"Input shape: {blur.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output range: [{output.min().item():.3f}, {output.max().item():.3f}]")
    
    # Calculate receptive field
    # For n_layers=3: receptive field is 70x70 pixels
    print(f"Patch size (receptive field): ~70x70 for n_layers=3")
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of trainable parameters: {num_params:,}")

