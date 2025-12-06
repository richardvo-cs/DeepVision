"""
U-Net Generator for DeblurGAN.

The U-Net architecture uses skip connections to preserve spatial information
at multiple scales, which is crucial for image restoration tasks.
"""
import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """Convolutional block with Conv -> InstanceNorm -> LeakyReLU."""
    
    def __init__(self, in_channels, out_channels, kernel_size=4, stride=2, 
                 padding=1, use_norm=True, use_activation=True):
        super().__init__()
        
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=not use_norm)
        ]
        
        if use_norm:
            layers.append(nn.InstanceNorm2d(out_channels))
        
        if use_activation:
            layers.append(nn.LeakyReLU(0.2, inplace=True))
        
        self.block = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.block(x)


class DeconvBlock(nn.Module):
    """Transposed convolution block with ConvTranspose -> InstanceNorm -> ReLU + Dropout."""
    
    def __init__(self, in_channels, out_channels, kernel_size=4, stride=2,
                 padding=1, use_dropout=False):
        super().__init__()
        
        layers = [
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ]
        
        if use_dropout:
            layers.append(nn.Dropout(0.5))
        
        self.block = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.block(x)


class ResidualBlock(nn.Module):
    """Residual block for feature refinement."""
    
    def __init__(self, channels):
        super().__init__()
        
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, 3, 1, 0, bias=False),
            nn.InstanceNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, 3, 1, 0, bias=False),
            nn.InstanceNorm2d(channels)
        )
    
    def forward(self, x):
        return x + self.block(x)


class UNetGenerator(nn.Module):
    """
    U-Net Generator with skip connections.
    
    Architecture:
        Encoder (downsampling path) -> Bottleneck with residual blocks -> Decoder (upsampling path)
        Skip connections concatenate encoder features to decoder features.
    
    Args:
        in_channels: Number of input channels (3 for RGB)
        out_channels: Number of output channels (3 for RGB)
        ngf: Number of generator filters in first conv layer
        n_residual: Number of residual blocks in bottleneck
    """
    
    def __init__(self, in_channels=3, out_channels=3, ngf=64, n_residual=9):
        super().__init__()
        
        # ============ Initial Convolution ============
        self.initial = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels, ngf, kernel_size=7, stride=1, padding=0, bias=False),
            nn.InstanceNorm2d(ngf),
            nn.ReLU(inplace=True)
        )
        
        # ============ Encoder (Downsampling) ============
        # Each encoder block halves the spatial dimensions and doubles the channels
        self.down1 = ConvBlock(ngf, ngf * 2)        # 256 -> 128, 64 -> 128
        self.down2 = ConvBlock(ngf * 2, ngf * 4)    # 128 -> 64, 128 -> 256
        self.down3 = ConvBlock(ngf * 4, ngf * 8)    # 64 -> 32, 256 -> 512
        self.down4 = ConvBlock(ngf * 8, ngf * 8)    # 32 -> 16, 512 -> 512
        
        # ============ Bottleneck (Residual Blocks) ============
        residual_blocks = []
        for _ in range(n_residual):
            residual_blocks.append(ResidualBlock(ngf * 8))
        self.bottleneck = nn.Sequential(*residual_blocks)
        
        # ============ Decoder (Upsampling with Skip Connections) ============
        # Input channels = decoder features + skip connection features
        self.up1 = DeconvBlock(ngf * 8, ngf * 8, use_dropout=True)      # 16 -> 32
        self.up2 = DeconvBlock(ngf * 8 * 2, ngf * 4, use_dropout=True)  # 32 -> 64 (with skip)
        self.up3 = DeconvBlock(ngf * 4 * 2, ngf * 2)                    # 64 -> 128 (with skip)
        self.up4 = DeconvBlock(ngf * 2 * 2, ngf)                        # 128 -> 256 (with skip)
        
        # ============ Final Output ============
        self.final = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(ngf * 2, out_channels, kernel_size=7, stride=1, padding=0),
            nn.Tanh()  # Output in range [-1, 1]
        )
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        """Initialize network weights."""
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.normal_(m.weight, 0.0, 0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.InstanceNorm2d):
            if m.weight is not None:
                nn.init.normal_(m.weight, 1.0, 0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """
        Forward pass with skip connections.
        
        Args:
            x: Input blurred image [B, C, H, W]
            
        Returns:
            Deblurred image [B, C, H, W]
        """
        # Initial convolution
        x0 = self.initial(x)  # [B, ngf, H, W]
        
        # Encoder
        x1 = self.down1(x0)   # [B, ngf*2, H/2, W/2]
        x2 = self.down2(x1)   # [B, ngf*4, H/4, W/4]
        x3 = self.down3(x2)   # [B, ngf*8, H/8, W/8]
        x4 = self.down4(x3)   # [B, ngf*8, H/16, W/16]
        
        # Bottleneck
        x4 = self.bottleneck(x4)
        
        # Decoder with skip connections
        x = self.up1(x4)                          # [B, ngf*8, H/8, W/8]
        x = self.up2(torch.cat([x, x3], dim=1))   # [B, ngf*4, H/4, W/4]
        x = self.up3(torch.cat([x, x2], dim=1))   # [B, ngf*2, H/2, W/2]
        x = self.up4(torch.cat([x, x1], dim=1))   # [B, ngf, H, W]
        
        # Final output with skip from initial
        x = self.final(torch.cat([x, x0], dim=1))
        
        return x


class UNetGeneratorLite(nn.Module):
    """
    Lighter U-Net variant for faster training/inference.
    Uses fewer residual blocks and shallower encoder.
    """
    
    def __init__(self, in_channels=3, out_channels=3, ngf=64, n_residual=6):
        super().__init__()
        
        # Initial conv
        self.initial = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels, ngf, 7, 1, 0, bias=False),
            nn.InstanceNorm2d(ngf),
            nn.ReLU(inplace=True)
        )
        
        # Encoder
        self.down1 = ConvBlock(ngf, ngf * 2)
        self.down2 = ConvBlock(ngf * 2, ngf * 4)
        self.down3 = ConvBlock(ngf * 4, ngf * 8)
        
        # Bottleneck
        self.bottleneck = nn.Sequential(
            *[ResidualBlock(ngf * 8) for _ in range(n_residual)]
        )
        
        # Decoder
        self.up1 = DeconvBlock(ngf * 8, ngf * 4, use_dropout=True)
        self.up2 = DeconvBlock(ngf * 4 * 2, ngf * 2)
        self.up3 = DeconvBlock(ngf * 2 * 2, ngf)
        
        # Final
        self.final = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(ngf * 2, out_channels, 7, 1, 0),
            nn.Tanh()
        )
        
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.normal_(m.weight, 0.0, 0.02)
    
    def forward(self, x):
        x0 = self.initial(x)
        x1 = self.down1(x0)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        
        x3 = self.bottleneck(x3)
        
        x = self.up1(x3)
        x = self.up2(torch.cat([x, x2], dim=1))
        x = self.up3(torch.cat([x, x1], dim=1))
        x = self.final(torch.cat([x, x0], dim=1))
        
        return x


if __name__ == "__main__":
    # Test the generator
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create model
    model = UNetGenerator(in_channels=3, out_channels=3, ngf=64).to(device)
    
    # Test forward pass
    x = torch.randn(1, 3, 256, 256).to(device)
    y = model(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")
    print(f"Output range: [{y.min().item():.3f}, {y.max().item():.3f}]")
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of trainable parameters: {num_params:,}")

