"""
Loss functions for DeblurGAN training.

Composite loss = Adversarial Loss + Perceptual Loss + Content Loss

- Adversarial Loss: Encourages generator to produce photorealistic outputs
- Perceptual Loss: VGG-based feature matching for content fidelity
- Content Loss: L1 pixel-wise loss for structure preservation
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class GANLoss(nn.Module):
    """
    GAN Loss with support for different GAN variants.
    
    Supports:
    - 'vanilla': Standard cross-entropy loss (original GAN)
    - 'lsgan': Least squares GAN (more stable training)
    - 'wgan': Wasserstein GAN (no log, critic output)
    - 'wgan-gp': WGAN with gradient penalty
    """
    
    def __init__(self, gan_mode='lsgan', target_real_label=1.0, target_fake_label=0.0):
        super().__init__()
        
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        
        self.gan_mode = gan_mode
        
        if gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_mode in ['wgan', 'wgan-gp']:
            self.loss = None  # No traditional loss function
        else:
            raise NotImplementedError(f'GAN mode {gan_mode} not implemented')
    
    def get_target_tensor(self, prediction, target_is_real):
        """Create label tensors with the same size as predictions."""
        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)
    
    def forward(self, prediction, target_is_real):
        """
        Calculate GAN loss.
        
        Args:
            prediction: Discriminator output
            target_is_real: Whether target should be real or fake
            
        Returns:
            GAN loss value
        """
        if self.gan_mode in ['vanilla', 'lsgan']:
            target_tensor = self.get_target_tensor(prediction, target_is_real)
            loss = self.loss(prediction, target_tensor)
        elif self.gan_mode == 'wgan':
            if target_is_real:
                loss = -prediction.mean()
            else:
                loss = prediction.mean()
        elif self.gan_mode == 'wgan-gp':
            if target_is_real:
                loss = -prediction.mean()
            else:
                loss = prediction.mean()
        
        return loss


class VGGFeatureExtractor(nn.Module):
    """
    VGG19 feature extractor for perceptual loss.
    
    Extracts features from intermediate layers of VGG19 pretrained on ImageNet.
    Different layers capture different levels of abstraction:
    - Early layers: edges, textures
    - Middle layers: patterns, parts
    - Deep layers: objects, scenes
    """
    
    def __init__(self, feature_layers=[2, 7, 12, 21, 30], use_bn=False):
        """
        Args:
            feature_layers: Indices of VGG layers to extract features from
            use_bn: Whether to use VGG with batch normalization
        """
        super().__init__()
        
        if use_bn:
            vgg = models.vgg19_bn(weights=models.VGG19_BN_Weights.IMAGENET1K_V1)
        else:
            vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1)
        
        # Freeze VGG parameters
        for param in vgg.parameters():
            param.requires_grad = False
        
        self.features = vgg.features
        self.feature_layers = feature_layers
        
        # ImageNet normalization
        self.register_buffer(
            'mean', 
            torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            'std',
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )
    
    def forward(self, x):
        """
        Extract VGG features.
        
        Args:
            x: Input tensor in range [-1, 1]
            
        Returns:
            List of feature maps from specified layers
        """
        # Normalize from [-1, 1] to ImageNet range
        x = (x + 1) / 2  # [-1, 1] -> [0, 1]
        x = (x - self.mean) / self.std
        
        features = []
        for i, layer in enumerate(self.features):
            x = layer(x)
            if i in self.feature_layers:
                features.append(x)
        
        return features


class PerceptualLoss(nn.Module):
    """
    Perceptual Loss (Feature Matching Loss).
    
    Compares high-level features extracted from VGG network rather than
    raw pixel values. This encourages perceptually similar outputs.
    """
    
    def __init__(self, feature_layers=[2, 7, 12, 21, 30], weights=None):
        """
        Args:
            feature_layers: VGG layer indices for feature extraction
            weights: Optional weights for each feature layer
        """
        super().__init__()
        
        self.vgg = VGGFeatureExtractor(feature_layers)
        
        if weights is None:
            # Default: equal weight for all layers
            weights = [1.0] * len(feature_layers)
        self.weights = weights
    
    def forward(self, generated, target):
        """
        Calculate perceptual loss.
        
        Args:
            generated: Generated/deblurred image
            target: Ground truth sharp image
            
        Returns:
            Perceptual loss value
        """
        gen_features = self.vgg(generated)
        target_features = self.vgg(target)
        
        loss = 0.0
        for gf, tf, w in zip(gen_features, target_features, self.weights):
            loss += w * F.l1_loss(gf, tf)
        
        return loss


class ContentLoss(nn.Module):
    """
    Content/Pixel Loss.
    
    L1 loss between generated and target images for structure preservation.
    L1 is preferred over L2 as it produces less blurry results.
    """
    
    def __init__(self, loss_type='l1'):
        super().__init__()
        
        if loss_type == 'l1':
            self.loss = nn.L1Loss()
        elif loss_type == 'l2':
            self.loss = nn.MSELoss()
        elif loss_type == 'smooth_l1':
            self.loss = nn.SmoothL1Loss()
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")
    
    def forward(self, generated, target):
        return self.loss(generated, target)


class SSIMLoss(nn.Module):
    """
    Structural Similarity Index (SSIM) Loss.
    
    Measures structural similarity between images, considering luminance,
    contrast, and structure. SSIM loss = 1 - SSIM.
    """
    
    def __init__(self, window_size=11, sigma=1.5, channels=3):
        super().__init__()
        
        self.window_size = window_size
        self.channels = channels
        
        # Create Gaussian window
        gaussian = torch.exp(
            -torch.pow(torch.arange(window_size).float() - window_size // 2, 2) / 
            (2 * sigma ** 2)
        )
        gaussian = gaussian / gaussian.sum()
        
        window_1d = gaussian.unsqueeze(1)
        window_2d = window_1d.mm(window_1d.t()).unsqueeze(0).unsqueeze(0)
        window = window_2d.expand(channels, 1, window_size, window_size).contiguous()
        
        self.register_buffer('window', window)
        
        self.C1 = 0.01 ** 2
        self.C2 = 0.03 ** 2
    
    def forward(self, generated, target):
        """Calculate SSIM loss (1 - SSIM)."""
        # Normalize to [0, 1]
        generated = (generated + 1) / 2
        target = (target + 1) / 2
        
        # Calculate means
        mu1 = F.conv2d(generated, self.window, padding=self.window_size // 2, groups=self.channels)
        mu2 = F.conv2d(target, self.window, padding=self.window_size // 2, groups=self.channels)
        
        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu1_mu2 = mu1 * mu2
        
        # Calculate variances and covariance
        sigma1_sq = F.conv2d(generated ** 2, self.window, padding=self.window_size // 2, groups=self.channels) - mu1_sq
        sigma2_sq = F.conv2d(target ** 2, self.window, padding=self.window_size // 2, groups=self.channels) - mu2_sq
        sigma12 = F.conv2d(generated * target, self.window, padding=self.window_size // 2, groups=self.channels) - mu1_mu2
        
        # SSIM formula
        ssim = ((2 * mu1_mu2 + self.C1) * (2 * sigma12 + self.C2)) / \
               ((mu1_sq + mu2_sq + self.C1) * (sigma1_sq + sigma2_sq + self.C2))
        
        return 1 - ssim.mean()


class CompositeLoss(nn.Module):
    """
    Composite loss combining multiple loss functions.
    
    Total Loss = λ_adv * Adversarial + λ_perc * Perceptual + λ_content * Content
    """
    
    def __init__(self, lambda_adversarial=1.0, lambda_perceptual=100.0, 
                 lambda_content=10.0, gan_mode='lsgan'):
        super().__init__()
        
        self.lambda_adv = lambda_adversarial
        self.lambda_perc = lambda_perceptual
        self.lambda_content = lambda_content
        
        self.gan_loss = GANLoss(gan_mode=gan_mode)
        self.perceptual_loss = PerceptualLoss()
        self.content_loss = ContentLoss()
    
    def forward(self, generated, target, disc_fake=None):
        """
        Calculate composite generator loss.
        
        Args:
            generated: Generated deblurred image
            target: Ground truth sharp image
            disc_fake: Discriminator output for fake image (optional)
            
        Returns:
            Dictionary with total loss and individual components
        """
        losses = {}
        
        # Content loss
        losses['content'] = self.content_loss(generated, target)
        
        # Perceptual loss
        losses['perceptual'] = self.perceptual_loss(generated, target)
        
        # Adversarial loss (if discriminator output provided)
        if disc_fake is not None:
            losses['adversarial'] = self.gan_loss(disc_fake, target_is_real=True)
        else:
            losses['adversarial'] = torch.tensor(0.0, device=generated.device)
        
        # Total loss
        losses['total'] = (
            self.lambda_content * losses['content'] +
            self.lambda_perc * losses['perceptual'] +
            self.lambda_adv * losses['adversarial']
        )
        
        return losses


if __name__ == "__main__":
    # Test losses
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create dummy inputs
    generated = torch.randn(2, 3, 256, 256).to(device)
    target = torch.randn(2, 3, 256, 256).to(device)
    disc_output = torch.randn(2, 1, 30, 30).to(device)
    
    # Test individual losses
    print("Testing individual losses...")
    
    gan_loss = GANLoss('lsgan').to(device)
    print(f"GAN Loss (real): {gan_loss(disc_output, True).item():.4f}")
    print(f"GAN Loss (fake): {gan_loss(disc_output, False).item():.4f}")
    
    perceptual_loss = PerceptualLoss().to(device)
    print(f"Perceptual Loss: {perceptual_loss(generated, target).item():.4f}")
    
    content_loss = ContentLoss().to(device)
    print(f"Content Loss: {content_loss(generated, target).item():.4f}")
    
    # Test composite loss
    print("\nTesting composite loss...")
    composite_loss = CompositeLoss().to(device)
    losses = composite_loss(generated, target, disc_output)
    
    for name, value in losses.items():
        print(f"  {name}: {value.item():.4f}")

