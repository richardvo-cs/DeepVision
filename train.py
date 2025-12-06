"""
Training script for DeblurGAN.

Trains the conditional GAN for motion deblurring using:
- U-Net Generator
- PatchGAN Discriminator  
- Composite loss (Adversarial + Perceptual + Content)
"""
import os
import argparse
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid, save_image
from tqdm import tqdm

from config import Config
from models import UNetGenerator, PatchGANDiscriminator, PerceptualLoss, GANLoss
from models.losses import ContentLoss
from data import get_dataloaders


class Trainer:
    """DeblurGAN Trainer."""
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device(config.DEVICE if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Create directories
        config.create_dirs()
        
        # Initialize models
        self._init_models()
        
        # Initialize losses
        self._init_losses()
        
        # Initialize optimizers
        self._init_optimizers()
        
        # Initialize dataloaders
        self._init_dataloaders()
        
        # Initialize logging
        self._init_logging()
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        
        # Resume from checkpoint if specified
        if config.RESUME and config.CHECKPOINT_PATH:
            self._load_checkpoint(config.CHECKPOINT_PATH)
    
    def _init_models(self):
        """Initialize generator and discriminator."""
        self.generator = UNetGenerator(
            in_channels=self.config.IMG_CHANNELS,
            out_channels=self.config.IMG_CHANNELS,
            ngf=self.config.NGF
        ).to(self.device)
        
        self.discriminator = PatchGANDiscriminator(
            in_channels=self.config.IMG_CHANNELS * 2,  # Conditional: blur + sharp/fake
            ndf=self.config.NDF,
            n_layers=self.config.N_LAYERS_D
        ).to(self.device)
        
        # Count parameters
        g_params = sum(p.numel() for p in self.generator.parameters() if p.requires_grad)
        d_params = sum(p.numel() for p in self.discriminator.parameters() if p.requires_grad)
        print(f"Generator parameters: {g_params:,}")
        print(f"Discriminator parameters: {d_params:,}")
    
    def _init_losses(self):
        """Initialize loss functions."""
        self.gan_loss = GANLoss(gan_mode='lsgan').to(self.device)
        self.perceptual_loss = PerceptualLoss().to(self.device)
        self.content_loss = ContentLoss().to(self.device)
    
    def _init_optimizers(self):
        """Initialize optimizers."""
        self.optimizer_G = torch.optim.Adam(
            self.generator.parameters(),
            lr=self.config.LEARNING_RATE_G,
            betas=(self.config.BETA1, self.config.BETA2)
        )
        
        self.optimizer_D = torch.optim.Adam(
            self.discriminator.parameters(),
            lr=self.config.LEARNING_RATE_D,
            betas=(self.config.BETA1, self.config.BETA2)
        )
        
        # Learning rate schedulers
        self.scheduler_G = torch.optim.lr_scheduler.StepLR(
            self.optimizer_G, step_size=100, gamma=0.5
        )
        self.scheduler_D = torch.optim.lr_scheduler.StepLR(
            self.optimizer_D, step_size=100, gamma=0.5
        )
    
    def _init_dataloaders(self):
        """Initialize data loaders."""
        self.train_loader, self.test_loader = get_dataloaders(
            train_dir=self.config.TRAIN_DIR,
            test_dir=self.config.TEST_DIR,
            batch_size=self.config.BATCH_SIZE,
            img_size=(self.config.IMG_HEIGHT, self.config.IMG_WIDTH),
            num_workers=self.config.NUM_WORKERS,
            use_random_crop=True
        )
    
    def _init_logging(self):
        """Initialize TensorBoard logging."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_dir = os.path.join(self.config.LOG_DIR, f"run_{timestamp}")
        self.writer = SummaryWriter(log_dir)
        print(f"TensorBoard logs: {log_dir}")
    
    def _train_discriminator(self, blur, sharp, fake):
        """
        Train discriminator for one step.
        
        Args:
            blur: Blurred input images
            sharp: Ground truth sharp images
            fake: Generated deblurred images
            
        Returns:
            Discriminator loss value
        """
        self.optimizer_D.zero_grad()
        
        # Real loss
        pred_real = self.discriminator(blur, sharp)
        loss_real = self.gan_loss(pred_real, target_is_real=True)
        
        # Fake loss (detach to not backprop through generator)
        pred_fake = self.discriminator(blur, fake.detach())
        loss_fake = self.gan_loss(pred_fake, target_is_real=False)
        
        # Combined loss
        loss_D = (loss_real + loss_fake) * 0.5
        
        loss_D.backward()
        self.optimizer_D.step()
        
        return loss_D.item()
    
    def _train_generator(self, blur, sharp, fake):
        """
        Train generator for one step.
        
        Args:
            blur: Blurred input images
            sharp: Ground truth sharp images
            fake: Generated deblurred images
            
        Returns:
            Dictionary of loss values
        """
        self.optimizer_G.zero_grad()
        
        # Adversarial loss
        pred_fake = self.discriminator(blur, fake)
        loss_adv = self.gan_loss(pred_fake, target_is_real=True)
        
        # Perceptual loss
        loss_perc = self.perceptual_loss(fake, sharp)
        
        # Content loss
        loss_content = self.content_loss(fake, sharp)
        
        # Total generator loss
        loss_G = (
            self.config.LAMBDA_ADVERSARIAL * loss_adv +
            self.config.LAMBDA_PERCEPTUAL * loss_perc +
            self.config.LAMBDA_CONTENT * loss_content
        )
        
        loss_G.backward()
        self.optimizer_G.step()
        
        return {
            'total': loss_G.item(),
            'adversarial': loss_adv.item(),
            'perceptual': loss_perc.item(),
            'content': loss_content.item()
        }
    
    def _save_checkpoint(self, epoch, is_best=False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'global_step': self.global_step,
            'generator_state_dict': self.generator.state_dict(),
            'discriminator_state_dict': self.discriminator.state_dict(),
            'optimizer_G_state_dict': self.optimizer_G.state_dict(),
            'optimizer_D_state_dict': self.optimizer_D.state_dict(),
            'scheduler_G_state_dict': self.scheduler_G.state_dict(),
            'scheduler_D_state_dict': self.scheduler_D.state_dict(),
        }
        
        # Save regular checkpoint
        path = os.path.join(self.config.CHECKPOINT_DIR, f"checkpoint_epoch_{epoch}.pth")
        torch.save(checkpoint, path)
        
        # Save latest checkpoint
        latest_path = os.path.join(self.config.CHECKPOINT_DIR, "checkpoint_latest.pth")
        torch.save(checkpoint, latest_path)
        
        if is_best:
            best_path = os.path.join(self.config.CHECKPOINT_DIR, "checkpoint_best.pth")
            torch.save(checkpoint, best_path)
        
        print(f"Saved checkpoint: {path}")
    
    def _load_checkpoint(self, path):
        """Load model checkpoint."""
        print(f"Loading checkpoint: {path}")
        checkpoint = torch.load(path, map_location=self.device)
        
        self.generator.load_state_dict(checkpoint['generator_state_dict'])
        self.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
        self.optimizer_G.load_state_dict(checkpoint['optimizer_G_state_dict'])
        self.optimizer_D.load_state_dict(checkpoint['optimizer_D_state_dict'])
        self.scheduler_G.load_state_dict(checkpoint['scheduler_G_state_dict'])
        self.scheduler_D.load_state_dict(checkpoint['scheduler_D_state_dict'])
        
        self.current_epoch = checkpoint['epoch'] + 1
        self.global_step = checkpoint['global_step']
        
        print(f"Resumed from epoch {self.current_epoch}")
    
    def _save_samples(self, blur, sharp, fake, step):
        """Save sample images."""
        # Denormalize from [-1, 1] to [0, 1]
        blur = (blur + 1) / 2
        sharp = (sharp + 1) / 2
        fake = (fake + 1) / 2
        
        # Create grid: blur | fake | sharp
        comparison = torch.cat([blur, fake, sharp], dim=3)  # Concatenate along width
        grid = make_grid(comparison, nrow=2, normalize=False)
        
        # Save to file
        save_path = os.path.join(self.config.SAMPLE_DIR, f"sample_step_{step}.png")
        save_image(grid, save_path)
        
        # Log to TensorBoard
        self.writer.add_image('samples/comparison', grid, step)
    
    def train_epoch(self, epoch):
        """Train for one epoch."""
        self.generator.train()
        self.discriminator.train()
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}")
        
        epoch_losses = {
            'D': 0.0,
            'G_total': 0.0,
            'G_adv': 0.0,
            'G_perc': 0.0,
            'G_content': 0.0
        }
        
        for batch_idx, batch in enumerate(pbar):
            blur = batch['blur'].to(self.device)
            sharp = batch['sharp'].to(self.device)
            
            # Generate fake images
            fake = self.generator(blur)
            
            # Train discriminator
            loss_D = self._train_discriminator(blur, sharp, fake)
            
            # Train generator
            fake = self.generator(blur)  # Re-generate for fresh computation graph
            losses_G = self._train_generator(blur, sharp, fake)
            
            # Update running losses
            epoch_losses['D'] += loss_D
            epoch_losses['G_total'] += losses_G['total']
            epoch_losses['G_adv'] += losses_G['adversarial']
            epoch_losses['G_perc'] += losses_G['perceptual']
            epoch_losses['G_content'] += losses_G['content']
            
            # Update progress bar
            pbar.set_postfix({
                'D': f"{loss_D:.4f}",
                'G': f"{losses_G['total']:.4f}"
            })
            
            # Logging
            if self.global_step % self.config.LOG_FREQ == 0:
                self.writer.add_scalar('loss/discriminator', loss_D, self.global_step)
                self.writer.add_scalar('loss/generator_total', losses_G['total'], self.global_step)
                self.writer.add_scalar('loss/generator_adversarial', losses_G['adversarial'], self.global_step)
                self.writer.add_scalar('loss/generator_perceptual', losses_G['perceptual'], self.global_step)
                self.writer.add_scalar('loss/generator_content', losses_G['content'], self.global_step)
            
            # Save samples
            if self.global_step % self.config.SAMPLE_FREQ == 0:
                self._save_samples(blur[:4], sharp[:4], fake[:4], self.global_step)
            
            self.global_step += 1
        
        # Average losses
        n_batches = len(self.train_loader)
        for key in epoch_losses:
            epoch_losses[key] /= n_batches
        
        return epoch_losses
    
    @torch.no_grad()
    def validate(self):
        """Run validation."""
        self.generator.eval()
        
        total_psnr = 0.0
        n_samples = 0
        
        for batch in tqdm(self.test_loader, desc="Validating"):
            blur = batch['blur'].to(self.device)
            sharp = batch['sharp'].to(self.device)
            
            fake = self.generator(blur)
            
            # Calculate PSNR
            mse = torch.mean((fake - sharp) ** 2, dim=[1, 2, 3])
            psnr = 10 * torch.log10(4.0 / mse)  # max value is 2 (for [-1,1] range), so max^2 = 4
            
            total_psnr += psnr.sum().item()
            n_samples += blur.size(0)
        
        avg_psnr = total_psnr / n_samples
        return avg_psnr
    
    def train(self):
        """Main training loop."""
        print(f"\n{'='*60}")
        print("Starting DeblurGAN Training")
        print(f"{'='*60}")
        print(f"Epochs: {self.config.NUM_EPOCHS}")
        print(f"Batch size: {self.config.BATCH_SIZE}")
        print(f"Learning rate (G): {self.config.LEARNING_RATE_G}")
        print(f"Learning rate (D): {self.config.LEARNING_RATE_D}")
        print(f"{'='*60}\n")
        
        best_psnr = 0.0
        
        for epoch in range(self.current_epoch, self.config.NUM_EPOCHS):
            # Train
            losses = self.train_epoch(epoch)
            
            # Update learning rates
            self.scheduler_G.step()
            self.scheduler_D.step()
            
            # Log epoch metrics
            self.writer.add_scalar('epoch/loss_D', losses['D'], epoch)
            self.writer.add_scalar('epoch/loss_G', losses['G_total'], epoch)
            self.writer.add_scalar('epoch/lr_G', self.scheduler_G.get_last_lr()[0], epoch)
            
            # Print epoch summary
            print(f"\nEpoch {epoch} Summary:")
            print(f"  D Loss: {losses['D']:.4f}")
            print(f"  G Loss: {losses['G_total']:.4f} (Adv: {losses['G_adv']:.4f}, "
                  f"Perc: {losses['G_perc']:.4f}, Content: {losses['G_content']:.4f})")
            
            # Validate
            if epoch % 5 == 0:
                psnr = self.validate()
                self.writer.add_scalar('epoch/psnr', psnr, epoch)
                print(f"  Validation PSNR: {psnr:.2f} dB")
                
                is_best = psnr > best_psnr
                if is_best:
                    best_psnr = psnr
                    print(f"  New best PSNR!")
            else:
                is_best = False
            
            # Save checkpoint
            if (epoch + 1) % self.config.SAVE_FREQ == 0:
                self._save_checkpoint(epoch, is_best)
        
        # Save final model
        self._save_checkpoint(self.config.NUM_EPOCHS - 1)
        print(f"\nTraining complete! Best PSNR: {best_psnr:.2f} dB")
        
        self.writer.close()


def main():
    parser = argparse.ArgumentParser(description='Train DeblurGAN')
    parser.add_argument('--data_root', type=str, default='.',
                        help='Path to dataset root (containing train/ and test/ folders)')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size')
    parser.add_argument('--epochs', type=int, default=300,
                        help='Number of epochs')
    parser.add_argument('--lr_g', type=float, default=1e-4,
                        help='Generator learning rate')
    parser.add_argument('--lr_d', type=float, default=1e-4,
                        help='Discriminator learning rate')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'cpu'], help='Device to use')
    
    args = parser.parse_args()
    
    # Update config with command line arguments
    Config.DATA_ROOT = args.data_root
    Config.TRAIN_DIR = os.path.join(args.data_root, "train")
    Config.TEST_DIR = os.path.join(args.data_root, "test")
    Config.BATCH_SIZE = args.batch_size
    Config.NUM_EPOCHS = args.epochs
    Config.LEARNING_RATE_G = args.lr_g
    Config.LEARNING_RATE_D = args.lr_d
    Config.DEVICE = args.device
    
    if args.resume:
        Config.RESUME = True
        Config.CHECKPOINT_PATH = args.resume
    
    # Create trainer and start training
    trainer = Trainer(Config)
    trainer.train()


if __name__ == "__main__":
    main()

