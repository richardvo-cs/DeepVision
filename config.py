"""
Configuration settings for DeblurGAN training and inference.
"""
import os

class Config:
    # ============ Dataset ============
    # Note: GoPro dataset extracted directly to project root (train/ and test/ folders)
    DATA_ROOT = "."
    TRAIN_DIR = os.path.join(DATA_ROOT, "train")
    TEST_DIR = os.path.join(DATA_ROOT, "test")
    
    # ============ Image Settings ============
    IMG_HEIGHT = 256
    IMG_WIDTH = 256
    IMG_CHANNELS = 3
    
    # ============ Training ============
    BATCH_SIZE = 4
    NUM_EPOCHS = 300
    NUM_WORKERS = 4
    
    # ============ Optimizer ============
    LEARNING_RATE_G = 1e-4
    LEARNING_RATE_D = 1e-4
    BETA1 = 0.5
    BETA2 = 0.999
    
    # ============ Loss Weights ============
    LAMBDA_PERCEPTUAL = 100.0  # Weight for perceptual loss
    LAMBDA_ADVERSARIAL = 1.0   # Weight for adversarial loss
    LAMBDA_CONTENT = 10.0      # Weight for L1 content loss
    
    # ============ Model Architecture ============
    # Generator (U-Net)
    NGF = 64  # Number of generator filters in first conv layer
    
    # Discriminator (PatchGAN)
    NDF = 64  # Number of discriminator filters in first conv layer
    N_LAYERS_D = 3  # Number of layers in PatchGAN
    
    # ============ Checkpoints & Logging ============
    # Saving to D: drive to avoid C: disk space issues
    CHECKPOINT_DIR = "D:/DeepVision_checkpoints"
    LOG_DIR = "D:/DeepVision_logs"
    SAMPLE_DIR = "D:/DeepVision_samples"
    
    SAVE_FREQ = 5  # Save checkpoint every N epochs
    LOG_FREQ = 100  # Log every N iterations
    SAMPLE_FREQ = 500  # Generate samples every N iterations
    
    # ============ Resume Training ============
    RESUME = False
    CHECKPOINT_PATH = None  # Path to checkpoint to resume from
    
    # ============ Device ============
    DEVICE = "cuda"  # "cuda" or "cpu"
    
    @classmethod
    def create_dirs(cls):
        """Create necessary directories."""
        os.makedirs(cls.CHECKPOINT_DIR, exist_ok=True)
        os.makedirs(cls.LOG_DIR, exist_ok=True)
        os.makedirs(cls.SAMPLE_DIR, exist_ok=True)

