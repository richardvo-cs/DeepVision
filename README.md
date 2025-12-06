# DeepVision: Motion Deblurring with DeblurGAN

A PyTorch implementation of a conditional Generative Adversarial Network for single-image motion deblurring. This project implements a U-Net generator with PatchGAN discriminator, trained using a composite loss function combining adversarial, perceptual, and content losses.

## Objective

Develop a deep learning model that learns an effective mapping from the blurred domain to the sharp, visually plausible domain. This serves as a foundational proof-of-concept for systems that could enhance visual perception for individuals with eyesight impairments.

## Architecture

### Generator (U-Net)
- **Encoder**: 4 downsampling blocks capturing multi-scale features
- **Bottleneck**: 9 residual blocks for deep feature refinement
- **Decoder**: 4 upsampling blocks with skip connections
- **Output**: Tanh activation for [-1, 1] normalized images

### Discriminator (PatchGAN)
- 70×70 receptive field patches
- Enforces high-frequency local realism
- Conditional discrimination (blur + sharp/fake concatenation)

### Loss Function
```
L_total = λ_adv × L_adversarial + λ_perc × L_perceptual + λ_content × L_content
```
- **Adversarial Loss**: LSGAN for stable training
- **Perceptual Loss**: VGG19 feature matching
- **Content Loss**: L1 pixel-wise reconstruction

## Project Structure

```
DeepVision/
├── config.py              # Configuration settings
├── train.py               # Training script
├── test.py                # Testing and inference
├── requirements.txt       # Dependencies
├── models/
│   ├── __init__.py
│   ├── generator.py       # U-Net Generator
│   ├── discriminator.py   # PatchGAN Discriminator
│   └── losses.py          # Loss functions
├── data/
│   ├── __init__.py
│   └── dataset.py         # GoPro dataset loader
├── utils/
│   ├── __init__.py
│   ├── metrics.py         # PSNR, SSIM metrics
│   └── visualization.py   # Visualization utilities
├── checkpoints/           # Saved models
├── logs/                  # TensorBoard logs
├── samples/               # Generated samples
├── train/                 # GoPro training data
└── test/                  # GoPro test data
```

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Prepare Dataset

The GoPro Large dataset should be extracted to the project root:
```
DeepVision/
├── train/
│   ├── GOPR0372_07_00/
│   │   ├── blur/
│   │   └── sharp/
│   └── ...
└── test/
    └── ...
```

### 3. Train the Model

```bash
# Basic training
python train.py

# With custom parameters
python train.py --batch_size 8 --epochs 5 --lr_g 1e-4 --lr_d 1e-4

# Resume from checkpoint
python train.py --resume checkpoints/checkpoint_latest.pth
```

### 4. Monitor Training

```bash
tensorboard --logdir logs/
```

### 5. Test the Model

```bash
# Test on GoPro test set
python test.py --checkpoint checkpoints/checkpoint_best.pth --mode gopro --input ./test

# Deblur a single image
python test.py --checkpoint checkpoints/checkpoint_best.pth --mode single --input blur.png --output deblurred.png

# Batch inference on directory
python test.py --checkpoint checkpoints/checkpoint_best.pth --mode directory --input ./blurred_images --output ./results
```

## ⚙️ Configuration

Key hyperparameters in `config.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `BATCH_SIZE` | 4 | Training batch size |
| `NUM_EPOCHS` | 300 | Number of training epochs |
| `LEARNING_RATE_G` | 1e-4 | Generator learning rate |
| `LEARNING_RATE_D` | 1e-4 | Discriminator learning rate |
| `LAMBDA_PERCEPTUAL` | 100.0 | Perceptual loss weight |
| `LAMBDA_ADVERSARIAL` | 1.0 | Adversarial loss weight |
| `LAMBDA_CONTENT` | 10.0 | Content (L1) loss weight |
| `IMG_HEIGHT` | 256 | Training image height |
| `IMG_WIDTH` | 256 | Training image width |

## 📊 Expected Results

On the GoPro test dataset:
- **PSNR**: ~28-30 dB
- **SSIM**: ~0.85-0.92

## 🔧 Model Variants

The implementation includes several model variants:

1. **UNetGenerator**: Full U-Net with 9 residual blocks (default)
2. **UNetGeneratorLite**: Lighter variant with 6 residual blocks
3. **PatchGANDiscriminator**: Standard 70×70 PatchGAN
4. **MultiscaleDiscriminator**: Multi-scale PatchGAN
5. **SpectralNormDiscriminator**: PatchGAN with spectral normalization

## 📖 References

- [DeblurGAN: Blind Motion Deblurring Using Conditional Adversarial Networks](https://arxiv.org/abs/1711.07064) (Kupyn et al., 2018)
- [Image-to-Image Translation with Conditional Adversarial Networks](https://arxiv.org/abs/1611.07004) (pix2pix)
- [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597)
- [Perceptual Losses for Real-Time Style Transfer and Super-Resolution](https://arxiv.org/abs/1603.08155)


