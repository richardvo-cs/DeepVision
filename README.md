# DeepVision: Motion Deblurring

A PyTorch implementation of a conditional Generative Adversarial Network for single-image motion deblurring. This project implements a U-Net generator with PatchGAN discriminator, trained using a composite loss function combining adversarial, perceptual, and content losses.

DATA: [https://drive.google.com/file/d/1y4wvPdOG3mojpFCHTqLgriexhbjoWVkK/view?usp=drive_link](https://seungjunnah.github.io/Datasets/gopro.html)](https://seungjunnah.github.io/Datasets/gopro.html)

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

## Configuration

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

## 📖 References

- [DeblurGAN: Blind Motion Deblurring Using Conditional Adversarial Networks](https://arxiv.org/abs/1711.07064) (Kupyn et al., 2018)
- [Image-to-Image Translation with Conditional Adversarial Networks](https://arxiv.org/abs/1611.07004) (pix2pix)
- [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597)
- [Perceptual Losses for Real-Time Style Transfer and Super-Resolution](https://arxiv.org/abs/1603.08155)


