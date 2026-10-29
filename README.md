# PyTorch-DDPM

Noise-prediction DDPM training for the UNet in `unet.py`. Images are scaled to `[-1, 1]`. The reverse process is the DDPM ancestral sampler with the true posterior variance, and sample grids are drawn from an EMA of the weights.

```bash
conda activate ddpm
python train.py --epochs 100 --batch-size 64 --sample-every 5
```

Checkpoints and sample grids are written to `runs/ddpm`. `--dataset mnist` trains a one-channel model. `--data-dir /path/to/imagefolder` trains on an `ImageFolder` instead of CIFAR-10.

The UNet conditions on a discrete timestep. Use the same `--timesteps` value for training and sampling. The image size must be divisible by `2 ** len(--channel-mult)` (16 for the default four stages).
