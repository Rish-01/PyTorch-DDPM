"""Train a DDPM with the UNet in unet.py.

Images are scaled to [-1, 1]. The default dataset is CIFAR-10 at 32x32.
Pass --data-dir to train on an ImageFolder instead.

Example:
    conda activate ddpm
    python train.py --epochs 100 --batch-size 64 --sample-every 5
"""

import argparse
import os
from copy import deepcopy

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, utils
from tqdm import tqdm

from diffusion import GaussianDiffusion
from unet import UNet


class EMA:
    """Exponential moving average of floating-point parameters, as used by DDPM."""

    def __init__(self, model, decay=0.9999):
        self.decay = decay
        self.shadow = {
            name: param.detach().clone()
            for name, param in model.state_dict().items()
        }

    @torch.no_grad()
    def update(self, model):
        for name, param in model.state_dict().items():
            shadow = self.shadow[name]
            if param.dtype.is_floating_point:
                shadow.mul_(self.decay).add_(param.detach(), alpha=1.0 - self.decay)
            else:
                shadow.copy_(param)

    def state_dict(self):
        return self.shadow

    def load_state_dict(self, state):
        self.shadow = state

    @torch.no_grad()
    def copy_to(self, model):
        model.load_state_dict(self.shadow, strict=False)


def build_loader(args):
    transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.CenterCrop(args.image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    if args.data_dir:
        dataset = datasets.ImageFolder(args.data_dir, transform=transform)
    elif args.dataset == "cifar10":
        dataset = datasets.CIFAR10(args.data_root, train=True, download=True, transform=transform)
    elif args.dataset == "mnist":
        mnist_transform = transforms.Compose([
            transforms.Resize((args.image_size, args.image_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ])
        dataset = datasets.MNIST(args.data_root, train=True, download=True, transform=mnist_transform)
    else:
        raise ValueError(f"Unknown dataset {args.dataset}")
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )


def save_checkpoint(path, model, ema, optimizer, step, epoch):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "ema": ema.state_dict(),
            "optimizer": optimizer.state_dict(),
            "step": step,
            "epoch": epoch,
        },
        path,
    )


@torch.no_grad()
def save_samples(path, diffusion, model, ema, shape, device):
    sample_model = deepcopy(model).eval()
    ema.copy_to(sample_model)
    samples = diffusion.sample(sample_model, shape, device=device)
    samples = (samples.clamp(-1, 1) + 1) * 0.5
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    utils.save_image(samples, path, nrow=int(shape[0] ** 0.5))


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    levels = len(args.channel_mult)
    if args.image_size % (2 ** levels) != 0:
        raise ValueError(
            f"image size {args.image_size} must be divisible by {2 ** levels} "
            f"for {levels} downsample stages"
        )
    if args.channel_mult[0] != 1:
        raise ValueError("the first channel multiplier must be 1")
    loader = build_loader(args)
    in_channels = 1 if args.dataset == "mnist" and not args.data_dir else 3

    model = UNet(
        timesteps=args.timesteps,
        embedding_dim=args.embedding_dim,
        in_channels=in_channels,
        out_channel=in_channels,
        ch_mult=tuple(args.channel_mult),
        dropout=args.dropout,
    ).to(device)
    diffusion = GaussianDiffusion(
        timesteps=args.timesteps,
        beta_start=args.beta_start,
        beta_end=args.beta_end,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    ema = EMA(model, decay=args.ema_decay)

    start_epoch = 0
    step = 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["model"])
        ema.load_state_dict(checkpoint["ema"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        step = checkpoint.get("step", 0)
        start_epoch = checkpoint.get("epoch", 0) + 1

    os.makedirs(args.out_dir, exist_ok=True)
    model.train()
    for epoch in range(start_epoch, args.epochs):
        running = 0.0
        seen = 0
        progress = tqdm(loader, desc=f"epoch {epoch + 1}/{args.epochs}")
        for images, _ in progress:
            if args.max_steps and step >= args.max_steps:
                break
            images = images.to(device)
            loss = diffusion.p_losses(model, images)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            ema.update(model)

            running += loss.item() * images.shape[0]
            seen += images.shape[0]
            step += 1
            progress.set_postfix(loss=f"{loss.item():.4f}", step=step)

        if seen:
            print(f"epoch {epoch + 1} mean loss {running / seen:.4f}")
        ckpt_path = os.path.join(args.out_dir, "checkpoint.pt")
        save_checkpoint(ckpt_path, model, ema, optimizer, step, epoch)

        if args.sample_every and (epoch + 1) % args.sample_every == 0:
            sample_path = os.path.join(args.out_dir, f"samples_epoch_{epoch + 1:04d}.png")
            save_samples(
                sample_path,
                diffusion,
                model,
                ema,
                (args.n_samples, in_channels, args.image_size, args.image_size),
                device,
            )
            model.train()
            print(f"wrote {sample_path}")

        if args.max_steps and step >= args.max_steps:
            break

    print(f"finished at step {step}. checkpoint: {os.path.join(args.out_dir, 'checkpoint.pt')}")


def parse_args():
    parser = argparse.ArgumentParser(description="Train a DDPM on CIFAR-10, MNIST, or an image folder")
    parser.add_argument("--dataset", choices=("cifar10", "mnist"), default="cifar10")
    parser.add_argument("--data-dir", default="", help="ImageFolder root. Overrides --dataset when set")
    parser.add_argument("--data-root", default="data", help="Download directory for torchvision datasets")
    parser.add_argument("--out-dir", default="runs/ddpm")
    parser.add_argument("--image-size", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--max-steps", type=int, default=0, help="Stop after this many optimizer steps. 0 means run all epochs")
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--timesteps", type=int, default=1000)
    parser.add_argument("--beta-start", type=float, default=1e-4)
    parser.add_argument("--beta-end", type=float, default=0.02)
    parser.add_argument("--embedding-dim", type=int, default=128)
    parser.add_argument("--channel-mult", type=int, nargs="+", default=[1, 2, 2, 2])
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--ema-decay", type=float, default=0.9999)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--sample-every", type=int, default=5, help="Write an EMA sample grid every N epochs. 0 disables sampling")
    parser.add_argument("--n-samples", type=int, default=16)
    parser.add_argument("--resume", default="")
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
