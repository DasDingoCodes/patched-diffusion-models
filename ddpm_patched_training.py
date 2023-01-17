import os
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from tqdm import tqdm
from torch import optim
from ddpm_utils import *
from ddpm_patched import UNetPatched
from ddpm_modules import UNet
import logging
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

logging.basicConfig(format="%(asctime)s - %(levelname)s: %(message)s", level=logging.INFO, datefmt="%I:%M:%S")


class Diffusion:
    def __init__(self, noise_steps=1000, beta_start=1e-4, beta_end=0.02, img_size=256, device="cuda", prediction_type="noise"):
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.img_size = img_size
        self.device = device
        self.prediction_type = prediction_type

        self.beta = self.prepare_noise_schedule().to(device)
        self.alpha = 1. - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

    def prepare_noise_schedule(self):
        return torch.linspace(self.beta_start, self.beta_end, self.noise_steps)

    def noise_images(self, x, t):
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[:, None, None, None]
        Ɛ = torch.randn_like(x)
        prev_x = sqrt_alpha_hat * x
        return prev_x + sqrt_one_minus_alpha_hat * Ɛ, Ɛ, prev_x

    def sample_timesteps(self, n):
        return torch.randint(low=1, high=self.noise_steps, size=(n,))

    def sample(self, model, n, x=None):
        logging.info(f"Sampling {n} new images....")
        model.eval()
        with torch.no_grad():
            if x == None:
                x = torch.randn((n, 3, self.img_size, self.img_size)).to(self.device)
            for i in tqdm(reversed(range(1, self.noise_steps)), position=0):
                t = (torch.ones(n) * i).long().to(self.device)
                alpha = self.alpha[t][:, None, None, None]
                alpha_hat = self.alpha_hat[t][:, None, None, None]
                beta = self.beta[t][:, None, None, None]
                if i > 1:
                    noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x)
                if self.prediction_type == "noise":
                    predicted_noise = model(x, t)
                    x = 1 / torch.sqrt(alpha) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + torch.sqrt(beta) * noise
                else:
                    x = model(x, t) + torch.sqrt(beta) * noise
        model.train()
        x = (x.clamp(-1, 1) + 1) / 2
        x = (x * 255).type(torch.uint8)
        return x


def train(args):
    setup_logging(args.run_name)
    if torch.cuda.device_count() > 1:
        device = args.device
    else:
        device = "cuda"
    dataloader = get_data(args)
    prediction_type = args.prediction_type or "noise"
    # prediction_type determines what the model returns
    # "noise" means the model returns the noise of an input image
    # "image" means the model returns the image without the noise
    # "image" should work better with Patched Diffusion Models according to https://arxiv.org/pdf/2207.04316.pdf
    assert prediction_type in ["noise", "image"]
    model = UNetPatched(
        img_shape=(3, args.image_size, args.image_size),
        hidden=args.hidden,
        num_patches=args.num_patches,
        level_mult = args.level_mult,
        use_self_attention=False,
        dropout=args.dropout
    )
    if torch.cuda.device_count() > 1:
        model= nn.DataParallel(model,device_ids = args.device_ids)
    model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    mse = nn.MSELoss()
    diffusion = Diffusion(img_size=args.image_size, device=device, prediction_type=prediction_type)
    logger = SummaryWriter(os.path.join("runs", args.run_name))
    l = args.steps_per_epoch # len(dataloader)

    # 8x8 grid of sample images with fixed random values
    # when saving images, 8 columns are default for grid
    num_sample_imgs = 8*8
    noise_sample = torch.randn((num_sample_imgs, 3, args.image_size, args.image_size)).to(device)

    for epoch in range(args.epochs):
        logging.info(f"Starting epoch {epoch}:")
        pbar = tqdm(range(args.steps_per_epoch))
        for i in pbar:
            images, _ = next(iter(dataloader))

            images = images.to(device)
            t = diffusion.sample_timesteps(images.shape[0]).to(device)
            x_t, noise, prev_x = diffusion.noise_images(images, t)
            if prediction_type == "noise":
                predicted_noise = model(x_t, t)
                loss = mse(noise, predicted_noise)
            else:
                predicted_x = model(x_t, t)
                loss = mse(prev_x, predicted_x)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pbar.set_postfix(MSE=loss.item())
            logger.add_scalar("MSE", loss.item(), global_step=epoch * l + i)

        sampled_images = diffusion.sample(model, n=num_sample_imgs, x=noise_sample)
        save_images(sampled_images, os.path.join("results", args.run_name, f"{epoch}.jpg"))
        torch.save(model.state_dict(), os.path.join("models", args.run_name, f"ckpt.pt"))


def launch():
    import argparse
    parser = argparse.ArgumentParser()
    dataset = "animefaces"
    args = parser.parse_args()
    args.epochs = 1000
    args.steps_per_epoch = 1000
    args.batch_size = 32
    args.image_size = 64
    args.num_patches = 2
    args.level_mult = [1,2,4,8]
    args.dataset_path = f"data/{dataset}"
    args.device = "cuda:2"
    args.device_ids = [2,3]
    args.lr = 3e-4
    args.hidden = 128
    args.prediction_type = "image"
    args.dropout = 0.0
    time_str = datetime.now().strftime('%Y-%m-%dT%H-%M-%S')
    args.run_name = f"{time_str}_DDPM_{args.num_patches}x{args.num_patches}_patches_{dataset}_{args.epochs}_epochs"
    train(args)


if __name__ == '__main__':
    launch()