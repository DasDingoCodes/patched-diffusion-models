import os
import torch
import torch.nn as nn
from pathlib import Path
from matplotlib import pyplot as plt
from tqdm import tqdm
from torch import optim
from ddpm_utils import *
from ddpm_patched import UNetPatched
import logging
from datetime import datetime
import torchvision.transforms as T

logging.basicConfig(format="%(asctime)s - %(levelname)s: %(message)s", level=logging.INFO, datefmt="%I:%M:%S")


class Diffusion:
    def __init__(self, noise_steps=1000, beta_start=1e-4, beta_end=0.02, img_size=256, device="cuda", super_resolution_factor=4):
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.img_size = img_size
        self.device = device
        self.super_resolution_factor = super_resolution_factor

        self.beta = self.prepare_noise_schedule().to(device)
        self.alpha = 1. - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

    def prepare_noise_schedule(self):
        return torch.linspace(self.beta_start, self.beta_end, self.noise_steps)

    def noise_images(self, x, t, prediction_type):
        if prediction_type == "noise":
            sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None]
            sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[:, None, None, None]
            Ɛ = torch.randn_like(x)
            prev_x = sqrt_alpha_hat * x
            return prev_x + sqrt_one_minus_alpha_hat * Ɛ, Ɛ, prev_x
        else:
            sqrt_alpha = torch.sqrt(self.alpha[t])[:, None, None, None]
            prev_sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t-1])[:, None, None, None]
            prev_sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t-1])[:, None, None, None]
            prev_x = prev_sqrt_alpha_hat * x + prev_sqrt_one_minus_alpha_hat * torch.randn_like(x)
            Ɛ = torch.randn_like(x)
            this_beta = self.beta[t][:, None, None, None]
            this_x = sqrt_alpha * prev_x + this_beta * Ɛ
            return this_x, Ɛ, prev_x
    
    def super_resolution_noise_data(self, x, t):
        """Returns x_L (low resolution x), x_t (diffused difference between x and x_L at timestep t) and Ɛ (noise inserted into x_t at timestep t)"""
        x_L = self.low_res_x(x)
        x_t, Ɛ, _ = self.noise_images(x - x_L, t, prediction_type="noise")
        return x_L, x_t, Ɛ
    
    def colourise_noise_data(self, x, t):
        """Returns x_g (grayscaled x), x_t (diffused difference between x and x_g at timestep t) and Ɛ (noise inserted into x_t at timestep t)"""
        x_g = self.grayscale(x, num_output_channels=3)
        x_t, Ɛ, _ = self.noise_images(x - x_g, t, prediction_type="noise")
        return x_g, x_t, Ɛ

    def low_res_x(self, x):
        x_L = T.Resize(size=self.img_size//self.super_resolution_factor)(x)
        x_L = T.Resize(size=self.img_size)(x_L)
        return x_L

    def grayscale(self, x, num_output_channels=3):
        return T.Grayscale(num_output_channels=num_output_channels)(x)

    def sample_timesteps(self, n):
        return torch.randint(low=1, high=self.noise_steps, size=(n,))

    def sample(self, model, n, prediction_type, x=None, original_imgs=None):
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
                if prediction_type == "noise":
                    predicted_noise = model(x, t)
                    x = 1 / torch.sqrt(alpha) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + torch.sqrt(beta) * noise
                elif prediction_type == "image":
                    x = 1 / torch.sqrt(alpha) * model(x, t) + torch.sqrt(beta) * noise
                elif prediction_type == "super_resolution":
                    model_input = torch.concat((original_imgs, x), dim=1)
                    predicted_noise = model(model_input, t)
                    x = 1 / torch.sqrt(alpha) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + torch.sqrt(beta) * noise
                elif prediction_type == "colourise":
                    model_input = torch.concat((original_imgs, x), dim=1)
                    predicted_noise = model(model_input, t)
                    x = 1 / torch.sqrt(alpha) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + torch.sqrt(beta) * noise

        image_retouching_types = ["super_resolution", "colourise"]
        model.train()
        if prediction_type in image_retouching_types and original_imgs != None:
            # if image retouching, then model predicts difference between original_imgs and wanted images
            # in that case, add original_imgs to the output of the model to get a prediction for the wanted image
            x += original_imgs
        x = (x.clamp(-1, 1) + 1) / 2
        x = (x * 255).type(torch.uint8)
        return x


def train(args):
    setup_logging(args.run_name)
    if torch.cuda.device_count() > 1:
        device = args.device
    else:
        device = "cuda"
    prediction_type = args.prediction_type or "noise"
    # prediction_type determines what the model returns
    # "noise" means the model returns the noise of an input image
    # "image" means the model returns the image without the noise
    # "image" should work better with Patched Diffusion Models according to https://arxiv.org/pdf/2207.04316.pdf
    assert prediction_type in ["noise", "image", "super_resolution", "colourise"]
    image_retouching_types = ["super_resolution", "colourise"]
    # if prediction_type is of the image retouching type, then the model gets not only diffusion step x_t as input
    # but also the original image. Concat both x_t and the original image together at dim=1 and you get 3+3=6 input channels
    input_channels = 6 if prediction_type in image_retouching_types else 3
    model = UNetPatched(
        img_shape=(args.image_size, args.image_size),
        input_channels=input_channels,
        hidden=args.hidden,
        num_patches=args.num_patches,
        level_mult = args.level_mult,
        use_self_attention=args.use_self_attention,
        dropout=args.dropout
    )
    if torch.cuda.device_count() > 1:
        model= nn.DataParallel(model,device_ids = args.device_ids)
    model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    mse = nn.MSELoss()
    diffusion = Diffusion(
        img_size=args.image_size, 
        device=device, 
        super_resolution_factor=args.super_resolution_factor,
        noise_steps=args.noise_steps
    )
    l = args.steps_per_epoch
    
    # 8x8 grid of sample images with fixed random values
    # when saving images, 8 columns are default for grid
    num_sample_imgs = 8*8
    train_dataloader, sample_dataloader = get_data(args, sample_percentage=0.1)
    noise_sample = torch.randn((num_sample_imgs, 3, args.image_size, args.image_size)).to(device)
    if prediction_type == "super_resolution" or prediction_type == "colourise":
        sample_imgs_from_dataset, _ = next(iter(sample_dataloader))
        while sample_imgs_from_dataset.shape[0] < num_sample_imgs:
            imgs_next_batch, _ = next(iter(sample_dataloader))
            sample_imgs_from_dataset = torch.concat((sample_imgs_from_dataset, imgs_next_batch))
        sample_imgs_from_dataset = sample_imgs_from_dataset[:num_sample_imgs].to(device)
        if prediction_type == "super_resolution":
            sample_imgs_from_dataset = diffusion.low_res_x(sample_imgs_from_dataset)
        elif prediction_type == "colourise":
            sample_imgs_from_dataset = diffusion.grayscale(sample_imgs_from_dataset)
        img_data = NormalizeInverse((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(sample_imgs_from_dataset)
        img_data = img_data.clamp(0, 1)
        img_data = (img_data * 255).type(torch.uint8)
        save_images(img_data, os.path.join("results", args.run_name, f"sample_imgs_from_dataset.jpg"))
    else:
        sample_imgs_from_dataset = None


    losses = []

    # avoid displaying matplotlib figures
    plt.ioff()


    for epoch in range(args.epochs):
        losses_epoch = 0
        logging.info(f"Starting epoch {epoch}:")
        pbar = tqdm(range(args.steps_per_epoch))
        for i in pbar:
            images, _ = next(iter(train_dataloader))
            images = images.to(device)
            t = diffusion.sample_timesteps(images.shape[0]).to(device)
            if prediction_type == "noise":
                x_t, noise, prev_x = diffusion.noise_images(images, t, prediction_type=prediction_type)
                predicted_noise = model(x_t, t)
                loss = mse(noise, predicted_noise)
            elif prediction_type == "image":
                x_t, noise, prev_x = diffusion.noise_images(images, t, prediction_type=prediction_type)
                predicted_x = model(x_t, t)
                loss = mse(prev_x, predicted_x)
            elif prediction_type == "super_resolution":
                x_L, x_t, noise = diffusion.super_resolution_noise_data(images, t)
                # Concat diffusion step x_t and low resolution image x_L at channel dimension dim=1
                model_input = torch.concat((x_L, x_t), dim=1)
                predicted_noise = model(model_input, t)
                loss = mse(noise, predicted_noise)
            elif prediction_type == "colourise":
                x_g, x_t, noise = diffusion.colourise_noise_data(images, t)
                # Concat diffusion step x_t and grayscaled image x_g at channel dimension dim=1
                model_input = torch.concat((x_g, x_t), dim=1)
                predicted_noise = model(model_input, t)
                loss = mse(noise, predicted_noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pbar.set_postfix(MSE=loss.item())
            losses_epoch += loss.item()

        sampled_images = diffusion.sample(
            model, 
            n=num_sample_imgs, 
            prediction_type=prediction_type, 
            x=noise_sample,
            original_imgs=sample_imgs_from_dataset
        )
        save_images(sampled_images, os.path.join("results", args.run_name, f"{epoch}.jpg"))
        torch.save(model.state_dict(), os.path.join("models", args.run_name, f"ckpt.pt"))
        losses.append(losses_epoch / args.steps_per_epoch)
        fig = plt.figure()
        plt.plot(losses)
        path_img = Path(f'results/{args.run_name}/losses.png')
        fig.savefig(path_img)
        plt.close(fig)


def launch():
    import argparse
    parser = argparse.ArgumentParser()
    dataset = "celeba"
    args = parser.parse_args()
    args.epochs = 1000
    args.steps_per_epoch = 1000
    args.noise_steps = 250
    args.batch_size = 32
    args.image_size = 128
    args.num_patches = 2
    args.level_mult = [1,2,24]
    args.dataset_path = f"data/{dataset}"
    args.device = "cuda:2"
    args.device_ids = [2,3]
    args.lr = 3e-4
    args.hidden = 32
    args.prediction_type = "colourise"
    args.super_resolution_factor = 4 # is ignored if prediction_type is not super_resolution
    args.dropout = 0.01
    args.use_self_attention = False
    time_str = datetime.now().strftime('%Y-%m-%dT%H-%M-%S')
    args.run_name = f"{time_str}_DDPM_{args.num_patches}x{args.num_patches}_patches_{dataset}_{args.epochs}_epochs"
    train(args)


if __name__ == '__main__':
    launch()