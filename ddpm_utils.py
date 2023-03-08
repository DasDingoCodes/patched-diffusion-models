import os
import torch
import torchvision
from torchvision import transforms
from PIL import Image
from pathlib import Path
from matplotlib import pyplot as plt
from torch.utils.data import Dataset, DataLoader, random_split
from torchmetrics.image.fid import FrechetInceptionDistance
import numpy as np
import torchvision.transforms.functional as TF
from skimage import io

def plot_images(images):
    plt.figure(figsize=(32, 32))
    plt.imshow(torch.cat([
        torch.cat([i for i in images.cpu()], dim=-1),
    ], dim=-2).permute(1, 2, 0).cpu())
    plt.show()


def save_images(images, path, **kwargs):
    grid = torchvision.utils.make_grid(images, **kwargs)
    ndarr = grid.permute(1, 2, 0).to('cpu').numpy()
    im = Image.fromarray(ndarr)
    im.save(path)


def get_data(args, sample_percentage = 0.1):
    transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize(80),  # args.image_size + 1/4 *args.image_size
        torchvision.transforms.RandomResizedCrop(args.image_size, scale=(0.8, 1.0)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    dataset = torchvision.datasets.ImageFolder(Path(args.dataset_path), transform=transforms)
    sample_dataset, train_dataset = random_split(dataset, (sample_percentage, 1.0 - sample_percentage))
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    sample_dataloader = DataLoader(sample_dataset, batch_size=args.batch_size, shuffle=True)
    return train_dataloader, sample_dataloader

def get_fid_init(args, batch_size=32, max_iter: int = None, feature: int = 64, normalize: bool = True):
    """Returns FrechetInceptionDistance object initialised with values of given dataset.
    See: https://torchmetrics.readthedocs.io/en/stable/image/frechet_inception_distance.html
    
    args.dataset_path: string of path to dataset
    batch_size: batch_size for iterating through dataset, defaults to 32
    max_iter: maximum number of iterations/batches that shall be used for FID calculation. Will run through entire dataset if max_iter = None
    feature: indicates the inceptionv3 feature layer to choose. Can be one of the following: 64, 192, 768, 2048
    normalize: whether to normalise images to [0,1]
    """
    transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize(299),  # torchmetrics FID calculation rescales images to 299x299
        torchvision.transforms.ToTensor(),
    ])
    dataset = torchvision.datasets.ImageFolder(Path(args.dataset_path), transform=transforms)
    whole_dataset = DataLoader(dataset, batch_size=batch_size)
    fid = FrechetInceptionDistance(feature=feature, normalize=normalize, reset_real_features=False)
    for i, (x, _) in enumerate(whole_dataset):
        if type(max_iter) == int and max_iter <= i:
            break
        print(f"{i}/{len(whole_dataset)}", end="\r")
        fid.update(x, real=True)
    return fid


class NormalizeInverse(torchvision.transforms.Normalize):
    """
    Undoes the normalization and returns the reconstructed images in the input domain.
    """

    def __init__(self, mean, std):
        mean = torch.as_tensor(mean)
        std = torch.as_tensor(std)
        std_inv = 1 / (std + 1e-7)
        mean_inv = -mean * std_inv
        super().__init__(mean=mean_inv, std=std_inv)

    def __call__(self, tensor):
        return super().__call__(tensor.clone())

def setup_logging(run_name):
    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    os.makedirs(os.path.join("models", run_name), exist_ok=True)
    os.makedirs(os.path.join("results", run_name), exist_ok=True)

def get_data_img_mask_text(args, sample_percentage):
    dataset = TextMaskDataset(
        image_dir=args.inpainting_image_dir,
        mask_dir=args.inpainting_mask_dir,
        text_dir=args.inpainting_text_dir,
        texts_per_img=args.inpainting_texts_per_img,
        img_size=args.image_size
    )
    sample_dataset, train_dataset = random_split(dataset, (sample_percentage, 1.0 - sample_percentage))
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    sample_dataloader = DataLoader(sample_dataset, batch_size=args.batch_size, shuffle=True)
    return train_dataloader, sample_dataloader

class TextMaskDataset(Dataset):
    """Images with masks and descriptions"""

    def __init__(self, image_dir, mask_dir, text_dir, texts_per_img=10, img_size=128):
        """
        Args:
            image_dir: Path of directory containing image files. All images are .jpg files and are named only with an index
            mask_dir: Path of directory containing mask files. All masks are .png files and are named only with an index
            text_dir: Path of directory containing embedded descriptions of the corresponding image. 
                For each image there is a folder and in each folder there are texts_per_img tensor files with the embedded description sentences 
            texts_per_img: How many descriptions there are for each image, defaults to 10
            img_size: height and width which the images shall be scaled to, defaults to 128
        """
        self.path_image_dir = Path(image_dir)
        self.path_mask_dir = Path(mask_dir)
        self.path_text_dir = Path(text_dir)
        self.texts_per_img = texts_per_img
        self.img_size = img_size

    def __len__(self):
        return len([x for x in self.path_image_dir.iterdir()])
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        path_img = self.path_image_dir / f"{idx}.jpg"
        path_mask = self.path_mask_dir / f"{idx}.png"

        image = io.imread(path_img)
        mask = io.imread(path_mask)
        random_description_index = np.random.randint(self.texts_per_img)
        embedded_description = self.path_text_dir / idx / f"{random_description_index}.pt"

        # transforms
        # To Tensor
        image = transforms.ToTensor()(image)
        mask = transforms.ToTensor()(mask)

        # Resize 
        resize = transforms.Resize(size=(self.img_size + 10, self.img_size + 10))
        image = resize(image)
        mask = resize(mask)

        # Random crop
        i, j, h, w = transforms.RandomCrop.get_params(
            image, output_size=(self.img_size, self.img_size))
        image = TF.crop(image, i, j, h, w)
        mask = TF.crop(mask, i, j, h, w)

        # Random horizontal flipping
        if np.random.random() > 0.5:
            image = TF.hflip(image)
            mask = TF.hflip(mask)

        # Rotate
        rotation = transforms.RandomRotation.get_params(degrees=[-30,30])
        image = TF.rotate(image, rotation)
        mask = TF.rotate(mask, rotation)

        # Normalise
        image = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(image)

        return image, mask, embedded_description

def remove_masked_area(images, masks):
    """Removes those areas in the images that are covered by the corresponding masks"""
    return images * (1 - masks)