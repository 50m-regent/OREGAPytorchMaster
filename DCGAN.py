from typing import Callable, Tuple, Dict
import argparse
from argparse import ArgumentParser

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import FashionMNIST
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from tqdm import tqdm
import cv2


LABELS: Dict[int, str] = {
    0: "T-Shirt",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle Boot",
}


def accept_parameters() -> None:
    global DEVICE, CHECK_DATA, LEARNING_RATE, BATCH_SIZE, EPOCHS, NUM_WORKERS, GENERATOR_CHECKPOINT, DISCRIMINATOR_CHECKPOINT, NOISE_DIMENSION, TERMINATE_COUNT

    parser: ArgumentParser = ArgumentParser()
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda', 'mps'])
    parser.add_argument('--check_data', action='store_true')
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--num_workers', type=int, default=3)
    parser.add_argument('--generator_checkpoint', type=str, default=None)
    parser.add_argument('--discriminator_checkpoint', type=str, default=None)
    parser.add_argument('--noise_dimension', type=int, default=128)
    parser.add_argument('--terminate_count', type=int, default=10)
    
    args: argparse.Namespace = parser.parse_args()
    DEVICE                   = torch.device(args.device)
    CHECK_DATA               = args.check_data
    LEARNING_RATE            = args.learning_rate
    BATCH_SIZE               = args.batch_size
    EPOCHS                   = args.epochs
    NUM_WORKERS              = args.num_workers
    GENERATOR_CHECKPOINT     = args.generator_checkpoint
    DISCRIMINATOR_CHECKPOINT = args.discriminator_checkpoint
    NOISE_DIMENSION          = args.noise_dimension
    TERMINATE_COUNT          = args.terminate_count
    
    
def get_dataset() -> Tuple[FashionMNIST, FashionMNIST]:
    training_data: FashionMNIST = FashionMNIST(
        root="data",
        train=True,
        download=True,
        transform=ToTensor()
    )

    test_data: FashionMNIST     = FashionMNIST(
        root="data",
        train=False,
        download=True,
        transform=ToTensor()
    )

    return training_data, test_data


def show_data(training_data: FashionMNIST) -> None:
    figure: Figure = plt.figure(figsize=(8, 8))
    cols: int = 3
    rows: int = 3

    for i in range(1, cols * rows + 1):
        sample_idx: torch.Tensor = torch.randint(len(training_data), size=(1,)).item()
        img: torch.Tensor
        label: torch.Tensor
        img, label = training_data[sample_idx]
        figure.add_subplot(rows, cols, i)
        plt.title(LABELS[label])
        plt.axis("off")
        plt.imshow(img.squeeze(), cmap="gray")
        
    plt.show()
    
    
def load_model(path: str) -> nn.Module:
    return torch.load(path)


class Generator(nn.Module):
    def __init__(
        self,
        noise_dimension: int,
        color_channel: int,
        feature_map_size: int = 64
    ) -> None:
        super(Generator, self).__init__()
        
        self.net: nn.Module = nn.Sequential(
            nn.ConvTranspose2d(noise_dimension, feature_map_size * 4, 4, 1, 0, bias=False),
            nn.BatchNorm2d(feature_map_size * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(feature_map_size * 4, feature_map_size * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_map_size * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(feature_map_size * 2, feature_map_size, 3, 2, 1, bias=False),
            nn.BatchNorm2d(feature_map_size),
            nn.ReLU(True),
            nn.ConvTranspose2d(feature_map_size, color_channel, 2, 2, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
    
    
class Discriminator(nn.Module):
    def __init__(
        self,
        color_channel: int,
        feature_map_size: int = 64
    ) -> None:
        super(Discriminator, self).__init__()
        
        self.net: nn.Module = nn.Sequential(
            nn.Conv2d(color_channel, feature_map_size, 4, 2, 0, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(feature_map_size, feature_map_size * 2, 4, 2, 0, bias=False),
            nn.BatchNorm2d(feature_map_size * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(feature_map_size * 2, 1, 4, 2, 0, bias=False),
            nn.Flatten(),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
    
    
class GANTrainer:
    def __init__(
        self,
        training_data: Dataset,
        device: torch.device,
        generator: nn.Module,
        discriminator: nn.Module,
        noise_dimension: int,
        learning_rate: float,
        batch_size: int,
        num_workers: int
    ) -> None:
        self.DISCRIMINATOR_TARGETS = {
            'real': torch.ones(batch_size, 1),
            'fake': torch.zeros(batch_size, 1)
        }
        
        self.device: torch.device = device
        
        self.generator: nn.Module = generator.to(self.device)
        self.discriminator: nn.Module = discriminator.to(self.device)
        
        self.noise_dimension: int = noise_dimension
        
        self.learning_rate: float = learning_rate
        self.batch_size: int      = batch_size
        self.num_workers: int     = num_workers
        
        self.loss_function: Callable = nn.BCELoss()
        
        self.generator_optimizer: torch.optim.Optimizer     = torch.optim.Adam(self.generator.parameters(), lr=self.learning_rate)
        self.discriminator_optimizer: torch.optim.Optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=self.learning_rate)
        
        self.training_loader: DataLoader = DataLoader(
            training_data,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True
        )

    def train_loop(self) -> Tuple[float, float]:
        generator_loss_sum: float = 0
        discriminator_loss_sum: float = 0
        for input, _ in tqdm(self.training_loader, desc='training'):
            generator_output = self.generator(torch.randn(self.batch_size, self.noise_dimension, 1, 1))
            
            discriminator_loss: nn.Module = \
                self.loss_function(self.discriminator(input), self.DISCRIMINATOR_TARGETS['real']) + \
                self.loss_function(self.discriminator(generator_output.detach()), self.DISCRIMINATOR_TARGETS['fake'])

            self.discriminator_optimizer.zero_grad()
            discriminator_loss.backward()
            self.discriminator_optimizer.step()

            generator_loss: nn.Module = self.loss_function(self.discriminator(generator_output), self.DISCRIMINATOR_TARGETS['real'])
            
            self.generator_optimizer.zero_grad()
            generator_loss.backward()
            self.generator_optimizer.step()
            
            generator_loss_sum += generator_loss
            discriminator_loss_sum += discriminator_loss
            
        return generator_loss_sum / self.batch_size, discriminator_loss_sum / self.batch_size
    
    def save_model(self, path: str) -> None:
        torch.save(self.model, path)
        
    def save_output(self, epoch: int) -> None:
        output = self.generator(torch.randn(1, self.noise_dimension, 1, 1, device=self.device)).detach().cpu().numpy()[0][0] * 256
        cv2.imwrite(f'outputs/DCGAN/{epoch:04}.png', output)
        
    def run(self, epochs: int, terminate_count: int, save_path: str = 'models/model.pth') -> None:
        min_generator_loss: float     = float('inf')
        min_discriminator_loss: float = float('inf')
        update_count: int = 0
        for epoch in tqdm(range(epochs), desc='epoch'):
            generator_loss: float
            discriminator_loss: float
            generator_loss, discriminator_loss = self.train_loop()
            
            self.save_output(epoch)
            
            if min_generator_loss > generator_loss:
                min_generator_loss = generator_loss
                update_count = 0
            if min_discriminator_loss > discriminator_loss:
                min_discriminator_loss = discriminator_loss
                update_count = 0
            update_count += 1
            
            tqdm.write(
                f'generator_loss: {generator_loss:0.7f}, min_generator_loss: {min_generator_loss:>0.7f}\n\
                discriminator_loss: {discriminator_loss:0.7f}, min_discriminator_loss: {min_discriminator_loss:>0.7f}\n'
            )
            
            if update_count >= terminate_count:
                break
            
        # self.save_model(save_path)


if __name__ == '__main__':
    accept_parameters()

    training_data: FashionMNIST
    test_data: FashionMNIST
    training_data, test_data = get_dataset()
    if CHECK_DATA:
        show_data(training_data)
    
    generator: Generator         = Generator(NOISE_DIMENSION, 1)
    if GENERATOR_CHECKPOINT != None:
        generator = load_model(GENERATOR_CHECKPOINT)
    discriminator: Discriminator = Discriminator(1)
    if DISCRIMINATOR_CHECKPOINT != None:
        generator = load_model(DISCRIMINATOR_CHECKPOINT)

    trainer: GANTrainer = GANTrainer(training_data, DEVICE, generator, discriminator, NOISE_DIMENSION, LEARNING_RATE, BATCH_SIZE, NUM_WORKERS)
    trainer.run(EPOCHS, TERMINATE_COUNT)
