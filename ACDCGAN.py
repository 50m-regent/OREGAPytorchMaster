import numpy
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision.datasets import FashionMNIST
from torchvision.transforms import ToTensor
from torchsummary import summary
from tqdm import tqdm
import cv2

class DeConv(nn.Module):
    def __init__(
        self,
        in_channels, out_channels
    ):
        super().__init__()
        
        self.net = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels, out_channels,
                kernel_size=4, stride=2
            ),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(),
            nn.Dropout2d()
        )
    
    def forward(self, x):
        return self.net(x)

class Generator(nn.Module):
    def __init__(
        self,
        noise_features,
        hidden_features=128
    ):
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Linear(noise_features, hidden_features),
            nn.BatchNorm1d(hidden_features),
            nn.LeakyReLU(),
            
            nn.Unflatten(1, (hidden_features, 1, 1)),
            nn.Upsample(scale_factor=2),
            
            DeConv(hidden_features, 64),
            DeConv(64, 32),
            DeConv(32, 16),
            
            nn.Conv2d(
                in_channels=16, out_channels=1,
                kernel_size=3
            ),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        return self.net(x)
    
class Conv(nn.Module):
    def __init__(
        self,
        in_channels, out_channels,
        kernel_size
    ):
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Conv2d(
                in_channels, out_channels,
                kernel_size=kernel_size, stride=2
            ),
            nn.LeakyReLU()
        )
        
    def forward(self, x):
        return self.net(x)
    
class Discriminator(nn.Module):
    def __init__(self, in_channels, out_classes):
        super().__init__()
        
        self.net = nn.Sequential(
            Conv(in_channels, 8, kernel_size=4),
            Conv(8, 16,  kernel_size=5),
            Conv(16, 32,  kernel_size=5),
            
            nn.Flatten(),
            
            nn.Linear(32, out_classes),
            nn.Softmax(dim=1)
        )
        
    def forward(self, x):
        return self.net(x)
    
class Trainer:
    def __init__(
        self,
        generator, discriminator,
        train_loader:DataLoader,
        device,
        noise_features,
        num_classes
    ):
        self.generator:Generator         = generator.to(device)
        self.discriminator:Discriminator = discriminator.to(device)
        
        self.train_loader = train_loader
        
        self.device         = device
        self.noise_features = noise_features
        self.num_classes    = num_classes
        
        self.criterion = nn.BCELoss()
        
        self.generator_optimizer     = optim.Adam(self.generator.parameters())
        self.discriminator_optimizer = optim.Adam(self.discriminator.parameters())
        
        self.noise = torch.randn((100, noise_features), dtype=torch.float, device=self.device)
        self.noise = torch.cat((
            self.noise,
            nn.functional.one_hot(
                torch.tensor([
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                    2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
                    3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
                    4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
                    5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
                    6, 6, 6, 6, 6, 6, 6, 6, 6, 6,
                    7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
                    8, 8, 8, 8, 8, 8, 8, 8, 8, 8,
                    9, 9, 9, 9, 9, 9, 9, 9, 9, 9,
                ]),
                self.num_classes + 1
            ).to(torch.float).to(self.device)
        ), dim=1)
        
        self.fake_y = torch.full((train_loader.batch_size,), num_classes)
        self.fake_y = nn.functional.one_hot(self.fake_y, num_classes + 1).to(torch.float).to(self.device)
        
    def step(self, epoch):
        self.generator.train()
        self.discriminator.train()
        
        generator_loss_sum     = 0
        discriminator_loss_sum = 0
        true_prediction_sum    = 0
        fake_prediction_sum    = 0
        for x, y in tqdm(self.train_loader, desc = 'Training'):
            true_x = x.to(self.device)
            true_y = nn.functional.one_hot(y, self.num_classes + 1).to(torch.float).to(self.device)
            
            noise  = torch.randn((len(x), self.noise_features), device=self.device)
            noise  = torch.cat((noise, true_y), dim=1)
            fake_x = self.generator(noise)
            
            # --------------------------------------------------------------
            
            true_prediction:torch.Tensor = self.discriminator(true_x)
            fake_prediction:torch.Tensor = self.discriminator(fake_x.detach())
            
            discriminator_loss:torch.Tensor = self.criterion(
                true_prediction, true_y
            ) + self.criterion(
                fake_prediction, self.fake_y
            )
            discriminator_loss_sum += discriminator_loss.item()
            
            true_prediction_sum += torch.where(true_prediction.argmax(dim=1) == true_y.argmax(dim=1), 1., 0.).mean()
            fake_prediction_sum += torch.where(fake_prediction.argmax(dim=1) == self.fake_y.argmax(dim=1), 1., 0.).mean()
            
            self.discriminator_optimizer.zero_grad()
            discriminator_loss.backward()
            self.discriminator_optimizer.step()
                
            # --------------------------------------------------------------
                
            fake_prediction = self.discriminator(fake_x)
            
            generator_loss:torch.Tensor = self.criterion(
                fake_prediction, true_y
            )
            generator_loss_sum += generator_loss.item()
            
            self.generator_optimizer.zero_grad()
            generator_loss.backward()
            self.generator_optimizer.step()

        fake = self.generator(self.noise).detach().cpu().numpy().reshape(10, 10, 28, 28)
        fake = numpy.concatenate(fake, axis=1)
        fake = numpy.concatenate(fake, axis=1)
        cv2.imwrite(f'temp/{epoch}.png', fake * 255)

        batch_count = len(self.train_loader)
        return \
            generator_loss_sum / batch_count, \
            discriminator_loss_sum / batch_count, \
            (true_prediction_sum / batch_count, fake_prediction_sum / batch_count)
        
    def run(self, num_epoch:int=100):
        for epoch in tqdm(range(1, num_epoch + 1), desc='Epochs'):
            generator_train_loss, discriminator_train_loss, accuracy = self.step(epoch)
            
            tqdm.write(
                f'Epoch {epoch:>3}: ' \
                f'loss = ({generator_train_loss:>.7f}, {discriminator_train_loss:>.7f}), ' \
                f'accuracy = ({accuracy[0]:>.7f}, {accuracy[1]:>.7f})'
            )

def get_dataset():    
    train_set = FashionMNIST('data/FashionMNIST', train=True,  transform=ToTensor(), download=True)
    
    return train_set
    
def main():
    NOISE_FEATURES = 32
    NUM_CLASSES    = 10
    BATCH_SIZE     = 64
    DEVICE         = 'mps'
    
    train_set = get_dataset()
    train_loader = DataLoader(
        train_set,
        BATCH_SIZE,
        shuffle=True,
        drop_last=True
    )
    
    generator = Generator(NOISE_FEATURES + NUM_CLASSES + 1)
    summary(generator, (NOISE_FEATURES + NUM_CLASSES + 1,))
    
    discriminator = Discriminator(train_set[0][0].shape[0], NUM_CLASSES + 1)
    summary(discriminator, train_set[0][0].shape)
    
    trainer = Trainer(
        generator, discriminator,
        train_loader,
        DEVICE, NOISE_FEATURES, NUM_CLASSES
    )
    trainer.run()

if __name__ == '__main__':
    main()