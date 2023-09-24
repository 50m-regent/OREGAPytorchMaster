import numpy
import torch
from torch import nn, optim, Tensor
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import FashionMNIST
from torchvision.transforms import ToTensor
from tqdm import tqdm
import cv2

class Generator(nn.Module):
    def __init__(
        self,
        in_features:int,
        hidden_features:int,
        num_head:int,
        num_layers:int,
        out_size:int,
        out_channels:int
    ) -> None:
        super().__init__()
        
        self.out_size     = out_size
        self.out_channels = out_channels
        
        self.entrance = nn.Linear(in_features, hidden_features)
        self.upsample = nn.PixelShuffle(2)
        
        self.encoder1 = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_features,
                nhead=num_head,
                batch_first=True
            ),
            num_layers=num_layers
        )
        self.encoder2 = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_features // 4,
                nhead=num_head,
                batch_first=True
            ),
            num_layers=num_layers
        )
        
        self.exit = nn.Sequential(
            nn.Linear(hidden_features // 16, out_channels),
            nn.Sigmoid()
        )
        
    def forward(self, x:Tensor) -> Tensor:
        x = self.entrance(x)
        
        x = x.flatten(1, 2)
        x = self.encoder1(x)
        
        x = x.unflatten(1, (self.out_size // 4, self.out_size // 4))
        x = self.upsample(torch.permute(x, (0, 3, 1, 2)))
        x = x.flatten(2, 3).transpose(1, 2)
        x = self.encoder2(x)
        
        x = x.unflatten(1, (self.out_size // 2, self.out_size // 2))
        x = self.upsample(torch.permute(x, (0, 3, 1, 2)))
        x = x.flatten(2, 3).transpose(1, 2)
        
        x = self.exit(x)
        
        return torch.permute(
            x.reshape(-1, self.out_size, self.out_size, self.out_channels),
            (0, 3, 1, 2)
        )

class Patcher(nn.Module):
    def __init__(self, patch_size:int) -> None:
        super().__init__()
        
        self.patching = nn.Unfold(
            kernel_size=(patch_size, patch_size),
            stride=patch_size
        )
        
    def forward(self, x:Tensor) -> Tensor:
        x:Tensor = self.patching(x)
        return x.transpose(1, 2)
    
class Embedding(nn.Module):
    def __init__(self, features:int, num_patches:int) -> None:
        super().__init__()
        
        self.token     = nn.Parameter(torch.randn(1, 1, features))
        self.embedding = nn.Parameter(torch.randn(1, num_patches + 1, features))
        
    def forward(self, x:Tensor) -> Tensor:
        assert 3 == len(x.shape)
        
        batch_size, *_ = x.shape
        
        tokens = torch.cat([self.token] * batch_size)
        x = torch.cat((tokens, x), dim = 1)
        
        x += self.embedding
        return x

'''
class Discriminator(nn.Module):
    def __init__(
        self,
        in_channels:int,
        image_size:int,
        patch_size:int,
        patch_features:int,
        num_head:int,
        num_layers:int,
        out_classes:int
    ) -> None:
        assert 0 == image_size % patch_size
        
        super().__init__()
        
        self.encoder = nn.Sequential(
            Patcher(patch_size),
            nn.Linear(in_channels * patch_size ** 2, patch_features),
            Embedding(
                patch_features,
                image_size // patch_size ** 2
            ),
            
            nn.TransformerEncoder(
                nn.TransformerEncoderLayer(
                    d_model=patch_features,
                    nhead=num_head,
                    batch_first=True
                ),
                num_layers=num_layers
            )
        )
        
        self.head = nn.Sequential(
            nn.Linear(patch_features, out_classes + 1),
            nn.Softmax(dim=1)
        )
        
    def forward(self, x:Tensor) -> Tensor:
        assert 4 == len(x.shape)

        x = self.encoder(x)
        return self.head(x[:, 0])
'''

class Discriminator(nn.Module):
    def __init__(self, in_size, out_classes):
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Flatten(),
            
            nn.Linear(in_size ** 2, 1024),
            nn.ReLU(),
            
            nn.Linear(1024, out_classes),
            nn.Softmax(dim=1)
        )
        
    def forward(self, x):
        return self.net(x)
    
class Trainer:
    def __init__(
        self,
        G:Generator, D:Discriminator,
        train_loader:DataLoader,
        device:str,
        noise_features:int,
        noise_size:int,
        num_classes:int
    ) -> None:
        self.G:Generator     = G.to(device)
        self.D:Discriminator = D.to(device)
        
        self.train_loader = train_loader
        
        self.device         = device
        self.noise_features = noise_features
        self.noise_size     = noise_size
        self.num_classes    = num_classes
        
        self.criterion = nn.BCELoss()
        
        self.G_optimizer = optim.Adam(self.G.parameters())
        self.D_optimizer = optim.Adam(self.D.parameters())
        
        self.noise = torch.randn(
            (100, noise_size, noise_size, noise_features),
            dtype=torch.float,
            device=self.device
        )
        label = nn.functional.one_hot(
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
        ).reshape(100, 1, 1, self.num_classes + 1).to(torch.float).to(self.device)
        label = self.expand_label(label)
        self.noise = torch.cat((self.noise, label), dim=3)
        
        self.fake_y:Tensor = torch.full((train_loader.batch_size,), num_classes)
        self.fake_y        = nn.functional.one_hot(self.fake_y, num_classes + 1).to(torch.float).to(self.device)
        
    def expand_label(self, x:Tensor) -> Tensor:
        return torch.cat(
            [torch.cat(
                [x] * self.noise_size,
                dim=1
            )] * self.noise_size,
            dim=2
        )
        
    def step(self, epoch:int) -> tuple[float, float, tuple[float, float]]:
        self.G.train()
        self.D.train()
        
        G_loss_sum = 0
        D_loss_sum = 0
        true_prediction_sum = 0
        fake_prediction_sum = 0
        for x, y in tqdm(self.train_loader, desc = 'Training'):
            true_x        = x.to(self.device)
            true_y:Tensor = nn.functional.one_hot(y, self.num_classes + 1).to(torch.float).to(self.device)
            
            noise = torch.randn(
                (len(x), self.noise_size, self.noise_size, self.noise_features),
                device=self.device
            )
            noise = torch.cat((
                noise,
                self.expand_label(true_y.view(len(true_y), 1, 1, -1))
            ), dim=3)
            fake_x:Tensor = self.G(noise)
            
            # --------------------------------------------------------------
            
            true_prediction:Tensor = self.D(true_x)
            fake_prediction:Tensor = self.D(fake_x.detach())
            
            D_loss:Tensor = self.criterion(
                true_prediction, true_y
            ) + self.criterion(
                fake_prediction, self.fake_y
            )
            D_loss_sum += D_loss.item()
            
            true_prediction_sum += torch.where(true_prediction.argmax(dim=1) == true_y.argmax(dim=1), 1., 0.).mean()
            fake_prediction_sum += torch.where(fake_prediction.argmax(dim=1) == self.fake_y.argmax(dim=1), 1., 0.).mean()
            
            self.D_optimizer.zero_grad()
            D_loss.backward()
            self.D_optimizer.step()
                
            # --------------------------------------------------------------
                
            fake_prediction = self.D(fake_x)
            
            G_loss:Tensor = self.criterion(
                fake_prediction, true_y
            )
            G_loss_sum += G_loss.item()
            
            self.G_optimizer.zero_grad()
            G_loss.backward()
            self.G_optimizer.step()

        fake = self.G(self.noise).detach().cpu().numpy().reshape(10, 10, 28, 28)
        fake = numpy.concatenate(fake, axis=1)
        fake = numpy.concatenate(fake, axis=1)
        cv2.imwrite(f'temp/{epoch}.png', fake * 255)

        batch_count = len(self.train_loader)
        return \
            G_loss_sum / batch_count, \
            D_loss_sum / batch_count, \
            (true_prediction_sum / batch_count, fake_prediction_sum / batch_count)
        
    def run(self, num_epoch:int=100) -> None:
        for epoch in tqdm(range(1, num_epoch + 1), desc='Epochs'):
            G_train_loss, D_train_loss, accuracy = self.step(epoch)
            
            tqdm.write(
                f'Epoch {epoch:>3}: ' \
                f'loss = ({G_train_loss:>.7f}, {D_train_loss:>.7f}), ' \
                f'accuracy = ({accuracy[0]:>.7f}, {accuracy[1]:>.7f})'
            )

def get_dataset() -> Dataset:    
    train_set = FashionMNIST('data', train=True,  transform=ToTensor(), download=True)
    
    return train_set
    
def main() -> None:
    NOISE_FEATURES  = 32
    NUM_CLASSES     = 10
    HIDDEN_FEATURES = 256
    NUM_HEAD        = 8
    NUM_LAYERS      = 3
    BATCH_SIZE      = 128
    DEVICE          = 'mps'
    
    train_set    = get_dataset()
    train_loader = DataLoader(
        train_set,
        BATCH_SIZE,
        shuffle=True,
        drop_last=True
    )
    
    G = Generator(
        NOISE_FEATURES + NUM_CLASSES + 1,
        HIDDEN_FEATURES,
        NUM_HEAD,
        NUM_LAYERS,
        out_size=train_set[0][0].shape[1],
        out_channels=train_set[0][0].shape[0]
    )
    '''
    D = Discriminator(
        train_set[0][0].shape[0],
        train_set[0][0].shape[1],
        train_set[0][0].shape[1] // 4,
        HIDDEN_FEATURES,
        NUM_HEAD,
        NUM_LAYERS,
        NUM_CLASSES
    )
    '''
    D = Discriminator(train_set[0][0].shape[1], NUM_CLASSES + 1)
    
    trainer = Trainer(
        G, D,
        train_loader,
        DEVICE,
        NOISE_FEATURES, train_set[0][0].shape[1] // 4,
        NUM_CLASSES
    )
    trainer.run()

if __name__ == '__main__':
    main()