import torch
from torch import Tensor, nn
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from torchsummary import summary
from tqdm import tqdm

class Model(nn.Module):
    def __init__(
        self,
        in_channels:int,
        features:int,
        classes:int
    ):
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, features // 4, kernel_size=3),
            nn.MaxPool2d(kernel_size=2),
            nn.ReLU(),
            
            nn.Conv2d(features // 4, features // 2, kernel_size=4),
            nn.MaxPool2d(kernel_size=2),
            nn.ReLU(),
            
            nn.Conv2d(features // 2, features, kernel_size=4),
            nn.MaxPool2d(kernel_size=2),
            nn.ReLU(),
            
            nn.Flatten(),
            nn.Linear(features, classes),
            nn.Softmax(dim=1)
        )
        
    def forward(self, x:Tensor) -> Tensor:
        return self.net(x)
    
class Trainer:
    class Decoy:
        def __enter__(self):
            pass
        
        def __exit__(self, _, __, ___):
            pass
    
    def __init__(
        self,
        model:Model,
        train_loader:DataLoader, valid_loader:DataLoader,
        device:str
    ):
        self.model:Model = model.to(device)
        
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        
        self.device:str = device
        
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = Adam(self.model.parameters())
        
    def step(self, train:bool=True) -> tuple[float, float]:
        if train:
            self.model.train()
            no_grad = self.Decoy
        else:
            self.model.eval()
            no_grad = torch.no_grad
        
        loss_sum    = 0
        correct_sum = 0
        with no_grad():
            for x, y in tqdm(
                self.train_loader if train else self.valid_loader,
                desc = 'Training' if train else 'Validating'
            ):
                x = x.to(self.device)
                y = y.to(self.device)
                
                prediction:Tensor = self.model(x)
                
                loss:Tensor = self.criterion(prediction, y)
                
                loss_sum    += loss.item()
                correct_sum += torch.where(prediction.argmax(dim=1) == y, 1., 0.).mean()
                
                if train:
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
            
        data_count = len(self.train_loader if train else self.valid_loader)
        return loss_sum / data_count, correct_sum / data_count
        
    def run(self, num_epoch:int=30):
        for epoch in range(1, num_epoch + 1):
            train_loss, train_accuracy = self.step()
            valid_loss, valid_accuracy = self.step(train=False)
            
            tqdm.write(
                f'Epoch {epoch:>3}: ' \
                f'train_loss = {train_loss:>.7f}, ' \
                f'train_accuracy = {train_accuracy:>.7f}, ' \
                f'valid_loss = {valid_loss:>.7f}, ' \
                f'valid_accuracy = {valid_accuracy:>.7f}'
            )

def get_dataset() -> tuple[Dataset, Dataset]:
    train_set = MNIST('data/MNIST', train=True,  download=True, transform=ToTensor())
    valid_set = MNIST('data/MNIST', train=False, download=True, transform=ToTensor())
    
    return train_set, valid_set

def main():
    BATCH_SIZE = 256
    FEATURES   = 128
    CLASSES    = 10
    DEVICE     = 'mps'
    
    train_set, valid_set = get_dataset()
    train_loader = DataLoader(
        train_set,
        BATCH_SIZE,
        shuffle=True
    )
    valid_loader = DataLoader(
        valid_set,
        BATCH_SIZE
    )
    print(f'Train data: ({len(train_set)}, {train_set[0][0].shape})')
    print(f'Valid data: ({len(valid_set)}, {valid_set[0][0].shape})')
    
    model = Model(
        in_channels=train_set[0][0].shape[0],
        features=FEATURES,
        classes=CLASSES
    )
    summary(model, input_size=train_set[0][0].shape)
    
    trainer = Trainer(model, train_loader, valid_loader, DEVICE)
    trainer.run()

if __name__ == '__main__':
    main()