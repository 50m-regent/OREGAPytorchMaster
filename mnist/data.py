from torchvision.datasets import MNIST
from torchvision.transforms import Compose, ToTensor, Normalize
from torch.utils.data import DataLoader
from numpy import linspace, uint8

def get_data():
    transform = Compose([
        ToTensor(),
        Normalize((0.5, ), (0.5, ))])

    train = MNIST(
        root='./data',
        train=True,
        download=True,
        transform=transform)
    test  = MNIST(
        root='./data',
        train = False,
        download=True,
        transform=transform)
    
    train_loader = DataLoader(
        train,
        batch_size=100,
        shuffle=True,
        num_workers=2)
    test_loader  = DataLoader(
        test,
        batch_size=100,
        shuffle=False,
        num_workers=2)

    classes = tuple(linspace(0, 9, 10, dtype=uint8))

    return train_loader, test_loader, classes