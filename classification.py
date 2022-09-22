from typing import Callable, Tuple, Dict, Iterable
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
    global DEVICE, CHECK_DATA, LEARNING_RATE, BATCH_SIZE, EPOCHS, NUM_WORKERS, CHECKPOINT, TERMINATE_COUNT

    parser: ArgumentParser = ArgumentParser()
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda', 'mps'])
    parser.add_argument('--check_data', action='store_true')
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--num_workers', type=int, default=3)
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--terminate_count', type=int, default=5)
    
    args: argparse.Namespace = parser.parse_args()
    DEVICE          = torch.device(args.device)
    CHECK_DATA      = args.check_data
    LEARNING_RATE   = args.learning_rate
    BATCH_SIZE      = args.batch_size
    EPOCHS          = args.epochs
    NUM_WORKERS     = args.num_workers
    CHECKPOINT      = args.checkpoint
    TERMINATE_COUNT = args.terminate_count
    
    
def get_dataset() -> Tuple[FashionMNIST, FashionMNIST]:
    training_data: FashionMNIST = FashionMNIST(
        root="data",
        train=True,
        download=True,
        transform=ToTensor(),
    )

    test_data: FashionMNIST     = FashionMNIST(
        root="data",
        train=False,
        download=True,
        transform=ToTensor(),
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


class Model(nn.Module):
    def __init__(self) -> None:
        super(Model, self).__init__()
        
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, stride=2)

        self.conv1 = nn.Conv2d(1, 32, 3, padding=[1, 1], padding_mode='replicate')
        self.conv2 = nn.Conv2d(32, 64, 3, padding=[1, 1], padding_mode='replicate')
        self.conv3 = nn.Conv2d(64, 128, 3, padding=[1, 1], padding_mode='replicate')
        self.conv4 = nn.Conv2d(128, 128, 3, padding=[1, 1], padding_mode='replicate')

        self.fc1 = nn.Linear(128 * 3 * 3, 512)
        self.fc2 = nn.Linear(512, 10)
        
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout2d(0.3)
        self.batch_normalization = nn.BatchNorm2d(64)
        
        self.fs: list[Callable] = [
            self.conv1,
            self.relu,
            self.pool,
            self.conv2,
            self.relu,
            self.pool,
            self.dropout,
            self.batch_normalization,
            self.conv3,
            self.conv4,
            self.relu,
            self.pool,
            self.dropout,
            self.flatten,
            self.fc1,
            self.relu,
            self.fc2,
        ]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for f in self.fs:
            x = f(x)

        return x
    
    
class SAMSGD(torch.optim.SGD):
    def __init__(self,
                 params: Iterable[torch.Tensor],
                 lr: float,
                 momentum: float = 0,
                 dampening: float = 0,
                 weight_decay: float = 0,
                 nesterov: bool = False,
                 rho: float = 0.05,
                 ):
        if rho <= 0:
            raise ValueError(f"Invalid neighborhood size: {rho}")
        super().__init__(params, lr, momentum, dampening, weight_decay, nesterov)
        # todo: generalize this
        if len(self.param_groups) > 1:
            raise ValueError("Not supported")
        self.param_groups[0]["rho"] = rho

    @torch.no_grad()
    def step(self,
             closure
             ) -> torch.Tensor:
        """
        Args:
            closure: A closure that reevaluates the model and returns the loss.
        Returns: the loss value evaluated on the original point
        """
        closure = torch.enable_grad()(closure)
        loss = closure().detach()

        for group in self.param_groups:
            grads = []
            params_with_grads = []

            rho = group['rho']
            # update internal_optim's learning rate

            for p in group['params']:
                if p.grad is not None:
                    # without clone().detach(), p.grad will be zeroed by closure()
                    grads.append(p.grad.clone().detach())
                    params_with_grads.append(p)
            device = grads[0].device

            # compute \hat{\epsilon}=\rho/\norm{g}\|g\|
            grad_norm = torch.stack([g.detach().norm(2).to(device) for g in grads]).norm(2)
            epsilon = grads  # alias for readability
            torch._foreach_mul_(epsilon, rho / grad_norm)

            # virtual step toward \epsilon
            torch._foreach_add_(params_with_grads, epsilon)
            # compute g=\nabla_w L_B(w)|_{w+\hat{\epsilon}}
            closure()
            # virtual step back to the original point
            torch._foreach_sub_(params_with_grads, epsilon)

        super().step()
        return loss
    
    
class Trainer:
    def __init__(self, training_data: Dataset, test_data: Dataset, device: torch.device, model: nn.Module, learning_rate: float, batch_size: int, num_workers: int) -> None:
        self.device: torch.device = device
        
        self.model: nn.Module = model.to(self.device)
        
        self.learning_rate: float = learning_rate
        self.batch_size: int      = batch_size
        self.num_workers: int     = num_workers
        
        self.loss_function: Callable          = nn.CrossEntropyLoss()
        # self.optimizer: torch.optim.Optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)
        self.optimizer: torch.optim.Optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        # self.optimizer: torch.optim.Optimizer = SAMSGD(self.model.parameters(), self.learning_rate)
        
        self.create_loader(training_data, test_data)
    
    def create_loader(self, training_data: Dataset, test_data: Dataset) -> None:
        self.training_loader: DataLoader = DataLoader(
            training_data,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )
        self.test_loader: DataLoader     = DataLoader(
            test_data,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def train_loop(self) -> None:
        for X, y in tqdm(self.training_loader, desc='training'):     
            X: torch.Tensor = X.to(self.device)
            y: torch.Tensor = y.to(self.device)
            
            prediction: torch.Tensor = self.model(X)
            loss: float = self.loss_function(prediction, y)
            
            self.optimizer.zero_grad()
            loss.backward()
            
            def closure():
                return loss
            
            self.optimizer.step(closure)

    def test_loop(self) -> Tuple[float, float]:
        size = len(self.test_loader.dataset)
        loss: float = 0
        accuracy: float   = 0

        with torch.no_grad():
            for X, y in self.test_loader:
                X: torch.Tensor = X.to(self.device)
                y: torch.Tensor = y.to(self.device)
                
                prediction: torch.Tensor = self.model(X)
                loss += self.loss_function(prediction, y).item() / size
                accuracy += (prediction.argmax(1) == y).type(torch.float).sum().item() / size
                
            tqdm.write(f'accuracy: {accuracy * 100:>0.1f}%, loss: {loss:>0.7f}, ', end='')
            
        return accuracy, loss
    
    def save_model(self, path: str) -> None:
        torch.save(self.model, path)
        
    def run(self, epochs: int, terminate_count: int, save_path: str = 'models/model.pth') -> None:
        min_loss: float = float('inf')
        update_count: int = 0
        for t in tqdm(range(epochs), desc='epoch'):
            self.train_loop()
            accuracy: float
            loss: float
            accuracy, loss = self.test_loop()
            
            if min_loss > loss:
                min_loss = loss
                update_count = 0
            update_count += 1
            
            tqdm.write(f'min_loss: {min_loss:>0.7f}\n')
            
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
    
    model = Model()
    if CHECKPOINT != None:
        model = load_model(CHECKPOINT)

    trainer: Trainer = Trainer(training_data, test_data, DEVICE, model, LEARNING_RATE, BATCH_SIZE, NUM_WORKERS)
    trainer.run(EPOCHS, TERMINATE_COUNT)
