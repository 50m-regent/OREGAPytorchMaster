from typing import Callable, Tuple
import argparse
from argparse import ArgumentParser

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


LETTER_LABELS = {
    'a': 0,
    'b': 1,
    'c': 2,
    'd': 3,
    'e': 4,
    'f': 5,
    'g': 6,
    'h': 7,
    'i': 8,
    'j': 9,
    'k': 10,
    'l': 11,
    'm': 12,
    'n': 13,
    'o': 14,
    'p': 15,
    'q': 16,
    'r': 17,
    's': 18,
    't': 19,
    'u': 20,
    'v': 21,
    'w': 22,
    'x': 23,
    'y': 24,
    'z': 25,
    ' ': 26
}
LETTER_REVERSE_LABELS = 'abcdefghijklmnopqrstuvwxyz '


def accept_parameters() -> None:
    global DEVICE, CHECK_DATA, LEARNING_RATE, BATCH_SIZE, EPOCHS, NUM_WORKERS, GENERATOR_CHECKPOINT, DISCRIMINATOR_CHECKPOINT, NOISE_DIMENSION, TERMINATE_COUNT

    parser: ArgumentParser = ArgumentParser()
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda', 'mps'])
    parser.add_argument('--check_data', action='store_true')
    parser.add_argument('--learning_rate', type=float, default=1e-3)
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
    EPOCHS                   = args.epochs
    NUM_WORKERS              = args.num_workers
    GENERATOR_CHECKPOINT     = args.generator_checkpoint
    DISCRIMINATOR_CHECKPOINT = args.discriminator_checkpoint
    NOISE_DIMENSION          = args.noise_dimension
    TERMINATE_COUNT          = args.terminate_count
    
    
def load_model(path: str) -> nn.Module:
    return torch.load(path)
    
    
class EnglishWords(Dataset):
    def __init__(self) -> None:
        super().__init__()
        
        with open('data/words_alpha.txt') as f:
            self.data = f.read().split('\n')[:-1]
            
        longest: int = 0
        for word in self.data:
            longest = max(longest, len(word))
        for i in range(len(self.data)):
            self.data[i] += ' ' * (longest - len(self.data[i]))
            
    def __getitem__(self, index: int) -> torch.Tensor:
        data = torch.zeros(len(self.data[index]), len(LETTER_LABELS))
        for i, c in enumerate(self.data[index]):
            data[i][LETTER_LABELS[c]] = 1
            
        return data, 1
    
    def __len__(self) -> int:
        return len(self.data)


class Generator(nn.Module):
    @classmethod
    def generate_noise(cls, device: torch.device, batch_size: int = 1) -> torch.Tensor:
        noise_size = ((len(LETTER_REVERSE_LABELS) * 2 + 3) * 2 + 3) * 2 + 3
        return torch.randn(batch_size, 1, noise_size, device=device)
    
    @classmethod
    def generate_word(cls, x: torch.Tensor) -> torch.Tensor:
        return [''.join(''.join([LETTER_REVERSE_LABELS[torch.argmax(letter_tensor)] for letter_tensor in word_tensor]).split()) for word_tensor in x]
    
    def __init__(self) -> None:
        super().__init__()
        
        self.net: nn.Module = nn.Sequential(
            nn.Conv1d(1, 4, 4, 1, 0, bias=False),
            nn.ReLU(),
            nn.MaxPool1d(2, 2),
            nn.Conv1d(4, 16, 4, 1, 0, bias=False),
            nn.ReLU(),
            nn.MaxPool1d(2, 2),
            nn.Conv1d(16, 32, 4, 1, 0, bias=False),
            nn.ReLU(),
            nn.MaxPool1d(2, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
    
    
class Discriminator(nn.Module):
    def __init__(
        self,
        device,
        input_size: int = len(LETTER_REVERSE_LABELS),
        hidden_size: int = 32
    ) -> None:
        super().__init__()
        
        self.hidden_size = hidden_size

        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, 1)
        self.softmax = nn.LogSoftmax(dim=1)
        
        self.hidden = torch.zeros(1, self.hidden_size, device=device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        combined = torch.cat((x, self.hidden), 1)
        self.hidden = self.i2h(combined)
        output = self.i2o(combined)
        output = self.softmax(output)
        return output
    
    
class GANTrainer:
    def __init__(
        self,
        training_data: Dataset,
        device: torch.device,
        generator: nn.Module,
        discriminator: nn.Module,
        learning_rate: float,
        num_workers: int
    ) -> None:
        self.batch_size = 1
        self.device: torch.device = device
        
        self.DISCRIMINATOR_TARGETS = {
            'real': torch.ones(self.batch_size, 1, device=self.device),
            'fake': torch.zeros(self.batch_size, 1, device=self.device)
        }
        
        self.generator: nn.Module = generator.to(self.device)
        self.discriminator: nn.Module = discriminator.to(self.device)
        
        self.learning_rate: float = learning_rate
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
            input = input.to(self.device)
            generator_output: torch.Tensor = self.generator(Generator.generate_noise(self.device, self.batch_size))
            
            real_prediction: torch.Tensor
            for i in range(len(input[0])):
                real_prediction = self.discriminator(input[:, i])
            fake_prediction: torch.Tensor
            for i in range(len(generator_output.detach()[0])):
                fake_prediction = self.discriminator(generator_output.detach()[:, i])
            
            discriminator_loss: nn.Module = \
                self.loss_function(real_prediction, self.DISCRIMINATOR_TARGETS['real']) + \
                self.loss_function(fake_prediction, self.DISCRIMINATOR_TARGETS['fake'])

            self.discriminator_optimizer.zero_grad()
            discriminator_loss.backward()
            self.discriminator_optimizer.step()
            
            prediction: torch.Tensor
            for i in range(len(generator_output[0])):
                prediction = self.discriminator(generator_output[:, i])

            generator_loss: nn.Module = self.loss_function(prediction, self.DISCRIMINATOR_TARGETS['real'])
            
            self.generator_optimizer.zero_grad()
            generator_loss.backward()
            self.generator_optimizer.step()
            
            generator_loss_sum += generator_loss
            discriminator_loss_sum += discriminator_loss
            
        return generator_loss_sum / self.batch_size, discriminator_loss_sum / self.batch_size
        
    def print_output(self) -> None:
        tqdm.write(f'{Generator.generate_word(self.generator(Generator.generate_noise(8)))}')
        
    def run(self, epochs: int, terminate_count: int, save_path: str = 'models/model.pth') -> None:
        min_generator_loss: float     = float('inf')
        min_discriminator_loss: float = float('inf')
        update_count: int = 0
        for _ in tqdm(range(epochs), desc='epoch'):
            generator_loss: float
            discriminator_loss: float
            generator_loss, discriminator_loss = self.train_loop()
            
            self.print_output()
            
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


if __name__ == '__main__':
    accept_parameters()

    training_data: EnglishWords = EnglishWords()
    
    generator: Generator         = Generator()
    if GENERATOR_CHECKPOINT != None:
        generator = load_model(GENERATOR_CHECKPOINT)
    discriminator: Discriminator = Discriminator(DEVICE)
    if DISCRIMINATOR_CHECKPOINT != None:
        generator = load_model(DISCRIMINATOR_CHECKPOINT)

    trainer: GANTrainer = GANTrainer(training_data, DEVICE, generator, discriminator, LEARNING_RATE, NUM_WORKERS)
    trainer.run(EPOCHS, TERMINATE_COUNT)
