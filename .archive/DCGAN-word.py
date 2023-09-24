from argparse import ArgumentParser
import random

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchmetrics import HingeLoss
from tqdm import tqdm


LABELS = {
    'a':  0,
    'b':  1,
    'c':  2,
    'd':  3,
    'e':  4,
    'f':  5,
    'g':  6,
    'h':  7,
    'i':  8,
    'j':  9,
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
    ' ': 26,
}

INVERSE_LABELS = 'abcdefghijklmnopqrstuvwxyz '


def accept_parameters():
    global DEVICE, LEARNING_RATE, BATCH_SIZE, EPOCHS, NUM_WORKERS, GENERATOR_CHECKPOINT, DISCRIMINATOR_CHECKPOINT, NOISE_DIMENSION, TERMINATE_COUNT

    parser = ArgumentParser()
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda', 'mps'])
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--num_workers', type=int, default=3)
    parser.add_argument('--generator_checkpoint', type=str, default=None)
    parser.add_argument('--discriminator_checkpoint', type=str, default=None)
    parser.add_argument('--noise_dimension', type=int, default=128)
    parser.add_argument('--terminate_count', type=int, default=10)
    
    args                     = parser.parse_args()
    DEVICE                   = torch.device(args.device)
    LEARNING_RATE            = args.learning_rate
    BATCH_SIZE               = args.batch_size
    EPOCHS                   = args.epochs
    NUM_WORKERS              = args.num_workers
    GENERATOR_CHECKPOINT     = args.generator_checkpoint
    DISCRIMINATOR_CHECKPOINT = args.discriminator_checkpoint
    NOISE_DIMENSION          = args.noise_dimension
    TERMINATE_COUNT          = args.terminate_count
    
    
class EnglishWords(Dataset):
    def __init__(self, words, max_length):
        super(EnglishWords, self).__init__()
        
        self.max_length = max_length
        
        self.data = []
        for word in words:
            self.data.append([])
            for c in word:
                self.data[-1].append(LABELS[c])
            self.data[-1].extend([LABELS[' ']] * (max_length - len(self.data[-1])))
                
        self.data = torch.tensor(self.data)
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, i):
        return nn.functional.one_hot(self.data[i], num_classes=len(LABELS)).to(torch.float32)
    
    
def get_dataset(test_ratio=0.1):
    with open('data/words_alpha.txt', 'r') as f:
        words = f.read().split()
        
    random.shuffle(words)
    max_length = max([len(word) for word in words])
    
    training_data = EnglishWords(words[:int(len(words) * (1 - test_ratio))], max_length)
    test_data     = EnglishWords(words[int(len(words) * (1 - test_ratio)):], max_length)

    return training_data, test_data, max_length
    
    
def load_model(path):
    return torch.load(path)


class Generator(nn.Module):
    def __init__(
        self,
        noise_dimension,
        output_length,
        feature_map_size=64,
    ) -> None:
        super(Generator, self).__init__()
        
        self.noise_dimension = noise_dimension
        self.output_length   = output_length
        
        self.net = nn.Sequential(
            nn.Linear(noise_dimension, feature_map_size * 4),
            nn.ReLU(),
            nn.Linear(feature_map_size * 4, feature_map_size * 2),
            nn.ReLU(),
            nn.Linear(feature_map_size * 2, feature_map_size),
            nn.ReLU(),
            nn.Linear(feature_map_size, self.output_length * len(LABELS))
        )

    def forward(self, x):
        x = self.net(x)
        out = torch.zeros(len(x), self.output_length, len(LABELS))
        
        softmax = nn.Softmax(dim=0)
        for i in range(self.output_length):
            idx = torch.argmax(softmax(x[:, i * len(LABELS):(i + 1) * len(LABELS)]), dim=1)
            for j, jdx in enumerate(idx):
                out[j][i][jdx] = 1
            
        return out
    
    def generate_noise(self, batch_size):
        return torch.randn(batch_size, self.noise_dimension)
    
    @classmethod
    def decode(cls, x):
        words = []
        for word in x:
            words.append('')
            for c in word:
                words[-1] += INVERSE_LABELS[torch.argmax(c)] if sum(c) else ' '
        
        return words
    
    
class Discriminator(nn.Module):
    def __init__(
        self,
        input_length,
        feature_map_size=64
    ) -> None:
        super(Discriminator, self).__init__()
        
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_length * len(LABELS), feature_map_size),
            nn.ReLU(),
            nn.Linear(feature_map_size, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)
    
    
class GANTrainer:
    def __init__(
        self,
        training_data,
        device,
        generator,
        discriminator,
        learning_rate,
        batch_size,
        num_workers
    ):
        self.DISCRIMINATOR_TARGETS = {
            'real': torch.ones(batch_size, 1),
            'fake': torch.zeros(batch_size, 1)
        }
        
        self.device = device
        
        self.generator     = generator.to(self.device)
        self.discriminator = discriminator.to(self.device)
        
        self.learning_rate = learning_rate
        self.batch_size    = batch_size
        self.num_workers   = num_workers
        
        self.loss_function = HingeLoss()
        
        self.generator_optimizer     = torch.optim.Adam(self.generator.parameters(), lr=self.learning_rate)
        self.discriminator_optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=self.learning_rate)
        
        self.training_loader = DataLoader(
            training_data,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
        )

    def train_loop(self):
        generator_loss_sum = 0
        discriminator_loss_sum = 0
        for input in tqdm(self.training_loader, desc='training'):
            generator_output = self.generator(self.generator.generate_noise(self.batch_size))
            
            discriminator_loss = \
                self.loss_function(self.discriminator(input), self.DISCRIMINATOR_TARGETS['real']) + \
                self.loss_function(self.discriminator(generator_output.detach()), self.DISCRIMINATOR_TARGETS['fake'])

            self.discriminator_optimizer.zero_grad()
            discriminator_loss.backward()
            self.discriminator_optimizer.step()

            generator_loss = self.loss_function(self.discriminator(generator_output), self.DISCRIMINATOR_TARGETS['real'])
            
            self.generator_optimizer.zero_grad()
            generator_loss.backward()
            self.generator_optimizer.step()
            
            generator_loss_sum += generator_loss
            discriminator_loss_sum += discriminator_loss
            
        return generator_loss_sum / self.batch_size, discriminator_loss_sum / self.batch_size
    
    def save_model(self, path):
        torch.save(self.model, path)
        
    def save_output(self, epoch, samples=5):
        with open(f'outputs/GANword/{epoch:04}.txt', 'w') as f:
            f.write('\n'.join(Generator.decode(self.generator(self.generator.generate_noise(samples)))))
        
    def run(self, epochs, terminate_count, save_path='models/model.pth'):
        min_generator_loss     = float('inf')
        min_discriminator_loss = float('inf')
        update_count = 0
        for epoch in tqdm(range(epochs), desc='epoch'):
            generator_loss, discriminator_loss = self.train_loop()
            
            self.save_output(epoch)
            
            if min_generator_loss > generator_loss:
                min_generator_loss = generator_loss
                update_count = 0
            if min_discriminator_loss > discriminator_loss:
                min_discriminator_loss = discriminator_loss
                update_count = 0
            update_count += 1
            
            tqdm.write(f'generator_loss: {generator_loss:0.7f}, min_generator_loss: {min_generator_loss:>0.7f}')
            tqdm.write(f'discriminator_loss: {discriminator_loss:0.7f}, min_discriminator_loss: {min_discriminator_loss:>0.7f}\n')
            
            if update_count >= terminate_count:
                break
            
        # self.save_model(save_path)


if __name__ == '__main__':
    accept_parameters()
    
    training_data, test_data, max_length = get_dataset(test_ratio=0)
    
    generator = Generator(NOISE_DIMENSION, max_length)
    if GENERATOR_CHECKPOINT != None:
        generator = load_model(GENERATOR_CHECKPOINT)

    discriminator: Discriminator = Discriminator(max_length)
    if DISCRIMINATOR_CHECKPOINT != None:
        discriminator = load_model(DISCRIMINATOR_CHECKPOINT)

    trainer: GANTrainer = GANTrainer(training_data, DEVICE, generator, discriminator, LEARNING_RATE, BATCH_SIZE, NUM_WORKERS)
    trainer.run(EPOCHS, TERMINATE_COUNT)
