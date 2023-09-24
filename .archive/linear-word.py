from argparse import ArgumentParser
import random

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
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
CHARACTERS = 'abcdefghijklmnopqrstuvwxyz '


def accept_parameters():
    global DEVICE, LEARNING_RATE, BATCH_SIZE, EPOCHS, NUM_WORKERS, CHECKPOINT, TERMINATE_COUNT, WORD_LENGTH_THRESHOLD, MAX_STRING_LENGTH

    parser = ArgumentParser()
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda', 'mps'])
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--num_workers', type=int, default=3)
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--noise_dimension', type=int, default=128)
    parser.add_argument('--terminate_count', type=int, default=10)
    parser.add_argument('--word_length_threshold', type=int, default=2)
    parser.add_argument('--max_string_length', type=int, default=20)
    args = parser.parse_args()
    
    DEVICE                = torch.device(args.device)
    LEARNING_RATE         = args.learning_rate
    BATCH_SIZE            = args.batch_size
    EPOCHS                = args.epochs
    NUM_WORKERS           = args.num_workers
    CHECKPOINT            = args.checkpoint
    TERMINATE_COUNT       = args.terminate_count
    WORD_LENGTH_THRESHOLD = args.word_length_threshold
    MAX_STRING_LENGTH     = args.max_string_length
    
    
class EnglishWords(Dataset):
    def __init__(self, words, strings, max_length):
        super(EnglishWords, self).__init__()
        
        self.max_length = max_length
        
        self.data = []
        for word in tqdm(words + strings, desc='loading'):
            self.data.append(EnglishWords.to_one_hot(self.to_tensor(word)))
            
        self.data = torch.stack(self.data)
        self.label = torch.cat((torch.ones(len(words), 1), torch.zeros(len(strings), 1)), dim=0)
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, i):
        return self.data[i], self.label[i]
    
    def to_tensor(self, string):
        data = []
        for c in string:
            data.append(LABELS[c])
        data.extend([LABELS[' ']] * (self.max_length - len(data)))
        
        return torch.tensor(data)
    
    @classmethod
    def to_one_hot(cls, data):
        return nn.functional.one_hot(data, num_classes=len(LABELS)).to(torch.float32)
    
    
def generate_random_string(length=None):
    if None == length:
        length = random.randint(WORD_LENGTH_THRESHOLD + 1, MAX_STRING_LENGTH)
        
    return ''.join(random.choices(CHARACTERS, k=length))


def load_words():
    with open('data/words_alpha.txt', 'r') as f:
        return [word for word in f.read().split() if len(word) > WORD_LENGTH_THRESHOLD]
    
    
def get_dataset(test_ratio=0.1):
    words = load_words()
        
    random.shuffle(words)
    max_length = max([len(word) for word in words])
    
    train_words = words[:int(len(words) * (1 - test_ratio))]
    test_words  = words[int(len(words) * (1 - test_ratio)):]
    
    train_strings = [generate_random_string() for _ in range(len(train_words) * 3)]
    test_strings  = [generate_random_string() for _ in range(len(test_words) * 3)]
    
    train_data = EnglishWords(train_words, train_strings, max_length)
    test_data  = EnglishWords(test_words, test_strings, max_length)

    return train_data, test_data, max_length
    
    
def load_model(path):
    return torch.load(path)


def decode(string):
    word = ''
    for c in string:
        word += CHARACTERS[torch.argmax(c)] if CHARACTERS[torch.argmax(c)] != ' ' else ''
    
    return word
    
    
class Model(nn.Module):
    def __init__(
        self,
        input_length,
        feature_map_size=64
    ) -> None:
        super(Model, self).__init__()
        
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_length * len(LABELS), feature_map_size),
            nn.ReLU(),
            nn.Linear(feature_map_size, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)
    
    
class Trainer:
    def __init__(
        self,
        train_data,
        test_data,
        device,
        model,
        learning_rate,
        batch_size,
        num_workers
    ):
        self.device = device

        self.model = model.to(self.device)
        
        self.learning_rate = learning_rate
        self.batch_size    = batch_size
        self.num_workers   = num_workers
        
        self.loss_function = nn.BCELoss()
        self.optimizer     = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        self.train_loader = DataLoader(
            train_data,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
        )
        self.test_loader = DataLoader(
            test_data,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
        )

    def train_loop(self):
        for input, label in tqdm(self.train_loader, desc='training'):     
            input = input.to(self.device)
            label = label.to(self.device)
            
            prediction = self.model(input)
            loss       = self.loss_function(prediction, label)
            
            self.optimizer.zero_grad()
            loss.backward()
            
            self.optimizer.step()

    def test_loop(self):
        size = len(self.test_loader.dataset)
        loss = 0

        with torch.no_grad():
            for i, (input, label) in enumerate(self.test_loader):
                input = input.to(self.device)
                label = label.to(self.device)
                
                prediction = self.model(input)
                loss += self.loss_function(prediction, label).item() / size
                
                if i == 0:
                    for input1, label1, prediction1 in zip(input, label, prediction):
                        tqdm.write(f'{decode(input1):<20}, label: {label1[0]}, prediction: {prediction1[0]:>0.4f}')
                
            tqdm.write(f'loss: {loss:>0.7f}, ', end='')
            
        return loss
    
    def save_model(self, path):
        torch.save(self.model, path)
        
    def run(self, epochs, terminate_count, save_path='models/model.pth'):
        min_loss = float('inf')
        update_count = 0
        for _ in tqdm(range(epochs), desc='epoch'):
            self.train_loop()
            loss = self.test_loop()
            
            if min_loss > loss:
                min_loss = loss
                update_count = 0
                self.save_model(save_path)
                
            update_count += 1
            
            tqdm.write(f'min_loss: {min_loss:>0.7f}\n')
            
            if update_count >= terminate_count:
                break


if __name__ == '__main__':
    accept_parameters()
    
    train_data, test_data, max_length = get_dataset(test_ratio=0.1)
    
    model = Model(max_length)
    if CHECKPOINT != None:
        model = load_model(CHECKPOINT)

    trainer = Trainer(train_data, test_data, DEVICE, model, LEARNING_RATE, BATCH_SIZE, NUM_WORKERS)
    trainer.run(EPOCHS, TERMINATE_COUNT)
    
    i = 0
    while True:
        string = generate_random_string()
        likelihood = model(torch.stack([EnglishWords.to_one_hot(train_data.to_tensor(string))])).item()
        
        if likelihood > 0.95:
            print(f'{string:<20}, score: {likelihood:>0.4f}')
            i += 1
            if i > 100:
                break
