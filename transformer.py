import math
from collections import Counter
import torch
from torch import nn, optim, Tensor
from torch.utils.data import DataLoader, Dataset
from torchtext.datasets import Multi30k
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import vocab
from tqdm import tqdm

class PositionalEncoding(nn.Module):
    def __init__(
        self,
        d_model:int,
        max_len:int,
        device:str,
        dropout:float=0.1
    ):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        
        self.encoding = torch.zeros(max_len, 1, d_model).to(device)
        self.encoding[:, 0, 0::2] = torch.sin(position * div_term)
        self.encoding[:, 0, 1::2] = torch.cos(position * div_term)

    def forward(self, x:Tensor) -> Tensor:
        x = x + self.encoding[:x.size(0)]
        return self.dropout(x)

class Discriminator(nn.Module):
    def __init__(
        self,
        num_embedding:int,
        embedding_dim:int,
        max_len:int,
        num_head:int,
        num_encoder_layer:int,
        device:str,
        hidden_dim:int=256
    ):
        super().__init__()
        
        self.embedding = nn.Embedding(num_embedding, embedding_dim)
        self.transformer = nn.Sequential(
            PositionalEncoding(embedding_dim, max_len, device),
            
            nn.TransformerEncoder(
                nn.TransformerEncoderLayer(embedding_dim, num_head),
                num_encoder_layer
            )
        )
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim // 4),
            nn.MaxPool1d(4),
            nn.ReLU(),
            
            nn.Flatten(),
            nn.Linear(max_len * embedding_dim // 16, hidden_dim),
            nn.ReLU(),
            
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x:Tensor) -> Tensor:
        x = self.embedding(x.transpose(0, 1))
        x = self.transformer(x)
        return self.mlp(x.transpose(0, 1))
    
class Generator(nn.Module):
    def __init__(
        self,
        in_num_embedding:int,
        out_num_embedding:int,
        embedding_dim:int,
        max_len:int,
        num_head:int,
        num_encoder_layer:int,
        device:str,
        hidden_dim:int=1024
    ):
        super().__init__()
        
        self.transformer = nn.Sequential(
            nn.Embedding(in_num_embedding, embedding_dim),
            PositionalEncoding(embedding_dim, max_len, device),
            
            nn.TransformerEncoder(
                nn.TransformerEncoderLayer(embedding_dim, num_head),
                num_encoder_layer
            )
        )
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            
            nn.Linear(hidden_dim, out_num_embedding),
            nn.Softmax(dim=2)
        )

    def forward(self, x:Tensor) -> Tensor:
        x = self.transformer(x.transpose(0, 1))
        return self.mlp(x.transpose(0, 1)).argmax(dim=2)
   
class Trainer:
    def __init__(
        self,
        a_G:Generator, a_D:Discriminator,
        b_G:Generator, b_D:Discriminator,
        a_loader:DataLoader,
        b_loader:DataLoader,
        device
    ):
        self.a_G = a_G.to(device)
        self.a_D = a_D.to(device)
        self.b_G = b_G.to(device)
        self.b_D = b_D.to(device)
        
        self.a_loader = a_loader
        self.b_loader = b_loader
        
        self.device = device
        
        self.adversial_criterion   = nn.BCELoss()
        self.consistency_criterion = nn.MSELoss()
        
        self.a_G_optimizer = optim.Adam(self.a_G.parameters())
        self.a_D_optimizer = optim.Adam(self.a_D.parameters())
        self.b_G_optimizer = optim.Adam(self.b_G.parameters())
        self.b_D_optimizer = optim.Adam(self.b_D.parameters())
        
        self.true_y = torch.ones(a_loader.batch_size, 1).to(device)
        self.fake_y = torch.zeros(a_loader.batch_size, 1).to(device)
        
    def step(self) -> tuple[tuple[float], tuple[float], float]:
        self.a_G.train()
        self.a_D.train()
        self.b_G.train()
        self.b_D.train()
        
        a_G_loss_sum = 0
        a_D_loss_sum = 0
        b_G_loss_sum = 0
        b_D_loss_sum = 0
        consistency_loss_sum = 0
        for a in tqdm(self.a_loader, desc='Training A'):
            true_a = a.to(self.device)
            fake_b = self.b_G(true_a)
            
            true_a_pred = self.a_D(true_a)
            fake_b_pred = self.b_D(fake_b.detach())
            
            a_D_loss = self.adversial_criterion(true_a_pred, self.true_y)
            b_D_loss = self.adversial_criterion(fake_b_pred, self.fake_y)
            a_D_loss_sum += a_D_loss.item()
            b_D_loss_sum += b_D_loss.item()
            
            self.a_D_optimizer.zero_grad()
            self.b_D_optimizer.zero_grad()
            a_D_loss.backward()
            b_D_loss.backward()
            self.a_D_optimizer.step()
            self.b_D_optimizer.step()
                
            fake_b_pred = self.b_D(fake_b)
            
            b_G_loss = self.adversial_criterion(fake_b_pred, self.true_y)
            b_G_loss_sum += b_G_loss.item()
            
            self.b_G_optimizer.zero_grad()
            b_G_loss.backward()
            
            fake_b  = self.b_G(true_a)
            cycle_a = self.a_G(fake_b)
            
            consistency_loss = 10 * self.consistency_criterion(
                self.a_D.embedding(cycle_a), self.a_D.embedding(true_a)
            )
            consistency_loss_sum += consistency_loss.item()
            
            self.a_G_optimizer.zero_grad()
            consistency_loss.backward()
            self.a_G_optimizer.step()
            self.b_G_optimizer.step()
            
        for b in tqdm(self.b_loader, desc='Training B'):
            true_b = b.to(self.device)
            fake_a = self.a_G(true_b)
            
            true_b_pred = self.b_D(true_b)
            fake_a_pred = self.a_D(fake_a.detach())
            
            a_D_loss = self.adversial_criterion(fake_a_pred, self.fake_y)
            b_D_loss = self.adversial_criterion(true_b_pred, self.true_y)
            a_D_loss_sum += a_D_loss.item()
            b_D_loss_sum += b_D_loss.item()
            
            self.a_D_optimizer.zero_grad()
            self.b_D_optimizer.zero_grad()
            a_D_loss.backward()
            b_D_loss.backward()
            self.a_D_optimizer.step()
            self.b_D_optimizer.step()
            
            fake_a_pred = self.a_D(fake_a)
            
            a_G_loss = self.adversial_criterion(fake_a_pred, self.true_y)
            a_G_loss_sum += a_G_loss.item()
            
            self.a_G_optimizer.zero_grad()
            a_G_loss.backward()
            
            fake_a  = self.a_G(true_b)
            cycle_b = self.b_G(fake_a)
            
            consistency_loss = 10 * self.consistency_criterion(
                self.b_D.embedding(cycle_b), self.b_D.embedding(true_b)
            )
            consistency_loss_sum += consistency_loss.item()
            
            self.b_G_optimizer.zero_grad()
            consistency_loss.backward()
            self.a_G_optimizer.step()
            self.b_G_optimizer.step()

        a_batch_count = len(self.a_loader)
        b_batch_count = len(self.b_loader)
        return \
            (a_G_loss_sum / a_batch_count, a_D_loss_sum / a_batch_count), \
            (b_G_loss_sum / b_batch_count, b_D_loss_sum / b_batch_count), \
            consistency_loss / (a_batch_count + b_batch_count)
        
    def run(self, num_epoch:int=100):
        for epoch in range(1, num_epoch + 1):
            (a_G_loss, a_D_loss), (b_G_loss, b_D_loss), consistency_loss = self.step()
            
            tqdm.write(
                f'Epoch {epoch:>3}: ' \
                f'a_loss = ({a_G_loss:>.7f}, {a_D_loss:>.7f}), ' \
                f'b_loss = ({b_G_loss:>.7f}, {b_D_loss:>.7f}), ' \
                f'consistency_loss = {consistency_loss:>.7f}'
            )
 
class SanitizedMulti30k(Dataset):
    def __init__(self, data:Tensor, max_length:int):
        self.data       = data
        self.max_length = max_length
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, i:int):
        return torch.cat((
            torch.as_tensor(self.data[i], dtype=torch.int64),
            torch.zeros(self.max_length - len(self.data[i]), dtype=torch.int64)
        ))

def get_data() -> Tensor:
    dataset = Multi30k('data', split='train', language_pair=('en', 'de'))
    loader  = DataLoader(dataset)
    
    en_tokenizer = get_tokenizer('spacy', language='en_core_web_sm')
    de_tokenizer = get_tokenizer('spacy', language='de_core_news_sm')
    
    en_counter = Counter()
    de_counter = Counter()
    
    for data in loader:
        en_counter.update(en_tokenizer(data[0][0]))
        de_counter.update(de_tokenizer(data[1][0]))
        
    en_vocab = vocab(en_counter, specials=['<padding>'])
    de_vocab = vocab(de_counter, specials=['<padding>'])
    
    en_data = []
    de_data = []
    for data in loader:
        en_tokens = en_tokenizer(data[0][0])
        de_tokens = de_tokenizer(data[1][0])
        
        if 0 == len(en_tokens) or 0 == len(de_tokens):
            continue
        
        en_data.append(en_vocab.lookup_indices(en_tokens))
        de_data.append(de_vocab.lookup_indices(de_tokens))

    return en_data, de_data

if __name__ == "__main__":
    DATA_LENGTH       = 64
    BATCH_SIZE        = 256
    EMBEDDING_DIM     = 128
    NUM_HEAD          = 8
    NUM_ENCODER_LAYER = 3
    DEVICE            = 'mps'
    
    en_data, de_data = get_data()
    en_loader = DataLoader(
        SanitizedMulti30k(en_data, DATA_LENGTH),
        BATCH_SIZE,
        shuffle=True,
        drop_last=True
    )
    de_loader = DataLoader(
        SanitizedMulti30k(de_data, DATA_LENGTH),
        BATCH_SIZE,
        shuffle=True,
        drop_last=True
    )
    
    en_num_embedding = max(list(map(lambda x: max(x), en_data))) + 1
    de_num_embedding = max(list(map(lambda x: max(x), de_data))) + 1
    en_D = Discriminator(
        en_num_embedding,
        EMBEDDING_DIM,
        DATA_LENGTH,
        NUM_HEAD,
        NUM_ENCODER_LAYER,
        DEVICE
    )
    de_D = Discriminator(
        de_num_embedding,
        EMBEDDING_DIM,
        DATA_LENGTH,
        NUM_HEAD,
        NUM_ENCODER_LAYER,
        DEVICE
    )
    en_G = Generator(
        de_num_embedding,
        en_num_embedding,
        EMBEDDING_DIM,
        DATA_LENGTH,
        NUM_HEAD,
        NUM_ENCODER_LAYER,
        DEVICE
    )
    de_G = Generator(
        en_num_embedding,
        de_num_embedding,
        EMBEDDING_DIM,
        DATA_LENGTH,
        NUM_HEAD,
        NUM_ENCODER_LAYER,
        DEVICE
    )
    
    trainer = Trainer(
        en_G, en_D,
        de_G, de_D,
        en_loader, de_loader,
        DEVICE
    )
    trainer.run()
    