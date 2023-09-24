import itertools
import math
import random

import numpy as np


CHARACTERS:str = 'abcdefghijklmnopqrstuvwxyz'
WORD_LENGTH_THRESHOLD:int = 2
MAX_STRING_LENGTH:int = 20


def load_words() -> list[str]:
    with open('data/words_alpha.txt', 'r') as f:
        return [word for word in f.read().split() if len(word) > WORD_LENGTH_THRESHOLD]


class WordEngine:
    def __init__(self, words:list[str], max_block_size:int=3):
        self.words = words
        
        self.block_frequency:list[dict[str, float]] = [[]]
        for i in range(1, max_block_size + 1):
            self.block_frequency.append(self.calculate_frequency(i))
        
    def calculate_frequency(self, block_size:int) -> dict[str, float]:
        frequency:dict[str, float] = {}
        for block in itertools.product(CHARACTERS, repeat=block_size):
            frequency[''.join(block)] = 0
        
        block_count:int = 0
        for word in self.words:
            for i in range(len(word) - block_size + 1):
                block:str = word[i:i + block_size]
                frequency[block] += 1
                
                block_count += 1
                
        for block in frequency.keys():
            frequency[block] = math.log(frequency[block] / block_count + 1e-10)
            
        frequency_average = np.average(list(frequency.values()))
        frequency_sd = np.std(list(frequency.values()))
        for block in frequency.keys():
            frequency[block] = (frequency[block] - frequency_average) / frequency_sd
            
        return frequency

    def calculate_likelihood(self, word:str) -> float:
        likelihood:float = 0
        block_count:int = 0
        
        for block_size in range(1, len(self.block_frequency)):
            for i in range(len(word) - block_size + 1):
                likelihood += self.block_frequency[block_size][word[i:i + block_size]]
                block_count += 1
            
        return likelihood / block_count
    

def generate_random_string(length:int=None) -> str:
    if None == length:
        length = random.randint(WORD_LENGTH_THRESHOLD + 1, MAX_STRING_LENGTH)
        
    return ''.join(random.choices(CHARACTERS, k=length))


if __name__ == '__main__':
    words:list[str] = load_words()
    engine = WordEngine(words, max_block_size=4)
    
    likelihoods:dict[str, float] = {}
    for word in words:
        likelihoods[word] = engine.calculate_likelihood(word)
        
    likelihood_rankings:list[tuple[str, float]] = list(reversed(sorted(likelihoods.items(), key=lambda x:x[1])))
    print(likelihood_rankings[:100])
    
    for i in range(1000):
        string = generate_random_string()
        likelihood = engine.calculate_likelihood(string)
        
        if likelihood > 1:
            print(string, likelihood)
    