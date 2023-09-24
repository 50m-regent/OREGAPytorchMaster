def load_words() -> list[str]:
    with open('data/words_alpha.txt', 'r') as f:
        words:list[str] = f.read().split()
        
    print(words[:100])
    for i, word in enumerate(words):
        if i < 100:
            print(word, len(word))

if __name__ == '__main__':
    load_words()