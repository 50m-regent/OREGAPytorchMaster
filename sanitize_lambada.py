import json

def sanitize(path:str) -> None:
    with open(path) as f:
        raw_data = [json.loads(line)['text'] for line in f.readlines()]
    
    sentences = []
    for raw_datum in raw_data:
        raw_datum = raw_datum.splitlines()
        
        for line in raw_datum:
            sentences += line.split('.')
            
    data = []
    for sentence in sentences:
        print(sentence)
        print('----------------------------')
        
    print(len(data))

if __name__ == '__main__':
    sanitize('data/lambada/en.jsonl')