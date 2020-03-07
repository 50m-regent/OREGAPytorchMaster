from torch.nn import CrossEntropyLoss
from torch.optim import SGD
from torch import no_grad
import torch

from nn import NN
from data import get_data

EPOCHS = 10

if __name__ == '__main__':
    nn = NN()

    criterion = CrossEntropyLoss()
    optimizer = SGD(nn.parameters(), lr = 0.001, momentum=0.9)

    train_loader, test_loader, classes = get_data()

    for epoch in range(EPOCHS):
        running_loss = 0.0

        for i, (inputs, labels) in enumerate(train_loader):
            optimizer.zero_grad()

            outputs = nn(inputs)
            loss    = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            
            if (i + 1) % 100 == 0:
                print('[{:d}, {:5d}] loss: {:.3f}'.format(
                    epoch + 1,
                    i + 1,
                    running_loss / 100))
                running_loss = 0

    print('Traning Finished!!')

    correct = 0
    total   = 0

    with no_grad():
        for (images, labels) in test_loader:
            outputs = nn(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    print('Accuracy: {}'.format(correct / total))