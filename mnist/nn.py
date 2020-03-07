from torch.nn import Module, Conv2d, MaxPool2d, Dropout2d, Linear
from torch.nn.functional import relu

class NN(Module):
    def __init__(self):
        super(NN, self).__init__()
        self.conv1   = Conv2d(1, 32, 3)
        self.conv2   = Conv2d(32, 64, 3)
        self.pool    = MaxPool2d(2, 2)
        self.dropout = Dropout2d()
        self.fc1     = Linear(12 * 12 * 64, 128)
        self.fc2     = Linear(128, 10)

    def forward(self, x):
        x = relu(self.conv1(x))
        x = self.pool(relu(self.conv2(x)))
        x = self.dropout(x)
        x = x.view(-1, 12 * 12 * 64)
        x = relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x