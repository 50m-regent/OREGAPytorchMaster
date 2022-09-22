import torch
from torch import nn
import cv2

class Generator(nn.Module):
    def __init__(
        self,
        noise_dimension: int,
        color_channel: int,
        feature_map_size: int = 64
    ) -> None:
        super(Generator, self).__init__()
        
        self.net: nn.Module = nn.Sequential(
            nn.ConvTranspose2d(noise_dimension, feature_map_size * 4, 4, 1, 0, bias=False),
            nn.BatchNorm2d(feature_map_size * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(feature_map_size * 4, feature_map_size * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_map_size * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(feature_map_size * 2, feature_map_size, 3, 2, 1, bias=False),
            nn.BatchNorm2d(feature_map_size),
            nn.ReLU(True),
            nn.ConvTranspose2d(feature_map_size, color_channel, 2, 2, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

cv2.imwrite('test.png', Generator(128, 1)(torch.randn(64, 128, 1, 1)).cpu().detach().numpy()[0][0] * 256)
