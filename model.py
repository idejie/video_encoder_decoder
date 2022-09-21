import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

# encoder model 3
class encoder(nn.Module):
    def __init__(self, in_channels=3, ):
        super(encoder, self).__init__()
        # self.downscale = modules.HaarDownsampling(3)

        self.conv_1st = nn.Conv2d(in_channels, 12, 3, stride=2, padding=1, groups=1, bias=True)
        self.conv_2nd = nn.Conv2d(12, 16, 3, stride=2, padding=1, groups=1, bias=True)
        self.subnet1 = nn.Conv2d(16, 16, 3, stride=1, padding=1, groups=1, bias=True)
        self.conv_last = nn.Conv2d(16, 3, 3, stride=1, padding=1, groups=1, bias=True)
        self.act = nn.ReLU(inplace=True)

        # self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv_1st(x)
        x = self.act(self.conv_2nd(x))
        x = self.act(self.subnet1(x))

        x = self.act(self.conv_last(x))
        return torch.clamp(x, 0., 1.)


# decoder model 1
class decoder(nn.Module):
    def __init__(self, mode):
        super(decoder, self).__init__()
        self.mode = mode
        self.sub_mean = MeanShift((0.4488, 0.4371, 0.4040), sub=True)
        self.add_mean = MeanShift((0.4488, 0.4371, 0.4040), sub=False)
        self.conv1 = nn.Conv2d(3, 64, 3, stride=1, padding=1)
        # self.conv2 = nn.Conv2d(64, 64, 3, stride=1, padding=1)
        self.body = nn.Sequential(Resblock(64),
                                  Resblock(64),
                                  Resblock(64), )
        # PixelShuffle Upsample
        self.upscale = nn.Sequential(nn.Conv2d(64, 256, 3, stride=1, padding=1),
                                     nn.ReLU(),
                                     nn.PixelShuffle(upscale_factor=2),
                                     nn.Conv2d(64, 256, 3, stride=1, padding=1),
                                     nn.ReLU(),
                                     nn.PixelShuffle(upscale_factor=2),
                                     nn.Conv2d(64, 3, 3, stride=1, padding=1), )

        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.sub_mean(x)
        x = self.conv1(x)
        # x = self.act(x)
        residual = x
        x = self.body(x)
        # x = self.conv2(x)
        # x = self.act(x)
        x = torch.add(x, residual)
        x = self.upscale(x)
        x = self.add_mean(x)
        if self.mode == 'train':
            return x
        else:
            # print(x.min(), x.max())
            return torch.clamp(x, 0., 1.)

class MeanShift(nn.Module):
    def __init__(self, mean_rgb, sub):
        super(MeanShift, self).__init__()

        sign = -1 if sub else 1
        r = mean_rgb[0] * sign
        g = mean_rgb[1] * sign
        b = mean_rgb[2] * sign

        self.shifter = nn.Conv2d(3, 3, 1, 1, 0)
        self.shifter.weight.data = torch.eye(3).view(3, 3, 1, 1)
        self.shifter.bias.data = torch.Tensor([r, g, b])

        # Freeze the mean shift layer
        for params in self.shifter.parameters():
            params.requires_grad = False

class Resblock(nn.Module):
    def __init__(self, channels):
        super(Resblock, self).__init__()
        self.channels = channels
        self.block = nn.Sequential(nn.Conv2d(channels, channels, 3, stride=1, padding=1, bias=True),
                                   nn.ReLU(),
                                   nn.Conv2d(channels, channels, 3, stride=1, padding=1, bias=True))

    def forward(self, x):
        return x + self.block(x)

    def forward(self, x):
        x = self.shifter(x)
        return x
