import torch.nn as nn
from torchsummary import summary
import torch


def Conv3x3BNReLU(in_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)):
    return nn.Sequential(
        nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                  padding=padding),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )


def Conv3x3BN(in_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)):
    return nn.Sequential(
        nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                  padding=padding),
        nn.BatchNorm2d(out_channels),
    )


def Conv1x1BN(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(1, 1), stride=(1, 1)),
        nn.BatchNorm2d(out_channels)
    )


def Conv1x1BNReLU(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(1, 1), stride=(1, 1)),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )


# ----------------------------------------------------------------------------------------------------------------------
class RecNetV1(nn.Module):
    """conv_3x3...."""

    def __init__(self, in_channels=3, out_channels=1, depth=1, width=24):
        super(RecNetV1, self).__init__()

        self.depth = depth
        self.width = width
        self.layer = self.make_layer(in_channels, self.width)
        self.conv3x3 = Conv3x3BN(in_channels=self.width, out_channels=out_channels)

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def make_layer(self, in_channels, out_channels):
        layers = [Conv3x3BNReLU(in_channels=in_channels, out_channels=out_channels)]
        for i in range(1, self.depth):
            layers.append(Conv3x3BNReLU(in_channels=out_channels, out_channels=out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.layer(x)
        x = self.conv3x3(x)
        return x


# ----------------------------------------------------------------------------------------------------------------------
class RecNetV2(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super(RecNetV2, self).__init__()

        self.conv1 = Conv3x3BNReLU(in_channels=in_channels, out_channels=1)
        self.conv2 = Conv3x3BNReLU(in_channels=1, out_channels=2)
        self.conv3 = Conv3x3BNReLU(in_channels=3, out_channels=3)
        self.conv4 = Conv3x3BNReLU(in_channels=6, out_channels=4)
        self.conv5 = Conv3x3BNReLU(in_channels=10, out_channels=5)
        self.conv6 = Conv3x3BNReLU(in_channels=15, out_channels=3)
        self.conv = Conv3x3BN(in_channels=3, out_channels=out_channels)

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x1 = self.conv1(x)  # 1*32*80
        x2 = self.conv2(x1)  # 2*32*80
        x3 = torch.cat([x1, x2], dim=1)  # 3*32*80
        x3 = self.conv3(x3)  # 3*32*80
        x4 = torch.cat([x1, x2, x3], dim=1)  # 6*32*80
        x4 = self.conv4(x4)
        x5 = torch.cat([x1, x2, x3, x4], dim=1)  # 10*32*80
        x5 = self.conv5(x5)
        x6 = torch.cat([x1, x2, x3, x4, x5], dim=1)
        x6 = self.conv6(x6)

        return self.conv(x6)


# ----------------------------------------------------------------------------------------------------------------------
class RecNetV3(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super(RecNetV3, self).__init__()

        self.conv1 = Conv3x3BNReLU(in_channels=in_channels, out_channels=3)
        self.conv2 = Conv3x3BNReLU(in_channels=3, out_channels=3)
        self.conv3 = Conv3x3BNReLU(in_channels=3, out_channels=3)
        self.conv4 = Conv1x1BNReLU(in_channels=3, out_channels=8)
        self.conv5 = Conv3x3BNReLU(in_channels=8, out_channels=8)
        self.conv6 = Conv3x3BNReLU(in_channels=8, out_channels=8)
        self.conv7 = Conv3x3BNReLU(in_channels=8, out_channels=8)
        self.conv8 = Conv1x1BNReLU(in_channels=8, out_channels=16)
        self.conv9 = Conv3x3BNReLU(in_channels=16, out_channels=16)
        self.conv10 = Conv3x3BNReLU(in_channels=16, out_channels=16)
        self.conv11 = Conv3x3BNReLU(in_channels=16, out_channels=16)
        self.conv = Conv1x1BN(in_channels=16, out_channels=out_channels)

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.conv8(x)
        x = self.conv9(x)
        x = self.conv10(x)
        x = self.conv11(x)
        return self.conv(x)


# ----------------------------------------------------------------------------------------------------------------------
class DoubleConvBlock(nn.Module):
    def __init__(self, in_size, out_size, padding, batch_norm):
        super(DoubleConvBlock, self).__init__()
        block = [nn.Conv2d(in_size, out_size, kernel_size=3, padding=int(padding)), nn.ReLU()]

        # block.append(nn.LeakyReLU())
        if batch_norm:
            block.append(nn.BatchNorm2d(out_size))

        block.append(nn.Conv2d(out_size, out_size, kernel_size=3, padding=int(padding), groups=out_size))
        block.append(nn.ReLU())
        # block.append(nn.LeakyReLU())
        if batch_norm:
            block.append(nn.BatchNorm2d(out_size))

        self.block = nn.Sequential(*block)

    def forward(self, x):
        out = self.block(x)
        return out


class ResBlock(nn.Module):
    def __init__(self, in_size, out_size, padding, batch_norm):
        super(ResBlock, self).__init__()

        self.up = nn.Sequential(nn.Conv2d(in_size, out_size, kernel_size=3, padding=1, stride=1))
        self.conv_block = DoubleConvBlock(in_size, out_size, padding, batch_norm)

    def forward(self, x, bridge):
        up = self.up(x)
        out = torch.cat([up, bridge], 1)
        out = self.conv_block(out)

        return out


class RecNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, depth=5, wf=2, padding=True, batch_norm=True):
        """
        Args:
            in_channels (int): number of input channels
            out_channels (int): number of output channels
            depth (int): depth of the network
            wf (int): number of filters in the first layer is 2**wf
            padding (bool): if True, apply padding such that the input shape
                            is the same as the output.
                            This may introduce artifacts
            batch_norm (bool): Use BatchNorm after layers with an
                               activation function
        """
        super(RecNet, self).__init__()
        self.padding = padding
        self.depth = depth
        prev_channels = in_channels
        self.down_path = nn.ModuleList()
        for i in range(depth):
            self.down_path.append(DoubleConvBlock(prev_channels, 2 ** (wf + i), padding, batch_norm))
            prev_channels = 2 ** (wf + i)

        self.up_path = nn.ModuleList()
        for i in reversed(range(depth - 1)):
            self.up_path.append(ResBlock(prev_channels, 2 ** (wf + i), padding, batch_norm))
            prev_channels = 2 ** (wf + i)

        self.last = nn.Conv2d(prev_channels, out_channels, kernel_size=1)

    def forward(self, x):
        blocks = []
        for i, down in enumerate(self.down_path):
            x = down(x)
            if i != len(self.down_path) - 1:
                blocks.append(x)

        for i, up in enumerate(self.up_path):
            x = up(x, blocks[-i - 1])

        return self.last(x)


if __name__ == '__main__':
    print('start')
    rec = RecNet().cuda()
    summary(rec, (3, 32, 80))
