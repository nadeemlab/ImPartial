import torch
import torch.nn as nn
import torch.nn.functional as F

""" Parts of the U-Net model """
'The following is adapted from: https://github.com/milesial/Pytorch-UNet'
class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None,
                 kernel_size=3, activation = 'relu', batchnorm = True, dropout=False,p_drop = 0.5):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels

        in_ch = in_channels
        out_ch = mid_channels
        sequential_list = []
        for i in range(2):

            sequential_list.append(nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, padding=1))
            if dropout:
                sequential_list.append(nn.Dropout(p=p_drop))
            if batchnorm:
                sequential_list.append(nn.BatchNorm2d(out_ch))
            if activation == 'elu':
                sequential_list.append(nn.ELU(inplace=True))
            else:
                sequential_list.append(nn.ReLU(inplace=True))


            in_ch = out_ch + 0
            out_ch = out_channels + 0

        self.double_conv = nn.Sequential(*sequential_list)

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels, activation = 'relu',
                 batchnorm = True, dropout=False, p_drop = 0.5):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels, activation = activation,
                       batchnorm = batchnorm, dropout=dropout, p_drop = p_drop)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True, activation = 'relu',
                 batchnorm = True, dropout=False, p_drop = 0.5):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2,
                                   activation = activation, batchnorm = batchnorm, dropout=dropout,p_drop = p_drop)
        else:
            self.up = nn.ConvTranspose2d(in_channels , in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels,
                                   activation = activation, batchnorm = batchnorm, dropout=dropout,p_drop = p_drop)


    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=False, p_drop = 0.5):
        super(OutConv, self).__init__()
        if dropout:
            self.conv = []
            self.conv.append(nn.Conv2d(in_channels, out_channels, kernel_size=1))
            self.conv.append(nn.Dropout(p=p_drop))
            self.conv = nn.Sequential(*self.conv)
        else:
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

'The following UNet is adapted from: https://github.com/milesial/Pytorch-UNet'

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True, base = 32, depth=4, activation = 'relu',
                 batchnorm = True, dropout=False, dropout_lastconv=False, p_drop = 0.5):
        super(UNet, self).__init__()
        self.activation = activation
        self.batchnorm = batchnorm
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.dropout = dropout
        self.dropout_lastconv = dropout_lastconv

        self.inc = DoubleConv(n_channels, base, activation = self.activation, batchnorm = self.batchnorm)
        self.down_list = []
        base_i = base + 0
        if depth > 1:
            for i in range(depth-1):
                self.down_list.append(Down(base_i, base_i*2, activation = self.activation,
                                           batchnorm = self.batchnorm, dropout=self.dropout, p_drop = p_drop))
                base_i *= 2
        factor = 2 if bilinear else 1
        self.down_list.append(Down(base_i, base_i*2 // factor , activation = self.activation,
                                   batchnorm = self.batchnorm, dropout=self.dropout, p_drop = p_drop))
        base_i *= 2
        self.down_list = nn.ModuleList(self.down_list)

        self.up_list = []
        if depth > 1:
            for i in range(depth - 1):
                # print(i, base_i)
                self.up_list.append(Up(int(base_i) , int((base_i/2)) // factor, bilinear, activation = self.activation,
                                       batchnorm = self.batchnorm, dropout=self.dropout, p_drop = p_drop))
                base_i /= 2
        self.up_list.append(Up(int(base_i), int((base_i / 2)), bilinear,
                               activation=self.activation, batchnorm=self.batchnorm, dropout=self.dropout, p_drop = p_drop))
        self.up_list = nn.ModuleList(self.up_list)
        self.outc = OutConv(base, n_classes, dropout=self.dropout_lastconv, p_drop = p_drop)

    def forward(self, x):

        x_list = []
        x_list.append(self.inc(x))
        for down in self.down_list:
            x_list.append(down(x_list[-1]))

        x_out = x_list[-1]
        ix = -2
        for up in self.up_list:
            x_out = up(x_out,x_list[ix])
            ix -= 1

        logits = self.outc(x_out)

        return logits

    def enable_dropout(self):

        for m in self.down_list.modules():
            if m.__class__.__name__.startswith('Dropout'):
                m.train()

        for m in self.up_list.modules():
            if m.__class__.__name__.startswith('Dropout'):
                m.train()

        for m in self.outc.modules():
            if m.__class__.__name__.startswith('Dropout'):
                m.train()

    def MCdropout_forward(self,x):
        self.enable_dropout()

        return self.forward(x)


class UNetFactors(nn.Module):
    def __init__(self, n_channels, n_outputs, n_factors, n_classes, bilinear=True, base = 32, depth=4, activation = 'relu', batchnorm = True):
        super(UNetFactors, self).__init__()
        self.activation = activation
        self.batchnorm = batchnorm
        self.n_channels = n_channels
        self.n_outputs = n_outputs
        self.bilinear = bilinear
        self.n_factors = n_factors #factors has to be smaller or equal than n_outputs
        self.n_classes = n_classes

        self.unet = UNet(self.n_channels, self.n_outputs, bilinear=self.bilinear,
                         base = base, depth=depth, activation = activation, batchnorm = batchnorm)
        self.fc_mixture = nn.Linear(self.n_factors, self.n_classes)

    def forward(self, x):
        b,c,h,w = x.shape
        x_channels = self.unet(x)
        x_factors = x_channels[:,0:self.n_factors,...]
        x_factors = torch.nn.Softmax(dim=1)(x_factors)

        # x_classes = x_factors.permute(0, 2, 3, 1)
        x_classes = self.fc_mixture(x_factors.permute(0, 2, 3, 1).contiguous().view(int(b*h*w),self.n_factors))
        x_classes = x_classes.view(b,h,w,-1).permute(0, 3, 1, 2) #backk to batch x channel x h x w

        return x_classes,x_factors,x_channels[:,self.n_factors:,...]

    def get_classes(self, x):
        b, c, h, w = x.shape
        x_channels = self.unet(x)
        x_factors = x_channels[:, 0:self.n_factors, ...]
        x_factors = torch.nn.Softmax(dim=1)(x_factors)

        x_classes = x_factors.permute(0, 2, 3, 1)
        x_classes = self.fc_mixture(x_classes.contiguous().view(int(b*h*w),self.n_factors))
        x_classes = x_classes.view(b,h,w,-1).permute(0, 3, 1, 2) #backk to batch x channel x h x w

        return x_classes
