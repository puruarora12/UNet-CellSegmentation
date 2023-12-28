import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

# import any other libraries you need below this line

class twoConvBlock(nn.Module):
    def __init__(self, input, output):
        super(twoConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(input, output, kernel_size=(3, 3))
        self.conv2 = nn.Conv2d(output, output, kernel_size=(3, 3))
        self.norm = nn.BatchNorm2d(output)
        self.relu = nn.ReLU()
        # initialize the block

    def forward(self, input):
        out = self.conv1(input)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.norm(out)
        out = self.relu(out)
        return out
        # implement the forward path

class downStep(nn.Module):
    def __init__(self, input, output):
        super(downStep, self).__init__()
        self.conv = twoConvBlock(input, output)
        self.maxPooling = nn.MaxPool2d((2, 2), stride=2)
        # initialize the down path

    def forward(self, input):
        copy_out = self.conv(input)
        out = self.maxPooling(copy_out)
        return out, copy_out
        # implement the forward path

class upStep(nn.Module):
    def __init__(self, input):
        super(upStep, self).__init__()
        output = int(input/2)
        self.upSampling = nn.ConvTranspose2d(input, output, kernel_size=(2, 2),stride=(2, 2))
        self.conv = twoConvBlock(input, output)

    def forward(self, input, copy_input):
        out = self.upSampling(input)
        _, _, h, w = out.size()

        cropTrans = transforms.Compose([
            transforms.CenterCrop((h, w)),
        ])
        copy = cropTrans(copy_input)

        out = torch.cat((copy, out), dim=1)
        out = self.conv(out)
        return out
        

class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        self.down1 = downStep(1,64)
        self.down2 = downStep(64,128)
        self.down3 = downStep(128,256)
        self.down4 = downStep(256,512)
        self.conv = twoConvBlock(512,1024)
        self.up1 = upStep(1024)
        self.up2 = upStep(512)
        self.up3 = upStep(256)
        self.up4 = upStep(128)
        self.endConv = nn.Conv2d(64, 2, kernel_size=(1, 1))
        

    def forward(self, input):
        out, copy_out1 = self.down1(input)
        out, copy_out2 = self.down2(out)
        out, copy_out3 = self.down3(out)
        out, copy_out4 = self.down4(out)
        out = self.conv(out)
        out = self.up1(out, copy_out4)
        out = self.up2(out, copy_out3)
        out = self.up3(out, copy_out2)
        out = self.up4(out, copy_out1)
        out = self.endConv(out)
        return out
        