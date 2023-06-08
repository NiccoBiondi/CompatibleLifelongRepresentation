import torch
import torch.nn as nn 
import torch.nn.functional as F
from torch.nn import init

import math


class DownsampleA(nn.Module):
    def __init__(self, nIn, nOut, stride):
        super(DownsampleA, self).__init__()
        assert stride == 2
        self.avg = nn.AvgPool2d(kernel_size=1, stride=stride)

    def forward(self, x):
        x = self.avg(x)
        return torch.cat((x, x.mul(0)), 1)


class ResNetBasicblock(nn.Module):
    expansion = 1
    """
    RexNet basicblock (https://github.com/facebook/fb.resnet.torch/blob/master/models/resnet.lua)
    """

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(ResNetBasicblock, self).__init__()

        self.conv_a = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn_a = nn.BatchNorm2d(planes)

        self.conv_b = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn_b = nn.BatchNorm2d(planes)

        self.downsample = downsample
        self.featureSize = 64

    def forward(self, x):
        residual = x

        basicblock = self.conv_a(x)
        basicblock = self.bn_a(basicblock)
        basicblock = F.relu(basicblock, inplace=True)

        basicblock = self.conv_b(basicblock)
        basicblock = self.bn_b(basicblock)

        if self.downsample is not None:
            residual = self.downsample(x)

        return F.relu(residual + basicblock, inplace=True)


class CifarResNet(nn.Module):
    """
    ResNet optimized for the Cifar Dataset, as specified in
    https://arxiv.org/abs/1512.03385.pdf
    """

    def __init__(self, block, depth, num_classes, channels=3):
        """ Constructor
        Args:
          depth: number of layers.
          num_classes: number of classes
          base_width: base width
        """
        super(CifarResNet, self).__init__()

        self.featureSize = 64
        # Model type specifies number of layers for CIFAR-10 and CIFAR-100 model
        assert (depth - 2) % 6 == 0, 'depth should be one of 20, 32, 44, 56, 110'
        layer_blocks = (depth - 2) // 6

        self.num_classes = num_classes

        self.conv_1_3x3 = nn.Conv2d(channels, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn_1 = nn.BatchNorm2d(16)

        self.inplanes = 16
        self.stage_1 = self._make_layer(block, 16, layer_blocks, 1)
        self.stage_2 = self._make_layer(block, 32, layer_blocks, 2)
        self.stage_3 = self._make_layer(block, 64, layer_blocks, 2)
        self.avgpool = nn.AvgPool2d(8)
        self.out_dim = 64 * block.expansion

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                # m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.kaiming_normal(m.weight)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = DownsampleA(self.inplanes, planes * block.expansion, stride)

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x, feature=False, T=1, labels=False, scale=None, keep=None):

        x = self.conv_1_3x3(x)
        x = F.relu(self.bn_1(x), inplace=True)
        x = self.stage_1(x)
        x = self.stage_2(x)
        x = self.stage_3(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return x

    def forwardFeature(self, x):
        pass


class Incremental_ResNet(nn.Module):
    def __init__(self, starting_classes=100, feat_size=99):
        super(Incremental_ResNet, self).__init__()
        self.feat_size = feat_size
        
        self.backbone = CifarResNet(ResNetBasicblock, 32, starting_classes)
        self.out_dim = self.backbone.out_dim
        
        self.fc1 = nn.Linear(self.out_dim, self.feat_size, bias=False)
        self.fc2 = nn.Linear(self.feat_size, starting_classes, bias=False)

            
    def forward(self, x):
        x = self.backbone(x) 
        x = self.fc1(x)
        y = self.fc2(x)

        return x, y


def dsimplex(num_classes=10, device='cuda'):
    def simplex_coordinates(n, device):
        t = torch.zeros((n + 1, n), device=device)
        torch.eye(n, out=t[:-1,:], device=device)
        val = (1.0 - torch.sqrt(1.0 + torch.tensor([n], device=device))) / n
        t[-1,:].add_(val)
        t.div_(torch.linalg.matrix_norm(t, keepdims=True) + 1e-8)
        t.add_(-torch.mean(t, dim=0, keepdim=True))
        return t.cpu()

    feat_dim = num_classes - 1
    ds = simplex_coordinates(feat_dim, device)
    return ds


def ResNet32Cifar(resume_path, starting_classes=100, feat_size=99, device=0):
    model = Incremental_ResNet(starting_classes, feat_size)
    fixed_weights = dsimplex(num_classes=(feat_size + 1), device=device)
    model.fc2.weight.requires_grad = False  # set no gradient for the fixed classifier
    model.fc2.weight.copy_(fixed_weights)   # set the weights for the classifier
    
    if resume_path not in [None, '']:
        print(f"Resuming Weights from {resume_path}")
        new_pretrained_dict = torch.load(resume_path, map_location='cpu')
        model.load_state_dict(new_pretrained_dict)

    model.cuda(device)
    return model
