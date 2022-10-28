import torch.nn.functional as F
import torchvision.models as imgnet_modelzoo
from torch import nn

_AFFINE = True


def weights_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, 0, 0.01)
        nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm2d):
        if m.weight is not None:
            m.weight.data.fill_(1.0)
        if m.bias is not None:
            m.bias.data.zero_()


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, affine=_AFFINE)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, affine=_AFFINE)

        self.downsample = None
        self.bn3 = None
        if stride != 1 or in_planes != planes:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False))
            self.bn3 = nn.BatchNorm2d(self.expansion * planes, affine=_AFFINE)

    def forward(self, x):
        # x: batch_size * in_c * h * w
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            residual = self.bn3(self.downsample(x))
        out += residual
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, usr_config=None, depth=56, num_classes=10):
        super(ResNet, self).__init__()

        if usr_config is not None:
            self.model_config = usr_config.module
            self.optimizer_config = usr_config.optimizer
            self.lr_scheduler_config = usr_config.lr_scheduler

            block = BasicBlock
            depth = usr_config.module.init_args.depth
            num_classes = usr_config.module.init_args.num_classes
        else:
            depth = depth
            num_classes = num_classes
            self.model_config = None
            self.optimizer_config = None
            self.lr_scheduler_config = None
            block = BasicBlock

        num_blocks = [(depth - 2) // 6] * 3
        _outputs = [32, 64, 128]

        self.in_planes = _outputs[0]
        self.conv1 = nn.Conv2d(3, _outputs[0], kernel_size=3, stride=1, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(_outputs[0], affine=_AFFINE)
        self.layer1 = self._make_layer(block, _outputs[0], num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, _outputs[1], num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, _outputs[2], num_blocks[2], stride=2)
        self.linear = nn.Linear(_outputs[2], num_classes)
        self.apply(weights_init)

    def forward(self, x):
        out = F.relu(self.bn(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)


class ResNetImgNet(nn.Module):
    def __init__(self, usr_config=None, model_cls_str='resnet50', pretrained=False, *args, **kwargs):
        super(ResNetImgNet, self).__init__()

        if usr_config is not None:
            model_cls = getattr(imgnet_modelzoo, usr_config.module.init_args.type)
            self.model = model_cls(**usr_config.module.init_args.__dict__)
        else:
            model_cls = getattr(imgnet_modelzoo, model_cls_str)
            self.model = model_cls(pretrained=pretrained, *args, **kwargs)

    def forward(self, x):
        return self.model(x)
