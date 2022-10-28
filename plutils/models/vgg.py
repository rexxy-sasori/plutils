import torch.nn.init as init

from torch import nn


def _weights_init(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif isinstance(m, nn.BatchNorm2d):
        if m.weight is not None:
            m.weight.data.fill_(1.0)
            m.bias.data.zero_()


_AFFINE = True

defaultcfg = {
    11: [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512],
    13: [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512],
    16: [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512],
    19: [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512],
}


class VGG(nn.Module):
    def __init__(
            self, depth=19, num_classes=10, num_channels=3, use_bn=False, init_weights=False,
            usr_config=None, cfg=None
    ):
        super(VGG, self).__init__()
        if usr_config is not None:
            self.model_config = usr_config.module
            self.optimizer_config = usr_config.optimizer
            self.lr_scheduler_config = usr_config.lr_scheduler

            depth = self.model_config.init_args.depth
            num_classes = self.model_config.init_args.num_classes
            num_channels = self.model_config.init_args.num_channels
            use_bn = self.model_config.init_args.use_bn
            init_weights = self.model_config.init_args.init_weights

        if cfg is None:
            cfg = defaultcfg[depth]

        self.num_channels = num_channels
        self.features = self._make_layers(cfg, use_bn)
        self.classifier = nn.Linear(cfg[-1], num_classes)
        if init_weights:
            self.apply(_weights_init)

    def _make_layers(self, cfg, batch_norm=False):
        layers = []
        in_channels = self.num_channels
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1, bias=False)
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(v, affine=_AFFINE), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.features(x)
        x = nn.AvgPool2d(x.size(3))(x)
        x = x.view(x.size(0), -1)
        y = self.classifier(x)
        return y
