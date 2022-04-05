import torch
from torch import nn, Tensor
from torch.hub import load_state_dict_from_url
from typing import List, Union, cast, Optional
from helpers import _initialize_weights

# This script has been adapted from the vgg.py in torchvision (https://github.com/pytorch/vision/blob/main/torchvision/models/vgg.py).

model_urls = {
    'vgg11': 'https://download.pytorch.org/models/vgg11-8a719046.pth',
    'vgg13': 'https://download.pytorch.org/models/vgg13-19584684.pth',
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
    'vgg11_bn': 'https://download.pytorch.org/models/vgg11_bn-6002323d.pth',
    'vgg13_bn': 'https://download.pytorch.org/models/vgg13_bn-abd245e5.pth',
    'vgg16_bn': 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth',
    'vgg19_bn': 'https://download.pytorch.org/models/vgg19_bn-c79401a0.pth',
}

model_cfgs = {
    'vgg16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512],
    'vgg16_bn': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512]
}


class VGG(nn.Module):
    def __init__(
        self,
        name: str = "vgg16_bn",
        pretrained: bool = True,
        start_idx: int = 0,
        end_idx: Optional[int] = None
    ) -> None:
        super(VGG, self).__init__()
        # Create the encoder.
        #
        # cfg   Receptive fields:
        # 64        3
        # 64        5
        # 'M'       6
        # 128       10
        # 128       14
        # 'M'       16
        # 256       24
        # 256       32
        # 256       40
        # 'M'       44
        # 512       60
        # 512       76
        # 512       92
        # 'M'       100
        # 512       132
        # 512       164
        # 512       192
        #
        # Check http://ziikki.com/posts/calculate-receptive-field-for-vgg-16/ for more info.

        cfg = model_cfgs[name]
        batch_norm = True if "bn" in name else False
        encoder = self.__make_layers__(
            cfg=cfg,
            in_channels=3,
            batch_norm=batch_norm,
        )
        if pretrained:
            encoder = self.__load_weights__(encoder, name)
        else:
            encoder = _initialize_weights(encoder)

        self.encoder = self.__assemble_modules__(encoder)

        assert start_idx >= 0
        self.start_idx = start_idx
        if end_idx is None:
            end_idx = len(self.encoder)
        elif end_idx < 0:
            end_idx = end_idx + len(self.encoder)
        assert end_idx <= len(self.encoder)
        self.end_idx = end_idx

    def __make_layers__(
        self,
        cfg: List[Union[int, str]],
        in_channels: int = 3,
        batch_norm: bool = True,
    ) -> nn.Module:
        layers = nn.ModuleList()
        for v in cfg:
            if v == 'M':
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            elif v == 'U':
                layers.append(nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False))
            else:
                v = cast(int, v)
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                layers.append(conv2d)
                if batch_norm:
                    layers.append(nn.BatchNorm2d(v))
                layers.append(nn.ReLU(inplace=True))
                in_channels = v
        return layers

    def __load_weights__(self, model: nn.Module, model_name: str) -> nn.Module:
        assert model_name in model_urls.keys()
        state_dict = load_state_dict_from_url(model_urls[model_name])
        state_dict_ = {}
        for k, v in state_dict.items():
            # Drop "features" in k.
            if "features" in k:
                new_k = k[9:]
                if new_k in model.state_dict().keys():
                    state_dict_[new_k] = v
        model.load_state_dict(state_dict_, strict=True)
        return model

    def __assemble_modules__(self, model: nn.Module) -> nn.Module:
        model_ = nn.ModuleList()
        counter = 0
        while counter < len(model):
            mod = model[counter]
            if isinstance(mod, nn.MaxPool2d):
                model_.append(mod)
                counter += 1
            else:
                assert isinstance(mod, nn.Conv2d)
                block = nn.ModuleList([mod])
                for i in range(counter + 1, len(model)):
                    mod = model[i]
                    if isinstance(mod, nn.BatchNorm2d):
                        block.append(mod)
                    if isinstance(mod, nn.ReLU):
                        block.append(mod)
                        break
                model_.append(nn.Sequential(*block))
                counter = i + 1
        return model_

    def forward(self, x: Tensor) -> List[Tensor]:
        feats = []
        for idx, mod in enumerate(self.encoder):
            x = mod(x)
            if self.start_idx <= idx < self.end_idx:
                feats.append(torch.clone(x))
        return feats
