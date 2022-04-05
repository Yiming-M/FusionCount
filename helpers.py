import torch
from torch import nn, Tensor
import torch.nn.functional as F
from typing import Optional, Callable, List, Tuple


def _initialize_weights(model: nn.Module) -> nn.Module:
    for m in model.modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0, 0.01)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
    return model


class ConvNormActivation(nn.Sequential):
    """
    Configurable block used for Convolution-Normalization-Activation blocks.

    Args:
        - `n_channels` (`int`): number of channels in the input image.
        - `out_channels` (`int`): number of channels produced by the Convolution-Normalization-Activation block.
        - `kernel_size`: (`int`, optional): size of the convolving kernel.
            - Default: `3`
        - `stride` (`int`, optional): stride of the convolution.
            - Default: `1`
        - `padding` (`int`, `tuple` or `str`, optional): padding added to all four sides of the input.
            - Default: `None`, in which case it will calculated as `padding = (kernel_size - 1) // 2 * dilation`.
        - `groups` (`int`, optional): number of blocked connections from input channels to output channels.
            - Default: `1`
        - `norm_layer` (`Callable[..., torch.nn.Module]`, optional): norm layer that will be stacked on top of the convolution layer. If `None` this layer won't be used.
            - Default: `torch.nn.BatchNorm2d`.
        - `activation_layer` (`Callable[..., torch.nn.Module]`, optional): activation function which will be stacked on top of the       normalization layer (if not `None`), otherwise on top of the `conv` layer. If `None` this layer wont be used.
            - Default: `torch.nn.ReLU6`
        - `dilation` (`int`): spacing between kernel elements.
            - Default: `1`
        - `inplace` (`bool`): parameter for the activation layer, which can optionally do the operation in-place.
            - Default `True`
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: Optional[int] = None,
        dilation: int = 1,
        groups: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = nn.BatchNorm2d,
        activation_layer: Optional[Callable[..., nn.Module]] = nn.ReLU6(inplace=True)
    ) -> None:
        if padding is None:
            padding = (kernel_size - 1) // 2 * dilation
        layers = [
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                dilation=dilation,
                groups=groups,
                bias=norm_layer is None,
            )
        ]
        if norm_layer is not None:
            layers.append(norm_layer(out_channels))
        if activation_layer is not None:
            layers.append(activation_layer)
        super().__init__(*layers)
        self.out_channels = out_channels


class ChannelReducer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dilation: int = 1,
        batch_norm: bool = True
    ) -> None:
        super(ChannelReducer, self).__init__()
        if batch_norm:
            norm_layer = nn.BatchNorm2d
        else:
            norm_layer = None

        conv_1 = ConvNormActivation(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            dilation=1,
            norm_layer=norm_layer,
            activation_layer=None
        )
        conv_2 = nn.Sequential(
            ConvNormActivation(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=1,
                dilation=dilation,
                norm_layer=norm_layer,
                activation_layer=nn.ReLU(inplace=True)
            ),
            ConvNormActivation(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=1,
                dilation=dilation,
                norm_layer=norm_layer,
                activation_layer=None
            )
        )
        self.conv_1 = _initialize_weights(conv_1)
        self.conv_2 = _initialize_weights(conv_2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, feat: Tensor) -> Tensor:
        feat_1 = self.conv_1(feat)
        feat_2 = self.conv_2(feat)

        feat = feat_1 + feat_2
        feat = self.relu(feat)
        return feat


class FeatureFuser(nn.Module):
    def __init__(self, in_channels_list: List[int], batch_norm: bool = True) -> None:
        super(FeatureFuser, self).__init__()
        if batch_norm:
            norm_layer = nn.BatchNorm2d
        else:
            norm_layer = None

        for idx, c in enumerate(in_channels_list):
            if idx == 0:
                num_1 = c
            elif idx == 1:
                num_2 = c
            else:
                assert num_2 == c

        prior_conv = ConvNormActivation(
            in_channels=num_1,
            out_channels=num_2,
            kernel_size=1,
            norm_layer=norm_layer,
            activation_layer=nn.ReLU(inplace=True)
        )
        self.prior_conv = _initialize_weights(prior_conv)

        weight_net = nn.Conv2d(
            in_channels=num_2,
            out_channels=num_2,
            kernel_size=1,
        )
        self.weight_net = _initialize_weights(weight_net)

        posterior_conv = ConvNormActivation(
            in_channels=num_2 * 2,
            out_channels=num_1,
            kernel_size=1,
            norm_layer=norm_layer,
            activation_layer=nn.ReLU(inplace=True)
        )
        self.posterior_conv = _initialize_weights(posterior_conv)

    def __make_weights__(self, feat: Tensor, scaled_feat: Tensor) -> Tensor:
        return torch.sigmoid(self.weight_net(feat - scaled_feat))

    def forward(self, feats: List[Tensor]) -> Tensor:
        feat_0, feats = feats[0], feats[1:]
        feat_0 = self.prior_conv(feat_0)
        weights = [self.__make_weights__(feat_0, feat) for feat in feats]
        feats = [sum([feats[i] * weights[i] for i in range(len(weights))]) / sum(weights)] + [feat_0]
        feats = torch.cat(feats, dim=1)
        feats = self.posterior_conv(feats)
        return feats
