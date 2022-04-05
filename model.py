from torch import Tensor, nn
import torch.nn.functional as F
from helpers import _initialize_weights, ConvNormActivation, FeatureFuser, ChannelReducer
from vgg import VGG


class FusionCount(nn.Module):
    """
    The official PyTorch implementation of the model proposed in FusionCount: Efficient Crowd Counting via Multiscale Feature Fusion.
    """
    def __init__(self, batch_norm: bool = True) -> None:
        super(FusionCount, self).__init__()
        if batch_norm:
            self.encoder = VGG(name="vgg16_bn", pretrained=True, start_idx=2)
        else:
            self.encoder = VGG(name="vgg16", pretrained=True, start_idx=2)

        self.fuser_1 = FeatureFuser([64, 128, 128], batch_norm=batch_norm)
        self.fuser_2 = FeatureFuser([128, 256, 256, 256], batch_norm=batch_norm)
        self.fuser_3 = FeatureFuser([256, 512, 512, 512], batch_norm=batch_norm)
        self.fuser_4 = FeatureFuser([512, 512, 512, 512], batch_norm=batch_norm)

        self.reducer_1 = ChannelReducer(in_channels=64, out_channels=32, dilation=2, batch_norm=batch_norm)
        self.reducer_2 = ChannelReducer(in_channels=128, out_channels=64, dilation=2, batch_norm=batch_norm)
        self.reducer_3 = ChannelReducer(in_channels=256, out_channels=128, dilation=2, batch_norm=batch_norm)
        self.reducer_4 = ChannelReducer(in_channels=512, out_channels=256, dilation=2, batch_norm=batch_norm)

        output_layer = ConvNormActivation(
            in_channels=32,
            out_channels=1,
            kernel_size=1,
            stride=1,
            dilation=1,
            norm_layer=None,
            activation_layer=nn.ReLU(inplace=True)
        )

        self.output_layer = _initialize_weights(output_layer)

    def forward(self, x: Tensor) -> Tensor:
        feats = self.encoder(x)

        feat_1, feat_2, feat_3, feat_4 = feats[0: 3], feats[3: 7], feats[7: 11], feats[11:]

        feat_1 = self.fuser_1(feat_1)
        feat_2 = self.fuser_2(feat_2)
        feat_3 = self.fuser_3(feat_3)
        feat_4 = self.fuser_4(feat_4)

        feat_4 = self.reducer_4(feat_4)
        feat_4 = F.interpolate(feat_4, size=feat_3.shape[-2:], mode="bilinear", align_corners=False)

        feat_3 = feat_3 + feat_4
        feat_3 = self.reducer_3(feat_3)
        feat_3 = F.interpolate(feat_3, size=feat_2.shape[-2:], mode="bilinear", align_corners=False)

        feat_2 = feat_2 + feat_3
        feat_2 = self.reducer_2(feat_2)
        feat_2 = F.interpolate(feat_2, size=feat_1.shape[-2:], mode="bilinear", align_corners=False)

        feat_1 = feat_1 + feat_2
        feat_1 = self.reducer_1(feat_1)
        feat_1 = F.interpolate(feat_1, size=x.shape[-2:], mode="bilinear", align_corners=False)

        output = self.output_layer(feat_1)

        return output
