# Video: https://www.youtube.com/watch?v=552FVdcHIUU
# Blog: https://jarvislabs.ai/blogs/tgs-salt-model/
from fastai.vision import *
from fastai.vision.all import *
import torch
import timm
from timm.models.layers import create_attn, get_attn

DEFAULT_ATTENTION_TYPE = "se"


def calc_hyperfeats(d1, d2, d3, d4, d5):
    hyperfeats = torch.cat(
        (
            d1,
            F.interpolate(d2, scale_factor=2, mode="bilinear", align_corners=False),
            F.interpolate(d3, scale_factor=4, mode="bilinear", align_corners=False),
            F.interpolate(d4, scale_factor=8, mode="bilinear", align_corners=False),
            F.interpolate(d5, scale_factor=16, mode="bilinear", align_corners=False),
        ),
        1,
    )
    return hyperfeats


class Encoder(Module):
    def __init__(
        self,
        encoder_name: str = "resnext50_32x4d",
        pretrained: bool = True,
    ):
        self.encoder = timm.create_model(
            encoder_name,
            features_only=True,
            pretrained=True,
        )

    def forward(self, x):
        return self.encoder(x)


class UnetBlock(Module):
    def __init__(
        self,
        in_channels: int,
        mid_channels: int,
        out_channels: int,
        activation_class: torch.nn.modules.activation = torch.nn.Mish,
        add_attention: bool = False,
        attention_type: str = DEFAULT_ATTENTION_TYPE,
    ):
        self.conv1 = ConvLayer(in_channels, mid_channels, act_cls=activation_class)
        self.conv2 = ConvLayer(mid_channels, out_channels, act_cls=activation_class)
        if add_attention:
            self.attention_layer = create_attn(attention_type, out_channels)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=True)
        x = self.conv1(x)
        x = self.conv2(x)
        if hasattr(self, "attention_layer"):
            x = self.attention_layer(x)
        return x


class UnetDecoder(Module):
    def __init__(
        self,
        feature_scale_factor: int = 32,
        expansion_scale_factor: int = 4,
        feature_dim_list: List[int] = [64, 256, 512, 1024, 2048],
        n_out: int = 1,
        hypercol: bool = False,
        activation_class: torch.nn.modules.activation = torch.nn.Mish,
        add_attention: bool = False,
        attention_type=DEFAULT_ATTENTION_TYPE,
    ):
        num_channels = feature_dim_list[2]
        num_center_layer_channels = num_channels * expansion_scale_factor
        num_decoder5_channels = num_center_layer_channels + (feature_dim_list[1] * expansion_scale_factor)
        self.hypercol = hypercol
        self.center_layer = nn.Sequential(
            ConvLayer(num_center_layer_channels, num_center_layer_channels, act_cls=activation_class),
            ConvLayer(num_center_layer_channels, num_center_layer_channels // 2, act_cls=activation_class),
            create_attn(attention_type, num_center_layer_channels // 2),
        )

        self.decoder5 = UnetBlock(
            num_decoder5_channels,
            num_channels,
            feature_scale_factor,
            activation_class=activation_class,
            add_attention=add_attention,
        )
        self.decoder4 = UnetBlock(
            feature_dim_list[1] * expansion_scale_factor + feature_scale_factor,
            feature_dim_list[1],
            feature_scale_factor,
            activation_class=activation_class,
            # add_attention = add_attention,
        )
        self.decoder3 = UnetBlock(
            feature_dim_list[1] // 2 * expansion_scale_factor + feature_scale_factor,
            feature_dim_list[1] // 2,
            feature_scale_factor,
            activation_class=activation_class,
            # add_attention = add_attention,
        )
        self.decoder2 = UnetBlock(
            feature_dim_list[0] * expansion_scale_factor + feature_scale_factor,
            feature_dim_list[0],
            feature_scale_factor,
            activation_class=activation_class,
            # add_attention = add_attention,
        )
        self.decoder1 = UnetBlock(
            feature_scale_factor,
            feature_scale_factor,
            feature_scale_factor,
            activation_class=activation_class,
            # add_attention = add_attention,
        )
        if self.hypercol:
            self.logit_layer = nn.Sequential(
                ConvLayer(feature_scale_factor * 5, feature_scale_factor * 2, act_cls=activation_class),
                ConvLayer(feature_scale_factor * 2, feature_scale_factor, act_cls=activation_class),
                nn.Conv2d(feature_scale_factor, n_out, kernel_size=1),
            )
        else:
            self.logit_layer = nn.Sequential(
                ConvLayer(feature_scale_factor, feature_scale_factor // 2, act_cls=activation_class),
                ConvLayer(feature_scale_factor // 2, feature_scale_factor // 2, act_cls=activation_class),
                nn.Conv2d(feature_scale_factor // 2, n_out, kernel_size=1),
            )

    def forward(self, feats):
        e1, e2, e3, e4, e5 = feats  #'64 256 512 1024 2048'
        f = self.center_layer(e5)
        d5 = self.decoder5(torch.cat([f, e5], 1))
        d4 = self.decoder4(torch.cat([d5, e4], 1))
        d3 = self.decoder3(torch.cat([d4, e3], 1))
        d2 = self.decoder2(torch.cat([d3, e2], 1))
        d1 = self.decoder1(d2)

        out = calc_hyperfeats(d1, d2, d3, d4, d5) if self.hypercol else d1
        return self.logit_layer(out)


class Unet(Module):
    def __init__(
        self,
        feature_scale_factor=32,
        expansion_scale_factor=4,
        encoder_name: str = "resnext50_32x4d",
        feature_dim_list: List[int] = [64, 256, 512, 1024, 2048],
        n_out: int = 1,
        hypercol: bool = False,
        activation_class: torch.nn.modules.activation = torch.nn.Mish,
        add_attention: bool = False,
    ):
        self.encoder = Encoder(encoder_name)
        self.decoder = UnetDecoder(
            feature_scale_factor=feature_scale_factor,
            expansion_scale_factor=expansion_scale_factor,
            feature_dim_list=feature_dim_list,
            n_out=n_out,
            hypercol=hypercol,
            activation_class=activation_class,
            add_attention=add_attention,
        )

    def forward(self, x):
        feats = self.encoder(x)  #'64 256 512 1024 2048'
        out = self.decoder(feats)
        return out
