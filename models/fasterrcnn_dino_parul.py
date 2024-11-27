import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.ops import MultiScaleRoIAlign
from torch.hub import load
import math

# -------------------------------------------------------------
# Utility Functions and Classes
# -------------------------------------------------------------

class Backbone(nn.Module):
    def output_shape(self):
        return {name: torch.Size([self.out_channels, None, None]) for name in self._out_features}

class PatchEmbed(nn.Module):
    def __init__(self, kernel_size, stride, in_chans, embed_dim):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=kernel_size, stride=stride)

    def forward(self, x):
        x = self.proj(x)
        return x

class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio, qkv_bias, drop_path, norm_layer, act_layer, 
                 use_rel_pos, rel_pos_zero_init, window_size, use_residual_block, input_size):
        super().__init__()
        # Simplified implementation. Real implementation should include Transformer layers.

    def forward(self, x):
        return x

class LastLevelMaxPool(nn.Module):
    def forward(self, x):
        return [F.max_pool2d(x, kernel_size=1, stride=2)]

def get_abs_pos(pos_embed, use_cls_token, shape):
    return pos_embed[:, 1:] if use_cls_token else pos_embed

def get_norm(norm, out_channels):
    if norm == "LN":
        return nn.LayerNorm(out_channels)
    else:
        return nn.Identity()

def _assert_strides_are_log2_contiguous(strides):
    for i in range(1, len(strides)):
        assert strides[i] == strides[i - 1] * 2, f"Strides {strides} are not log2 contiguous"

# -------------------------------------------------------------
# Vision Transformer (ViT) Backbone
# -------------------------------------------------------------

class ViT(Backbone):
    def __init__(self, img_size=1024, patch_size=16, in_chans=3, embed_dim=768, depth=12, num_heads=12,
                 mlp_ratio=4.0, qkv_bias=True, drop_path_rate=0.0, norm_layer=nn.LayerNorm, act_layer=nn.GELU,
                 use_abs_pos=True, use_rel_pos=False, rel_pos_zero_init=True, window_size=0,
                 window_block_indexes=(), residual_block_indexes=(), use_act_checkpoint=False,
                 pretrain_img_size=224, pretrain_use_cls_token=True, out_feature="last_feat"):
        super().__init__()
        self.pretrain_use_cls_token = pretrain_use_cls_token

        self.patch_embed = PatchEmbed(
            kernel_size=(patch_size, patch_size),
            stride=(patch_size, patch_size),
            in_chans=in_chans,
            embed_dim=embed_dim,
        )

        if use_abs_pos:
            num_patches = (pretrain_img_size // patch_size) * (pretrain_img_size // patch_size)
            num_positions = (num_patches + 1) if pretrain_use_cls_token else num_patches
            self.pos_embed = nn.Parameter(torch.zeros(1, num_positions, embed_dim))
        else:
            self.pos_embed = None

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]

        self.blocks = nn.ModuleList()
        for i in range(depth):
            block = Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, 
                drop_path=dpr[i], norm_layer=norm_layer, act_layer=act_layer, 
                use_rel_pos=use_rel_pos, rel_pos_zero_init=rel_pos_zero_init, 
                window_size=window_size if i in window_block_indexes else 0, 
                use_residual_block=i in residual_block_indexes,
                input_size=(img_size // patch_size, img_size // patch_size)
            )
            self.blocks.append(block)

        self._out_feature_channels = {out_feature: embed_dim}
        self._out_feature_strides = {out_feature: patch_size}
        self._out_features = [out_feature]

        if self.pos_embed is not None:
            nn.init.trunc_normal_(self.pos_embed, std=0.02)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        x = self.patch_embed(x)
        if self.pos_embed is not None:
            x = x + get_abs_pos(self.pos_embed, self.pretrain_use_cls_token, (x.shape[1], x.shape[2]))

        for blk in self.blocks:
            x = blk(x)

        outputs = {self._out_features[0]: x.permute(0, 3, 1, 2)}
        return outputs

# -------------------------------------------------------------
# Simple Feature Pyramid
# -------------------------------------------------------------

class SimpleFeaturePyramid(Backbone):
    def __init__(self, net, in_feature, out_channels, scale_factors, top_block=None, norm="LN", square_pad=0):
        super(SimpleFeaturePyramid, self).__init__()
        assert isinstance(net, Backbone)

        self.scale_factors = scale_factors

        input_shapes = net.output_shape()
        strides = [int(input_shapes[in_feature].stride / scale) for scale in scale_factors]
        _assert_strides_are_log2_contiguous(strides)

        dim = input_shapes[in_feature].channels
        self.stages = []
        use_bias = norm == ""
        for idx, scale in enumerate(scale_factors):
            out_dim = dim
            if scale == 4.0:
                layers = [
                    nn.ConvTranspose2d(dim, dim // 2, kernel_size=2, stride=2),
                    get_norm(norm, dim // 2),
                    nn.GELU(),
                    nn.ConvTranspose2d(dim // 2, dim // 4, kernel_size=2, stride=2),
                ]
                out_dim = dim // 4
            elif scale == 2.0:
                layers = [nn.ConvTranspose2d(dim, dim // 2, kernel_size=2, stride=2)]
                out_dim = dim // 2
            elif scale == 1.0:
                layers = []
            elif scale == 0.5:
                layers = [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                raise NotImplementedError(f"scale_factor={scale} is not supported yet.")

            layers.extend(
                [
                    nn.Conv2d(
                        out_dim,
                        out_channels,
                        kernel_size=1,
                        bias=use_bias,
                        norm=get_norm(norm, out_channels),
                    ),
                    nn.Conv2d(
                        out_channels,
                        out_channels,
                        kernel_size=3,
                        padding=1,
                        bias=use_bias,
                        norm=get_norm(norm, out_channels),
                    ),
                ]
            )
            layers = nn.Sequential(*layers)

            stage = int(math.log2(strides[idx]))
            self.add_module(f"simfp_{stage}", layers)
            self.stages.append(layers)

        self.net = net
        self.in_feature = in_feature
        self.top_block = top_block
        self._out_feature_strides = {"p{}".format(int(math.log2(s))): s for s in strides}

        if self.top_block is not None:
            for s in range(stage, stage + self.top_block.num_levels):
                self._out_feature_strides["p{}".format(s + 1)] = 2 ** (s + 1)

        self._out_features = list(self._out_feature_strides.keys())
        self._out_feature_channels = {k: out_channels for k in self._out_features}
        self._size_divisibility = strides[-1]
        self._square_pad = square_pad

    @property
    def padding_constraints(self):
        return {
            "size_divisiblity": self._size_divisibility,
            "square_size": self._square_pad,
        }

    def forward(self, x):
        bottom_up_features = self.net(x)
        features = bottom_up_features[self.in_feature]
        results = []

        for stage in self.stages:
            results.append(stage(features))

        if self.top_block is not None:
            if self.top_block.in_feature in bottom_up_features:
                top_block_in_feature = bottom_up_features[self.top_block.in_feature]
            else:
                top_block_in_feature = results[self._out_features.index(self.top_block.in_feature)]
            results.extend(self.top_block(top_block_in_feature))
        assert len(self._out_features) == len(results)
        return {f: res for f, res in zip(self._out_features, results)}

# -------------------------------------------------------------
# Create Object Detection Model with DINOv2 Backbone
# -------------------------------------------------------------

class DINOv2Backbone(Backbone):
    def __init__(self, dinov2_model, out_features='last_feat', out_channels=256):
        super().__init__()
        self.dinov2_model = dinov2_model
        self.out_features = out_features
        self.out_channels = out_channels
        
    def forward(self, x):
        x = self.dinov2_model(x)
        return {self.out_features: x}

# Load the DINOv2 model (replace with actual loading if necessary)
dinov2_vits14 = load('facebookresearch/dinov2', 'dinov2_vits14')

# Create the SimpleFeaturePyramid using DINOv2 backbone
backbone = SimpleFeaturePyramid(
    DINOv2Backbone(dinov2_vits14),
    in_feature='last_feat',
    out_channels=256,
    scale_factors=(4.0, 2.0, 1.0, 0.5),
    top_block=LastLevelMaxPool(),
    norm="LN",
    square_pad=1024
)

# RoIAlign pooling layer
roi_pooler = MultiScaleRoIAlign(
    featmap_names=backbone._out_features,
    output_size=7,
    sampling_ratio=2
)

# Create the Faster R-CNN model with DINOv2 backbone
model = FasterRCNN(
    backbone=backbone,
    num_classes=81,
    box_roi_pool=roi_pooler
)


