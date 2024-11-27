"""
A lot of scripts borrowed/adapted from Detectron2.
https://github.com/facebookresearch/detectron2/blob/38af375052d3ae7331141bc1a22cfa2713b02987/detectron2/modeling/backbone/backbone.py#L11
"""

import torch.nn as nn
import torch
import torch.nn.functional as F
import math
import torchvision

from functools import partial
from torchvision.models.detection import FasterRCNN
from models.layers import (
    Backbone, 
    PatchEmbed, 
    Block, 
    get_abs_pos,
    get_norm,
    Conv2d,
    LastLevelMaxPool
)
from models.utils import _assert_strides_are_log2_contiguous

class ViT(Backbone):
    """
    This module implements Vision Transformer (ViT) backbone in :paper:`vitdet`.
    "Exploring Plain Vision Transformer Backbones for Object Detection",
    https://arxiv.org/abs/2203.16527
    """

    def __init__(
        self,
        img_size=1024,
        patch_size=14,
        in_chans=3,
        embed_dim=768,
        drop_path_rate=0.0,
        norm_layer=nn.LayerNorm,
        act_layer=nn.GELU,
        use_act_checkpoint=False,
        pretrain_img_size=224,
        pretrain_use_cls_token=True,
        out_feature="last_feat",
    ):
        """
        """
        super().__init__()

        # Load the pre-trained DINO model
        # self.dinov2_vits14_lc = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14_lc')
        self.dinov2_vitb14_lc = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14_lc')
        # Extract the backbone from the DINO model (removing the classification head)
        self.dinov2_backbone = self.dinov2_vitb14_lc.backbone

        self._out_features = [out_feature]

        # Freeze the backbone parameters
        for param in self.dinov2_backbone.parameters():
            param.requires_grad = False

        self._out_feature_channels = {out_feature: embed_dim}
        self._out_feature_strides = {out_feature: patch_size}

    #     self.apply(self._init_weights)

    # def _init_weights(self, m):
    #     if isinstance(m, nn.Linear):
    #         nn.init.trunc_normal_(m.weight, std=0.02)
    #         if isinstance(m, nn.Linear) and m.bias is not None:
    #             nn.init.constant_(m.bias, 0)
    #     elif isinstance(m, nn.LayerNorm):
    #         nn.init.constant_(m.bias, 0)
    #         nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        # print ("input to DINO: ", x.shape)
        feature_vector = self.dinov2_backbone(x)
        # print ("feature_vector: ", feature_vector.shape)
        feature_vector = feature_vector.unsqueeze(2).unsqueeze(3)
        replicated_tensor = feature_vector.repeat(1, 1, 50, 50)

        # x = self.patch_embed(x)
        # if self.pos_embed is not None:
        #     x = x + get_abs_pos(
        #         self.pos_embed, self.pretrain_use_cls_token, (x.shape[1], x.shape[2])
        #     )

        # for blk in self.blocks:
        #     x = blk(x)

        # outputs = {self._out_features[0]: x.permute(0, 3, 1, 2)}
        # print ("outputs[key], outputs[val] : ", self._out_features[0], outputs[self._out_features[0]].shape)
        # return outputs
        outputs = {self._out_features[0]: replicated_tensor}
        return outputs

class SimpleFeaturePyramid(Backbone):
    """
    This module implements SimpleFeaturePyramid in :paper:`vitdet`.
    It creates pyramid features built on top of the input feature map.
    """

    def __init__(
        self,
        net,
        in_feature,
        out_channels,
        scale_factors,
        top_block=None,
        norm="LN",
        square_pad=0,
    ):
        """
        :param net (Backbone): module representing the subnetwork backbone.
                Must be a subclass of :class:`Backbone`.
        :param in_feature (str): names of the input feature maps coming
                from the net.
        :param out_channels (int): number of channels in the output feature maps.
        :param scale_factors (list[float]): list of scaling factors to upsample or downsample
                the input features for creating pyramid features.
        :param top_block (nn.Module or None): if provided, an extra operation will
                be performed on the output of the last (smallest resolution)
                pyramid output, and the result will extend the result list. The top_block
                further downsamples the feature map. It must have an attribute
                "num_levels", meaning the number of extra pyramid levels added by
                this block, and "in_feature", which is a string representing
                its input feature (e.g., p5).
        :param norm (str): the normalization to use.
        :param square_pad (int): If > 0, require input images to be padded to specific square size.
        """
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
                    Conv2d(
                        out_dim,
                        out_channels,
                        kernel_size=1,
                        bias=use_bias,
                        norm=get_norm(norm, out_channels),
                    ),
                    Conv2d(
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
        # Return feature names are "p<stage>", like ["p2", "p3", ..., "p6"]
        self._out_feature_strides = {"p{}".format(int(math.log2(s))): s for s in strides}
        # top block output feature maps.
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
        """
        :param x: Tensor of shape (N,C,H,W). H, W must be a multiple of ``self.size_divisibility``.
        Returns:
            dict[str->Tensor]:
                mapping from feature map name to pyramid feature map tensor
                in high to low resolution order. Returned feature names follow the FPN
                convention: "p<stage>", where stage has stride = 2 ** stage e.g.,
                ["p2", "p3", ..., "p6"].
        """
        #print ("input to SimpleFeaturePyramid: ", x.shape)
        bottom_up_features = self.net(x)
        #print ("bottom_up_features keys: ", bottom_up_features.keys())
        #print ("bottom_up_features values: ", bottom_up_features.values())
        features = bottom_up_features[self.in_feature]
        #print ("features: ", features.shape)
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
        print ("self._out_features ", self._out_features)
        return {f: res for f, res in zip(self._out_features, results)}

def create_model(num_classes=81, pretrained=True, coco_model=False):
    # Base
    embed_dim, depth, num_heads, dp = 768, 12, 12, 0.1
    # Load the pretrained SqueezeNet1_1 backbone.
    net = ViT(  # Single-scale ViT backbone
        img_size=1024,
        patch_size=16,
        embed_dim=embed_dim,
        out_feature="last_feat",
    )

    if pretrained:
        print('Loading MAE Pretrained ViT Base weights...')
        # ckpt = torch.utis('weights/mae_pretrain_vit_base.pth')
        ckpt = torch.hub.load_state_dict_from_url('https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_base.pth')
        net.load_state_dict(ckpt['model'], strict=False)

    backbone = SimpleFeaturePyramid(
        net,
        in_feature="last_feat",
        out_channels=256,
        scale_factors=(4.0, 2.0, 1.0, 0.5),
        top_block=LastLevelMaxPool(),
        norm="LN",
        square_pad=1024,
    )

    print ("SFP backbone ", backbone)

    backbone.out_channels = 256

    # Feature maps to perform RoI cropping.
    # If backbone returns a Tensor, `featmap_names` is expected to
    # be [0]. We can choose which feature maps to use.
    roi_pooler = torchvision.ops.MultiScaleRoIAlign(
        featmap_names=backbone._out_features,
        output_size=7,
        sampling_ratio=2
    )

    # Final Faster RCNN model.
    model = FasterRCNN(
        backbone=backbone,
        num_classes=num_classes,
        box_roi_pool=roi_pooler
    )
    return model

if __name__ == '__main__':
    from model_summary import summary
    model = create_model(81, pretrained=True)
    summary(model)
