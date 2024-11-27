from torchvision.models.detection import fasterrcnn_resnet50_fpn
import torch
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.transforms import functional as F
import torch.nn as nn
import torchvision.transforms as T


class MobileNet():
    """
    This module implements Vision Transformer (ViT) backbone in :paper:`vitdet`.
    "Exploring Plain Vision Transformer Backbones for Object Detection",
    https://arxiv.org/abs/2203.16527
    """

    def __init__(
        self,
    ):
        super().__init__()
        
        self.student_model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn(pretrained=True)
        # Ensure the student model matches the number of classes in Pascal VOC (20 classes)
        self.num_classes = 21  # 20 classes + background
        self.in_features = student_model.roi_heads.box_predictor.cls_score.in_features
        self.student_model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)
        # device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    def forward(self, x):
        
        outputs = self.student_model(x)

        return outputs

def create_model(num_classes=81, pretrained=True, coco_model=False):
    # Base
    embed_dim, depth, num_heads, dp = 768, 12, 12, 0.1
    # Load the pretrained SqueezeNet1_1 backbone.
    net = ViT(  # Single-scale ViT backbone
        img_size=1024,
        patch_size=16,
        embed_dim=embed_dim,
        depth=depth,
        num_heads=num_heads,
        drop_path_rate=dp,
        window_size=14,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        window_block_indexes=[
            # 2, 5, 8 11 for global attention
            0,
            1,
            3,
            4,
            6,
            7,
            9,
            10,
        ],
        residual_block_indexes=[],
        use_rel_pos=True,
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
