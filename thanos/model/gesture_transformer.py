import torch
import torch.nn as nn
from thanos.model.transformer import EncoderSelfAttention
from thanos.model.resnet import resnet10, resnet18

class GestureTransformer(nn.Module):
    """A Transformer-Based Network for Dynamic Hand Gesture Recognition
    https://iris.unimore.it/retrieve/handle/11380/1212263/282584/3DV_2020.pdf
    """

    def __init__(self, backbone: nn.Module, num_classes: int, n_encoder_heads:int, n_encoders:int, **kwargs):
        super().__init__()
        self.backbone = backbone
        self.ft_map_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.self_attention = EncoderSelfAttention(
            512, 64, 64, dff=512,
            n_head=n_encoder_heads, n_module=n_encoders,**kwargs)
        self.pool = nn.AdaptiveAvgPool2d((1, 512))
        self.classifier = nn.Linear(512, num_classes)

    def forward(self, x):
        # x -> (B, T, C, H, W)
        shape = list(x.shape)
        x = x.reshape([-1] + shape[-3:]) # (B*T, C, H, W)
        x = self.backbone(x) # (B*T, 512, h, w)
        x = self.ft_map_avg_pool(x) # (B*T, 512, 1, 1)
        x = x.flatten(1) # (B*T, 512)
        x = x.reshape(shape[0], shape[1], -1) # (B, T, 512)

        # x = self.self_attention(x)
        # x = self.pool(x).squeeze(dim=1)
        # x = self.classifier(x)
        return x

def build_detector():
    backbone = resnet10()
    model = GestureTransformer(
        backbone,
        num_classes=2, 
        n_encoder_heads=4,
        n_encoders=3
    )
    return model

if __name__ == "__main__":
    from thanos.model.utils import count_parameters
    detector = build_detector()
    print("detector: ", count_parameters(detector))
    print("detetor backbone:", count_parameters(detector.backbone))
    print("detector encoder:", count_parameters(detector.self_attention))
    x = torch.rand(2, 16, 3, 112, 112)
    out = detector(x)
    print(out.shape)
