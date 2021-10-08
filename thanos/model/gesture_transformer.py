import torch
import torch.nn as nn
from thanos.model.transformer import EncoderSelfAttention
from thanos.model.resnet import resnet10, resnet18

build_resnet_fn_dict = {
    "resnet10": resnet10,
    "resnet18": resnet18
}

class GestureTransformer(nn.Module):
    """A Transformer-Based Network for Dynamic Hand Gesture Recognition
    https://iris.unimore.it/retrieve/handle/11380/1212263/282584/3DV_2020.pdf
    """

    def __init__(self, 
        backbone="resnet18", 
        num_classes=14, 
        encoder_dim=256,
        vqk_dim=128,
        encoder_fc_dim=256,
        n_encoder_heads=6, 
        n_encoders=6, 
        **kwargs):

        super().__init__()
        self.backbone = build_resnet_fn_dict[backbone]()
        self.conv_proj = nn.Conv2d(512, encoder_dim, 1)
        self.ft_map_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.self_attention = EncoderSelfAttention(
            encoder_dim, vqk_dim, vqk_dim, dff=encoder_fc_dim,
            n_head=n_encoder_heads, n_module=n_encoders,**kwargs)
        self.pool = nn.AdaptiveAvgPool2d((1, encoder_dim))
        self.classifier = nn.Linear(encoder_dim, num_classes)

    def forward(self, x):
        # x -> (B, T, C, H, W)
        shape = list(x.shape)
        x = x.reshape([-1] + shape[-3:]) # (B*T, C, H, W)
        x = self.backbone(x) # (B*T, 512, h, w)
        x = self.conv_proj(x) # (B*T, encoder_dim, h, w)
        x = self.ft_map_avg_pool(x) # (B*T, encoder_dim, 1, 1)
        x = x.flatten(start_dim=1) # (B*T, encoder_dim)
        x = x.reshape(shape[0], shape[1], -1) # (B, T, encoder_dim)

        x = self.self_attention(x) # (B, T, encoder_dim)
        x = self.pool(x).squeeze(dim=1) # (B, encoder dim)
        x = self.classifier(x) # (B, num_classes)
        return x

    def inference(self, x, threshold=0.5):
        with torch.no_grad():
            probs = self.forward(x).sigmoid()
            pred_cls = (probs > threshold).to(torch.int)
        return pred_cls


if __name__ == "__main__":
    from thanos.model.utils import count_parameters
    detector = GestureTransformer()
    print("detector: ", count_parameters(detector))
    print("detetor backbone:", count_parameters(detector.backbone))
    print("detector encoder:", count_parameters(detector.self_attention))
    x = torch.rand(2, 16, 3, 240, 320)
    out = detector(x)
    print(out.shape)
