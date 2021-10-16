import torch
import torch.nn as nn
import torchvision
from thanos.model.resnet import resnet10, resnet18
from thanos.model.transformer import EncoderSelfAttention

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
        return_aux=False,
        seq_len=22,
        **kwargs):

        super().__init__()
        self.encoder_dim = encoder_dim
        self.num_classes = num_classes
        self.return_aux = return_aux
        self.backbone = build_resnet_fn_dict[backbone]()
        self.conv_proj = nn.Conv2d(512, encoder_dim, 1)
        self.ft_map_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.self_attention = EncoderSelfAttention(
            encoder_dim, vqk_dim, vqk_dim, dff=encoder_fc_dim,
            n_head=n_encoder_heads, n_module=n_encoders, return_aux=return_aux, 
            seq_len=seq_len, **kwargs)
        # self.pool = nn.AdaptiveAvgPool2d((1, encoder_dim))
        self.classifier = nn.Linear(encoder_dim, num_classes)

    def forward(self, x):
        # x -> (B, T, C, H, W)
        # shape = list(x.shape)
        # TensorRT does not support reshape with -1
        B, T, C, H, W = x.shape
        x = x.reshape(B*T, C, H, W) # (B*T, C, H, W)
        x = self.backbone(x) # (B*T, 512, h, w)
        x = self.conv_proj(x) # (B*T, encoder_dim, h, w)
        x = self.ft_map_avg_pool(x) # (B*T, encoder_dim, 1, 1)
        x = x.flatten(start_dim=1) # (B*T, encoder_dim)
        x = x.reshape(B, T, self.encoder_dim) # (B, T, encoder_dim)
        if self.return_aux:
            x = self.self_attention(x) # (nb_encoders, B, T, encoder_dim)
            logits = self.classifier(x[-1].mean(dim=-2))
                # self.pool(x[-1]).squeeze(dim=1))
            aux = []
            for i in range(x.shape[0]-1):
                aux_logits = self.classifier(x[i].mean(dim=-2))
                    # self.pool(x[i]).squeeze(dim=1))
                aux.append(aux_logits)
            return {"logits": logits, "aux": aux}
        else:
            x = self.self_attention(x) # (B, T, encoder_dim)
            x = x.mean(dim=-2)
            # x = self.pool(x).squeeze(dim=1) # (B, encoder dim)
            x = self.classifier(x) # (B, num_classes)
            return {"logits": x}


@torch.no_grad()
def classification_inference(logits:torch.Tensor, return_prob=False) -> torch.Tensor:
    if return_prob:
        return logits.argmax(dim=-1), logits.sigmoid()
    else:
        return logits.argmax(dim=-1)

        
if __name__ == "__main__":
    from thanos.model.utils import count_parameters
    detector = GestureTransformer(encoder_dim=512, seq_len=20)
    print("detector: ", count_parameters(detector))
    print("detetor backbone:", count_parameters(detector.backbone))
    print("detector encoder:", count_parameters(detector.self_attention))
    x = torch.rand(1, 20, 3, 240, 320)
    out = detector(x)
    # print(out["logits"].shape)
