import torch
import torch.nn as nn
from thanos.model import GestureTransformer, build_detector

class GestureDetector(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = build_detector()

    def forward(self, x):
        logits = self.model(x)
        return logits

    def inference(self, x, threshold=0.5):
        probs = self.forward(x).sigmoid()
        pred_cls = (probs > threshold).to(torch.int)
        return pred_cls