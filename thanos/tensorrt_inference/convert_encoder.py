import argparse
import os
import time
import numpy as np

import tensorrt as trt
import torch
import torch.nn as nn
from thanos.model.gesture_transformer import GestureTransformer
from thanos.trainers import load_config
from thanos.trainers.lit_detector import LitGestureTransformer
from torch2trt import torch2trt


class GestureEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attention = None
        self.classifier = None
    
    def forward(self, x):
        # X (1, T, encoder_dim)
        x = self.self_attention(x)
        x = x.mean(dim=-2)
        x = self.classifier(x)
        return x

    def load_from_model(self, model: GestureTransformer):
        if not isinstance(model, GestureTransformer):
            raise ValueError("must be GestureTransformer class")
        self.self_attention = model.self_attention
        self.classifier = model.classifier
        return self

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str, help="Path to config py file")
    parser.add_argument("checkpoint", type=str, help="Path to checkpoint")
    # parser.add_argument("--fp16", action="store_true", help="Use FP16 precision")
    parser.add_argument("--verbose", action="store_true", help="TensorRT verbose log")

    args = parser.parse_args()
    trt_logger = trt.Logger.VERBOSE if args.verbose else trt.Logger.WARNING
    # Load model from checkpoint and config 
    config = load_config(args.config)
    lit_model = LitGestureTransformer.load_from_checkpoint(
        args.checkpoint, 
        config=config, 
        override_model_config={"return_aux": False}).eval()
    encoder = GestureEncoder().load_from_model(lit_model.model).eval().cuda()
    seq_len = encoder.self_attention.seq_len
    encoder_dim = encoder.self_attention.encoder_dim
    x = torch.rand((1, seq_len, encoder_dim)).to(torch.float32).cuda()
    # Convert to TensorRT
    trt_model = torch2trt(
        encoder, [x],
        input_names=["ft_vec_seq"],
        output_names=["logits"],
        use_onnx=True,
        fp16_mode=True,
        log_level=trt_logger)
    # Save trt_model
    trt_engine_path = "".join(args.checkpoint.split(".")[:-1]) + "_encoder_fp16.trt"
    # torch.save(trt_model.state_dict(), trt_engine_path)
    with open(trt_engine_path, "wb") as f:
        print("Serializing TensorRT engine:")
        print(trt_engine_path)
        f.write(trt_model.engine.serialize())
    
    # Measure inference time
    meas_time = []
    for _ in range(50):
        tic = time.time()
        m_outputs = trt_model(x)
        torch.cuda.synchronize()
        toc = time.time()
        inference_time = (toc - tic)*1000
        meas_time.append(inference_time)
        print(inference_time, "ms")
    meas_time = np.array(meas_time)

    print(m_outputs.shape)
    # Get sample torch outputs
    torch_outputs = encoder(x)
    # Compare Torch ouput and TensorRT output
    print(torch_outputs.shape)
    print(m_outputs.shape)
    diff = torch_outputs - m_outputs.reshape(torch_outputs.shape) 
    diff = diff.abs()
    print("Difference - PyTorch vs TensorRT")
    print("Min", diff.min())
    print("Max", diff.max())
    print("Mean", diff.mean())
    print("Std", diff.std())
