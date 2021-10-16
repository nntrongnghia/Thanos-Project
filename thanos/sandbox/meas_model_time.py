import time
import torch
import os
import pytorch_lightning as pl
import argparse
import tensorrt as trt
from torch2trt import torch2trt

from thanos.trainers.lit_detector import LitGestureTransformer
from thanos.trainers import load_config


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str, help="Path to config py file")
    parser.add_argument("--verbose", action="store_true", help="TensorRT verbose log")
    args = parser.parse_args()
    trt_logger = trt.Logger.VERBOSE if args.verbose else trt.Logger.WARNING
    config = load_config(args.config)
    lit_model = LitGestureTransformer(
        config=config, 
        override_model_config={"return_aux": False}).cuda()
    lit_model.eval()
    x = torch.rand((1, 20, 3, 240, 240)).cuda()
    torch_outputs = lit_model(x)
    if isinstance(torch_outputs, dict) and "logits" in torch_outputs:
        torch_outputs = torch_outputs["logits"]
    # with torch.no_grad():
    #     for _ in range(10):
    #         tic = time.time()
    #         lit_model(x)
    #         toc = time.time()
    #         print((toc - tic)*1000)
    trt_model = torch2trt(
        lit_model.model, [x],
        use_onnx=True,
        # fp16_mode=True,
        # strict_type_constraints=True,
        log_level=trt_logger)
    for _ in range(10):
        tic = time.time()
        m_outputs = trt_model(x)
        toc = time.time()
        print(m_outputs.shape)
        print((toc-tic)*1000, "ms")
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