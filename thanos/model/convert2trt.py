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
    parser.add_argument("checkpoint", type=str, help="Path to checkpoint")
    parser.add_argument("--verbose", action="store_true", help="TensorRT verbose log")
    args = parser.parse_args()
    trt_logger = trt.Logger.VERBOSE if args.verbose else trt.Logger.WARNING
    # Load model from checkpoint and config 
    config = load_config(args.config)
    lit_model = LitGestureTransformer.load_from_checkpoint(
        args.checkpoint, 
        config=config, 
        override_model_config={"return_aux": False}).eval().cuda()
    x = torch.rand((1, config.input_duration, 3, 240, 240)).cuda()
    # Convert to TensorRT
    trt_model = torch2trt(
        lit_model.model, [x],
        use_onnx=True,
        # fp16_mode=True,
        # strict_type_constraints=True,
        log_level=trt_logger)
    # Save trt_model
    trt_model_path = "".join(args.checkpoint.split(".")[:-1]) + ".trt"
    torch.save(trt_model.state_dict(), trt_model_path)
    # Measure inference time
    for _ in range(10):
        tic = time.time()
        m_outputs = trt_model(x)
        toc = time.time()
        print(m_outputs.shape)
        print((toc-tic)*1000, "ms")
    # Get sample torch outputs
    torch_outputs = lit_model(x)
    if isinstance(torch_outputs, dict) and "logits" in torch_outputs:
        torch_outputs = torch_outputs["logits"]
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
