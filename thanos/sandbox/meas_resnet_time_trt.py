import numpy as np
import torch
from torch2trt import torch2trt
import time
from tqdm import tqdm

from thanos.model.resnet import resnet10, resnet18
from thanos.model.utils import count_parameters

def trt_measure_inference_time(model, input_shape, N=10):
    device = torch.device("cuda")
    print(f"=== Measure inference time on {device} ===")
    x = torch.rand(input_shape).cuda()
    # warm-up run
    for i in range(3):
        model(x)

    meas_time = []
    for i in tqdm(range(N)):
        tic = time.time()
        model(x)
        toc = time.time()
        meas_time.append(toc - tic)
    meas_time_ms = np.array(meas_time) * 1000

    print(f"Mean: {meas_time_ms.mean():.1f} ms")
    print(f"Std: {meas_time_ms.std():.2f} ms")
    print(f"FPS: {(1000/meas_time_ms).mean()}")
    return meas_time_ms


if __name__ == "__main__":
    device = torch.device("cuda")
    print("=== Resnet 10 ===")
    model = resnet10().eval().cuda()
    x = torch.rand((2, 3, 640, 480)).cuda()
    model_trt = torch2trt(model, [x], max_batch_size=2)
    print(f"Gesture Detector: {count_parameters(model):,d} params")
    meas_time_ms = trt_measure_inference_time(model_trt, (2, 3, 640, 480), N=5)
    print(meas_time_ms)

    print("=== Resnet 18 ===")
    model = resnet18().eval().cuda()
    x = torch.rand((2, 3, 640, 480)).cuda()
    model_trt = torch2trt(model, [x], max_batch_size=2)
    print(f"Gesture Detector: {count_parameters(model):,d} params")
    meas_time_ms = trt_measure_inference_time(model_trt, (2, 3, 640, 480), N=5)
    print(meas_time_ms)

