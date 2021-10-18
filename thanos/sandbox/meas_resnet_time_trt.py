import numpy as np
import torch
from torch2trt import torch2trt
import time

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
    for i in range(N):
        tic = time.time()
        y = model(x)
        print(y.cpu().shape)
        toc = time.time()
        print((toc - tic)*1000, "ms")
        meas_time.append(toc - tic)
    meas_time_ms = np.array(meas_time) * 1000

    print(f"Mean: {meas_time_ms.mean():.1f} ms")
    print(f"Std: {meas_time_ms.std():.2f} ms")
    print(f"FPS: {(1000/meas_time_ms).mean()}")
    return meas_time_ms


if __name__ == "__main__":
    device = torch.device("cuda")
    test_shape = (1, 3, 240, 240)
    model = resnet18().eval().cuda()
    x = torch.rand(test_shape).cuda()
    model_trt = torch2trt(
        model, [x], max_batch_size=test_shape[0],
        fp16_mode=True)
    print(f"Gesture Detector: {count_parameters(model):,d} params")
    meas_time_ms = trt_measure_inference_time(model_trt, test_shape, N=10)
    print(meas_time_ms)

