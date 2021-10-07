import numpy as np
import torch
from torch2trt import torch2trt
import time
from tqdm import tqdm

from thanos.model.resnet import resnet10, resnet18
from thanos.model.utils import count_parameters

def torch_measure_inference_time(model, input_shape, device=None, N=10):
    if device is None:
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"=== Measure inference time on {device} ===")
    model.to(device)
    x = torch.rand(input_shape, device=device)
    # warm-up run
    for i in range(5):
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
    with torch.no_grad():
        # model = resnet10().eval()
        # print("=== Resnet 10 ===")
        # print(f"Resnet 10: {count_parameters(model):,d} params")
        # meas_time_ms = torch_measure_inference_time(model, (8, 3, 640, 480), N=10)
        # print(meas_time_ms)

        model = resnet18().eval()
        print("=== Resnet 18 ===")
        print(f"Resnet 18: {count_parameters(model):,d} params")
        meas_time_ms = torch_measure_inference_time(model, (8, 3, 640, 480), N=5)
        print(meas_time_ms)

