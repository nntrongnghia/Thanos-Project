import argparse
import numpy as np
import torch
from torch2trt import torch2trt
import time
from tqdm import tqdm

from thanos.model.transformer import EncoderSelfAttention
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
        out = model(x)
        toc = time.time()
        print((toc - tic)*1000)
        meas_time.append(toc - tic)
    meas_time_ms = np.array(meas_time) * 1000

    print("out.shape", out.shape)
    print(f"Mean: {meas_time_ms.mean():.1f} ms")
    print(f"Std: {meas_time_ms.std():.2f} ms")
    print(f"FPS: {(1000/meas_time_ms).mean()}")
    return meas_time_ms


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--encoder_dim", type=int, default=512)
    parser.add_argument("--att_dim", type=int, default=64)
    parser.add_argument("--hidden_dim", type=int, default=512)
    parser.add_argument("--nb_heads", type=int, default=8)
    parser.add_argument("--nb_modules", type=int, default=6)
    args = parser.parse_args()
    
    with torch.no_grad():
        print(f"=== Encoder ===")
        print("Encoder dim", args.encoder_dim)
        print("Input dim", args.att_dim)
        print("Hidden dim", args.hidden_dim)
        print("Nb heads", args.nb_heads)
        print("Nb modules", args.nb_modules)

        model = EncoderSelfAttention(
            d_model=args.encoder_dim,
            d_k=args.att_dim,
            d_v=args.att_dim,
            n_head=args.nb_heads,
            dff=args.hidden_dim,
            n_module=args.nb_modules
        ).eval()
        print(f"Encoder: {count_parameters(model):,d} params")
        meas_time_ms = torch_measure_inference_time(model, (1, 20, args.encoder_dim), N=10)
        print(meas_time_ms)


