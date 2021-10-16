import argparse
import numpy as np
import torch
import torch.nn as nn
from torch2trt import torch2trt
import time
from tqdm import tqdm
import tensorrt as trt

from thanos.model.transformer import EncoderSelfAttention
from thanos.model.utils import count_parameters

class ScaledDotProductAttention(nn.Module):
    """
    Scaled dot-product attention
    """

    def __init__(self, d_model, d_k, d_v, h):
        """
        :param d_model: Output dimensionality of the model
        :param d_k: Dimensionality of queries and keys
        :param d_v: Dimensionality of values
        :param h: Number of heads
        """
        super(ScaledDotProductAttention, self).__init__()
        self.fc_q = nn.Linear(d_model, h * d_k)
        self.fc_k = nn.Linear(d_model, h * d_k)
        self.fc_v = nn.Linear(d_model, h * d_v)
        self.fc_o = nn.Linear(h * d_v, d_model)

        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.h = h

        self.init_weights(gain=1.0)

    def init_weights(self, gain=1.0):
        nn.init.xavier_normal_(self.fc_q.weight, gain=gain)
        nn.init.xavier_normal_(self.fc_k.weight, gain=gain)
        nn.init.xavier_normal_(self.fc_v.weight, gain=gain)
        nn.init.xavier_normal_(self.fc_o.weight, gain=gain)
        nn.init.constant_(self.fc_q.bias, 0)
        nn.init.constant_(self.fc_k.bias, 0)
        nn.init.constant_(self.fc_v.bias, 0)
        nn.init.constant_(self.fc_o.bias, 0)

    def forward(self, x):
        """
        Computes
        :param queries: Queries (b_s, nq, d_model)
        :param keys: Keys (b_s, nk, d_model)
        :param values: Values (b_s, nk, d_model)
        :return:
        """
        queries = keys = values = x
        b_s, nq = queries.shape[:2]
        nk = keys.shape[1]
        q = self.fc_q(queries).view(b_s, nq, self.h, self.d_k).permute(0, 2, 1, 3)  # (b_s, h, nq, d_k)
        k = self.fc_k(keys).view(b_s, nk, self.h, self.d_k).permute(0, 2, 3, 1)  # (b_s, h, d_k, nk)
        v = self.fc_v(values).view(b_s, nk, self.h, self.d_v).permute(0, 2, 1, 3)  # (b_s, h, nk, d_v)
        att = torch.matmul(q, k) / np.sqrt(self.d_k)  # (b_s, h, nq, nk)

        att = torch.nn.functional.softmax(att, -1)

        out = torch.matmul(att, v).permute(0, 2, 1, 3).contiguous().view(b_s, nq, self.h * self.d_v)  # (b_s, nq, h*d_v)
        out = self.fc_o(out)  # (b_s, nq, d_model)
        return out

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
    parser = argparse.ArgumentParser()
    parser.add_argument("--encoder_dim", type=int, default=512)
    parser.add_argument("--att_dim", type=int, default=64)
    parser.add_argument("--hidden_dim", type=int, default=512)
    parser.add_argument("--nb_heads", type=int, default=8)
    parser.add_argument("--nb_modules", type=int, default=6)
    args = parser.parse_args()
    
    print(f"=== Encoder ===")
    print("Encoder dim", args.encoder_dim)
    print("Input dim", args.att_dim)
    print("Hidden dim", args.hidden_dim)
    print("Nb heads", args.nb_heads)
    print("Nb modules", args.nb_modules)
    test_shape = (1, 20, args.encoder_dim)

    model = ScaledDotProductAttention(
        args.encoder_dim, args.att_dim, args.att_dim, args.nb_heads
    ).eval().cuda()
    x = torch.rand(test_shape).cuda()
    print(f"Encoder: {count_parameters(model):,d} params")
    trt_model = torch2trt(model, [x], log_level=trt.Logger.VERBOSE,
        fp16_mode=True, 
        max_workspace_size=1<<30, 
        strict_type_constraints=True, 
    )
    meas_time_ms = trt_measure_inference_time(model, test_shape, N=10)
    print(meas_time_ms)
    # for _ in range(10):
    #     tic = time.time()
    #     model(x)
    #     toc = time.time()
    #     print((toc - tic)*1000, "ms")

