import argparse
import os
import time
import pycuda.driver as cuda
import numpy as np

from thanos.tensorrt_inference import TRTExecutor
# from thanos.dataset import IPN, IPN_HAND_ROOT
# from thanos.dataset.target_transform import read_label_from_target_dict
# from thanos.model.gesture_transformer import classification_inference
# from thanos.trainers.data_augmentation import (get_temporal_transform_fn,
#                                                get_val_spatial_transform_fn)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("trt_engine", type=str, help="Path to serialized TensorRT engine")
    parser.add_argument("--temporal_stride", type=int, default=2, help="Model temporal stride")
    parser.add_argument("--temporal_len", type=int, default=22, help="Model input temporal length")
    parser.add_argument("--val_limit", type=int, default=None, help="Number of validation samples")
    args = parser.parse_args()

    trt_model = TRTExecutor(args.trt_engine, sync_mode=True)
    trt_model.print_bindings_info()
    
    # ann_path = os.path.join(IPN_HAND_ROOT, "annotations", "ipnall.json")
    # ipn = IPN(IPN_HAND_ROOT, ann_path, "validation",
    #         temporal_stride=args.temporal_stride,
    #         spatial_transform=get_val_spatial_transform_fn(), 
    #         temporal_transform=get_temporal_transform_fn(args.temporal_len, training=False),
    #         target_transform=read_label_from_target_dict)

    # for i, (sequence, target) in enumerate(ipn):
    #     if args.val_limit is not None:
    #         if i >= args.val_limit:
    #             break
    #     m_input = sequence[None].detach().numpy()
    #     tic = time.time()
    #     logits = trt_model(m_input)
    #     toc = time.time()
    #     print(logits)
    #     print((toc-tic)*1000, "ms")
    #     # print("Pred/target: ", preds, target)
    
    x = np.random.rand(1, 22, 3, 240, 240).astype(np.float32)
    trt_model.inputs[0].host = x
    inp = trt_model.inputs[0]
    for i in range(args.val_limit):
        cuda.memcpy_htod(inp.device, inp.host)
        tic = time.time()
        trt_model.context.execute_v2(bindings=trt_model.bindings)
        toc = time.time()
        print((toc - tic)*1000, "ms")

