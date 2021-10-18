import argparse
import os
import time


import torch
from torch2trt import TRTModule

from thanos.model.gesture_transformer import classification_inference
from thanos.dataset import IPN_HAND_ROOT, IPN
from thanos.dataset.target_transform import read_label_from_target_dict
from thanos.trainers.data_augmentation import get_temporal_transform_fn, get_val_spatial_transform_fn


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("saved_engine", type=str, help="Path to serialized TensorRT engine")
    parser.add_argument("--temporal_stride", type=int, default=2, help="Model temporal stride")
    parser.add_argument("--temporal_len", type=int, default=22, help="Model input temporal length")
    parser.add_argument("--val_limit", type=int, default=None, help="Number of validation samples")
    args = parser.parse_args()

    model_trt = TRTModule()
    model_trt.load_state_dict(torch.load(args.saved_engine))

    
    # ann_path = os.path.join(IPN_HAND_ROOT, "annotations", "ipnall.json")
    # ipn = IPN(IPN_HAND_ROOT, ann_path, "validation",
    #         temporal_stride=args.temporal_stride,
    #         spatial_transform=get_val_spatial_transform_fn(), 
    #         temporal_transform=get_temporal_transform_fn(args.temporal_len, training=False),
    #         target_transform=read_label_from_target_dict)
    
    # for i, (sequence, target) in enumerate(ipn):
    #     tic = time.time()
    #     m_input = sequence[None].detach().cuda()
    #     logits = model_trt(m_input)
    #     toc = time.time()
    #     preds = classification_inference(logits)
    #     preds = preds.cpu()
    #     print((toc-tic)*1000, "ms")
    #     print("Pred/target: ", preds, target)
        
    #     if args.val_limit is not None:
    #         if i > args.val_limit:
    #             break
    x = torch.rand(1, args.temporal_len, 3, 240, 240).detach()
    for i in range(args.val_limit):
        tic = time.time()
        x = x.cuda()
        y = x.max().cpu()
        # logits = model_trt(x.cuda())
        # preds = classification_inference(logits)
        toc = time.time()
        # preds = preds.cpu()
        print((toc-tic)*1000, "ms")
    
