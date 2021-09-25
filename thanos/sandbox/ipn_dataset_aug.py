import cv2
import torch
import os
import numpy as np
from torchvision.transforms.transforms import CenterCrop, RandomHorizontalFlip, RandomSizedCrop
from thanos.dataset_config import IPN_HAND_ROOT, INPUT_MEAN, INPUT_STD
from thanos.dataset import IPN, TemporalRandomCrop
import torchvision.transforms as T

def get_spatial_transform_fn():
    return T.Compose([
        T.RandomHorizontalFlip(),
        T.RandomResizedCrop((240, 240), scale=(0.8, 1.2), ratio=(1, 1)),
        T.RandomRotation(15),
        T.Normalize(INPUT_MEAN, INPUT_STD)
    ])

if __name__ == "__main__":
    ann_path = os.path.join(IPN_HAND_ROOT, "annotations", "ipnall.json")
    ipn = IPN(IPN_HAND_ROOT, ann_path, "training",
        spatial_transform=get_spatial_transform_fn(), 
        temporal_transform=TemporalRandomCrop(16)
    )
    for sequences, target in ipn:
        label = target["label"]
        for i in range(sequences.shape[0]):
            np_img = sequences[i].permute(1, 2, 0).numpy()
            np_img = np.ascontiguousarray(np_img)
            cv2.putText(
                np_img, 
                str(label), 
                (10, int(np_img.shape[0]*0.1)),
                cv2.FONT_HERSHEY_COMPLEX,
                1, (1.0, 0, 0)
            )
            cv2.imshow("sequences", np_img)
            cv2.waitKey(40)
        if cv2.waitKey(0) == 27: # ESC key
            cv2.destroyAllWindows
            break