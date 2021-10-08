import cv2
import torch
import os
import numpy as np
from torch.utils.data import DataLoader
import torchvision.transforms as T
from thanos.dataset import (
    IPN, binary_label_transform, read_label_from_target_dict,
    one_hot_label_transform,
    IPN_HAND_ROOT, INPUT_MEAN, INPUT_STD)

from thanos.trainers.data_augmentation import ( 
    get_temporal_transform_fn,
    get_train_spatial_transform_fn,
    get_val_spatial_transform_fn)


# def get_spatial_transform_fn():
#     return T.Compose([
#         T.RandomHorizontalFlip(),
#         T.RandomResizedCrop((240, 240), scale=(0.8, 1.2), ratio=(1, 1)),
#         T.RandomRotation(15),
#         T.Normalize(INPUT_MEAN, INPUT_STD)
#     ])


if __name__ == "__main__":
    ann_path = os.path.join(IPN_HAND_ROOT, "annotations", "ipnall.json")
    ipn = IPN(IPN_HAND_ROOT, ann_path, "training",
        temporal_stride=2,
        spatial_transform=get_train_spatial_transform_fn(), 
        temporal_transform=get_temporal_transform_fn(20),
        target_transform=one_hot_label_transform
    )
    dataloader = DataLoader(ipn, batch_size=8, shuffle=True)
    # ipn = IPN(IPN_HAND_ROOT, ann_path, "validation",
    #     spatial_transform=get_val_spatial_transform_fn(), 
    #     temporal_transform=get_temporal_transform_fn(16),
    #     target_transform=binary_label_transform)
    # dataloader = DataLoader(ipn, batch_size=8, shuffle=True)
    # for sequences, target in ipn:
    #     for i in range(sequences.shape[0]):
    #         np_img = sequences[i].permute(1, 2, 0).numpy()
    #         np_img = np.ascontiguousarray(np_img)
    #         cv2.putText(
    #             np_img, 
    #             str(target), 
    #             (10, int(np_img.shape[0]*0.1)),
    #             cv2.FONT_HERSHEY_COMPLEX,
    #             1, (1.0, 0, 0)
    #         )
    #         cv2.imshow("sequences", np_img)
    #         cv2.waitKey(40)
    #     if cv2.waitKey(0) == 27: # ESC key
    #         cv2.destroyAllWindows
    #         break
    key = 0
    for batch, targets in dataloader:
        print(batch.shape)
        print(targets)
        print(targets.shape)
        break
        for b in range(batch.shape[0]):
            sequences = batch[b]
            target = targets[b]
            for i in range(sequences.shape[0]):
                np_img = sequences[i].permute(1, 2, 0).numpy()
                np_img = np.ascontiguousarray(np_img)
                cv2.putText(
                    np_img, 
                    str(int(target)), 
                    (10, int(np_img.shape[0]*0.1)),
                    cv2.FONT_HERSHEY_COMPLEX,
                    1, (1.0, 0, 0)
                )
                cv2.imshow("sequences", np_img)
                cv2.waitKey(40)
            key = cv2.waitKey(0)
            if key == 27: # ESC key
                cv2.destroyAllWindows
                break
        if key == 27:
            break

            