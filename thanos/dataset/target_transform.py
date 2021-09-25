import torch

def binary_label_transform(target_dict):
    label = target_dict["label"]
    if label != 0:
        label = 1
    return label
