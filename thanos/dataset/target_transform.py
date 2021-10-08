import torch
import torch.nn.functional as F

def binary_label_transform(target_dict, **kwargs):
    label = target_dict["label"]
    if label != 0:
        label = 1
    return label

def read_label_from_target_dict(target_dict, **kwargs):
    return target_dict["label"]

def one_hot_label_transform(target_dict, num_classes):
    label = target_dict["label"]
    return F.one_hot(torch.tensor([label]), num_classes=num_classes).reshape(-1,)