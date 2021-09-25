import torchvision.transforms as T
from thanos.dataset_config import INPUT_MEAN, INPUT_STD
from thanos.dataset import TemporalRandomCrop

def get_spatial_transform_fn():
    return T.Compose([
        T.RandomHorizontalFlip(),
        T.RandomResizedCrop((240, 240), scale=(0.8, 1.2), ratio=(1, 1)),
        T.RandomRotation(15),
        T.Normalize(INPUT_MEAN, INPUT_STD)
    ])

def get_temporal_transform_fn(duration:int):
    return TemporalRandomCrop(duration)