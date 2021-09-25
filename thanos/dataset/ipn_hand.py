import torch
import torch.utils.data as data
from PIL import Image
import os
import math
import functools
import json
import copy
from numpy.random import randint
import numpy as np
import random
from tqdm import tqdm
import torchvision.transforms as T

def pil_loader(path, modality):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        #print(path)
        with Image.open(f) as img:
            if modality in ['RGB', 'flo']:
                return img.convert('RGB')
            elif modality in ['Depth', 'seg']:
                return img.convert('L') # 8-bit pixels, black and white check from https://pillow.readthedocs.io/en/3.0.x/handbook/concepts.html


def get_default_image_loader():
    return pil_loader


def video_loader(video_dir_path, frame_indices, modality, sample_duration, image_loader):
    video = []
    if modality in ['RGB', 'flo', 'seg']:
        for i in frame_indices:
            image_path = os.path.join(video_dir_path, '{:s}_{:06d}.jpg'.format(video_dir_path.split('/')[-1],i))
            if os.path.exists(image_path):  
                video.append(image_loader(image_path, modality))
            else:
                print(image_path, "------- Does not exist")
                return video
    return video

def get_default_video_loader():
    image_loader = get_default_image_loader()
    return functools.partial(video_loader, image_loader=image_loader)


def load_annotation_data(data_file_path):
    with open(data_file_path, 'r') as data_file:
        return json.load(data_file)


def get_class_labels(data):
    class_labels_map = {}
    index = 0
    for class_label in data['labels']:
        class_labels_map[class_label] = index
        index += 1
    return class_labels_map


def get_video_names_and_annotations(data, subset):
    video_names = []
    annotations = []

    for key, value in data['database'].items():
        this_subset = value['subset']
        if this_subset == subset:
            label = value['annotations']['label']
            #video_names.append('{}/{}'.format(label, key))
            video_names.append(key.split('^')[0])
            annotations.append(value['annotations'])

    return video_names, annotations


def make_dataset(root_path, annotation_path, subset, n_samples_for_each_video,
                 sample_duration):
    data = load_annotation_data(annotation_path)
    video_names, annotations = get_video_names_and_annotations(data, subset)
    class_to_idx = get_class_labels(data)
    idx_to_class = {}
    for name, label in class_to_idx.items():
        idx_to_class[label] = name

    dataset = []
    print("[INFO]: IPN Dataset - " + subset + " is loading...")
    # print("  path: " + video_names[0])
    for i in tqdm(range(len(video_names))):
        video_path = os.path.join(root_path, video_names[i])
        
        if not os.path.exists(video_path):
            continue
    
        begin_t = int(annotations[i]['start_frame'])
        end_t = int(annotations[i]['end_frame'])
        n_frames = end_t - begin_t + 1
        sample = {
            'video': video_path,
            'segment': [begin_t, end_t],
            'n_frames': n_frames,
            #'video_id': video_names[i].split('/')[1]
            'video_id': i
        }
        if len(annotations) != 0:
            sample['label'] = class_to_idx[annotations[i]['label']]
        else:
            sample['label'] = -1

        if n_samples_for_each_video == 1:
            sample['frame_indices'] = list(range(begin_t, end_t + 1))
            dataset.append(sample)
        else:
            if n_samples_for_each_video > 1:
                step = max(1,
                           math.ceil((n_frames - 1 - sample_duration) /
                                     (n_samples_for_each_video - 1)))
            else:
                step = sample_duration
            for j in range(1, n_frames, step):
                sample_j = copy.deepcopy(sample)
                sample_j['frame_indices'] = list(
                    range(j, min(n_frames + 1, j + sample_duration)))
                dataset.append(sample_j)

    return dataset, idx_to_class


class IPN(data.Dataset):
    """
    Args:
        root (string): Root directory path.
        spatial_transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        temporal_transform (callable, optional): A function/transform that  takes in a list of frame indices
            and returns a transformed version
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an video given its path and frame indices.
     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        imgs (list): List of (image path, class_index) tuples
    """

    def __init__(self,
                 root_path,
                 annotation_path,
                 subset,
                 n_samples_for_each_video=1,
                 spatial_transform=None,
                 temporal_transform=None,
                 target_transform=None,
                 sample_duration=16,
                 modality='RGB',
                 get_loader=get_default_video_loader):
        assert subset in ["training", "validation"]
        self.data, self.class_names = make_dataset(
            root_path, annotation_path, subset, n_samples_for_each_video,
            sample_duration)

        self.spatial_transform = spatial_transform
        self.temporal_transform = temporal_transform
        self.target_transform = target_transform
        self.modality = modality
        self.sample_duration = sample_duration
        self.loader = get_loader()

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """

        path = self.data[index]['video']

        frame_indices = self.data[index]['frame_indices']


        if self.temporal_transform is not None:
            frame_indices = self.temporal_transform(frame_indices)
        clip = self.loader(path, frame_indices, self.modality, self.sample_duration)
        clip = torch.stack([T.ToTensor()(img) for img in clip])
        
        if self.spatial_transform is not None:
            clip = self.spatial_transform(clip)
     
        target = self.data[index]
        if self.target_transform is not None:
            target = self.target_transform(target)

        return clip, target

    def __len__(self):
        return len(self.data)


if __name__ == "__main__":
    import cv2
    from thanos.dataset_config import IPN_HAND_ROOT
    from thanos.dataset.temporal_transform import TemporalRandomCrop
    import os
    ann_path = os.path.join(IPN_HAND_ROOT, "annotations", "ipnall.json")
    ipn = IPN(IPN_HAND_ROOT, ann_path, "training", 
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
            