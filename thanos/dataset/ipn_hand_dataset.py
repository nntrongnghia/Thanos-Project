from typing import Dict, List, Tuple
import os
import torch
import cv2
import copy
import numpy as np
import pandas as pd
from thanos.dataset_config import IPN_HAND_ROOT

class IPNHandDataset():
    """
    IPN Hand Dataset: https://gibranbenitez.github.io/IPN_Hand/
    """

    LABEL2ID = {
        "D0X": 1,
        "B0A": 2,
        "B0B": 3,
        "G01": 4,
        "G02": 5,
        "G03": 6,
        "G04": 7,
        "G05": 8,
        "G06": 9,
        "G07": 10,
        "G08": 11,
        "G09": 12,
        "G10": 13,
        "G11": 14,
    }

    ID2LABEL = {
        1: "D0X",
        2: "B0A",
        3: "B0B",
        4: "G01",
        5: "G02",
        6: "G03",
        7: "G04",
        8: "G05",
        9: "G06",
        10: "G07",
        11: "G08",
        12: "G09",
        13: "G10",
        14: "G11"
    }

    LABEL_NAMES = {
        1: "Non-gesture",
        2: "Pointing with one finger",
        3: "Pointing with two fingers",
        4: "Click with one finger",
        5: "Click with two fingers",
        6: "Throw up",
        7: "Throw down",
        8: "Throw left",
        9: "Throw right",
        10: "Open twice",
        11: "Double click with one finger",
        12: "Double click with two fingers",
        13: "Zoom in",
        14: "Zoom out"
    }


    def _read_df_annotation(self) -> pd.DataFrame:
        """Read annotations from dataset root

        Return a DataFrame with columns:
        - video: video name
        - label: gesture label
        - id: gesture id
        - t_start: start frame
        - t_end: end frame
        - frames: gesture duration in frames
        """
        annotation_path = os.path.join(self.root, "annotations", "Annot_List.txt")
        df_anns = pd.read_csv(annotation_path)
        return df_anns

    def _get_all_train_test_names(self) -> Tuple[List[str], List[str]]:
        """Get train/test split given by the authors of IPN Hand Dataset

        Returns tuple of list of string containing sequence names for train/test
        """
        sequence_train_path = os.path.join(self.root, "annotations", "Video_TrainList.txt")
        sequence_test_path = os.path.join(self.root, "annotations", "Video_TestList.txt")
        sequence_train = pd.read_csv(sequence_train_path, sep="\t", header=None)[0].tolist()
        sequence_test = pd.read_csv(sequence_test_path, sep="\t", header=None)[0].tolist()
        return sequence_train, sequence_test

    def _get_train_test_split(self) -> Tuple[Dict[str,int], Dict[str,int]]:
        """Get train/test sequence names and its length in `frames` directory

        Return a tuple of dict for train/test with key = sequence name, value = length in frame number
        """
        train_dict = {}
        test_dict = {}
        all_train_sequences, all_test_sequences = self._get_all_train_test_names()
        frame_dir = os.path.join(self.root, "frames")
        sequence_names = os.listdir(frame_dir)
        for sequence_name in sequence_names:
            sequence_dir = os.path.join(frame_dir, sequence_name)
            frame_count = len([n for n in os.listdir(sequence_dir) if n.endswith(".jpeg")])
            if sequence_name in all_train_sequences:
                train_dict[sequence_name] = frame_count
            elif sequence_name in all_test_sequences:
                test_dict[sequence_name] = frame_count
        return train_dict, test_dict

    def __init__(self, root:str=IPN_HAND_ROOT):
        """Init the dataset
        """
        self.root = root
        self.df_anns = self._read_df_annotation()
        # dict of sequence names and its length in frame number
        self.train_dict, self.test_dict = self._get_train_test_split()
        # concat train_dict and test_dict
        self.sequence_dict = self.train_dict.copy()
        self.sequence_dict.update(self.test_dict)
        self.sequence_names = list(self.sequence_dict.keys())


    def get_sequence(self, sequence:str):
        pass

    def __len__(self):
        """Number of sequences"""
        return len(self.sequence_names)

# test script
if __name__ == "__main__":
    dataset = IPNHandDataset()
    print(len(dataset))
