from typing import List, Tuple
import os
import torch
import cv2
import numpy as np
import pandas as pd
from dataset_config import IPN_HAND_ROOT

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

    @staticmethod
    def convert_videos_to_frame(root):
        video_dir = os.path.join(root, "videos")
        frame_dis = os.path.join(root, "frames")

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

    def _get_train_test_sequences(self) -> Tuple[List[str], List[str]]:
        """Get train/test split given by the authors of IPN Hand Dataset

        Returns tuple of pd.DataFrame for train and test with columns:
        - video: video name (str)
        - length: length in number of frames (int)
        """
        video_train_path = os.path.join(self.root, "annotations", "Video_TrainList.txt")
        video_test_path = os.path.join(self.root, "annotations", "Video_TestList.txt")
        df_video_train = pd.read_csv(video_train_path, sep="\t", header=None)[0]
        df_video_test = pd.read_csv(video_test_path, sep="\t", header=None)[0]
        return df_video_train, df_video_test

    def __init__(self, root:str=IPN_HAND_ROOT, sequences:List[str]=None):
        """Init the dataset
        """
        self.root = root
        self.df_anns = self._read_df_annotation()
        self.df_train_videos, self.df_test_videos = self._get_train_test_sequences()

    def get_sequence(self, sequence:str):
        pass


# test script
if __name__ == "__main__":
    # dataset = IPNHandDataset()
    pass
