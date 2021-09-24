"""
Because the IPN Dataset downloaded from their official page
contains resized frames with half of the original resolution.
This script will export frames from videos with original resolution 640x480
"""
from typing import List
import os
import cv2
import numpy as np
from dataset_config import IPN_HAND_ROOT

def get_video_names(dataset_root:str=IPN_HAND_ROOT) -> List[str]:
    video_dir = os.path.join(dataset_root, "videos")
    video_names = [name for name in os.listdir(video_dir) if name.endswith(".avi")]
    return video_names

def export_frames_from_video(video_path:str, save_to:str, prefix_name:str):
    cap = cv2.VideoCapture(video_path)
    i = 1
    print("Export frame for video: ", video_path)
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            print(f"Frame {i:06d}", end="\r")
            image_name = f"{prefix_name}_{i:06d}.jpeg"
            image_path = os.path.join(save_to, image_name)
            cv2.imwrite(image_path, frame)
            i += 1
        else:
            break
    cap.release()
    print(f"Done. Exported {i} frames")

if __name__ == "__main__":
    dataset_root = IPN_HAND_ROOT
    frame_dir = os.path.join(dataset_root, "frames")
    if not os.path.isdir(frame_dir):
        os.mkdir(frame_dir)
    video_names = get_video_names(dataset_root)
    for video_name in video_names:
        name = video_name.split(".")[0]
        video_path = os.path.join(dataset_root, "videos", video_name)
        sequence_dir = os.path.join(frame_dir, name)
        if not os.path.isdir(sequence_dir):
            os.mkdir(sequence_dir)
        export_frames_from_video(video_path, sequence_dir, name)
    print(f"Exported frames for {len(video_names)} videos")