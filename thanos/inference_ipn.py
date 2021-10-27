import argparse
import os
import time
import cv2
import numpy as np
from thanos.dataset.target_transform import read_label_from_target_dict

from thanos.tensorrt_inference import TRTGestureTransformer
from thanos.tensorrt_inference.utils import draw_result_on_frame

from thanos.dataset import IPN, IPN_HAND_ROOT, one_hot_label_transform, INPUT_MEAN, INPUT_STD
from thanos.trainers.data_augmentation import (get_temporal_transform_fn,
                                               get_val_spatial_transform_fn)


def gstreamer_sink(ip, port=5000):
    return f"appsrc ! video/x-raw, format=BGR ! queue \
        ! videoconvert ! video/x-raw,format=BGRx ! nvvidconv \
        ! omxh264enc insert-sps-pps=true ! h264parse ! rtph264pay pt=96 \
        ! queue ! application/x-rtp, media=video, encoding-name=H264 \
        ! udpsink host={ip} port={port} sync=false"

def denormalize_image(image: np.ndarray, mean=INPUT_MEAN, std=INPUT_STD):
    """De-normalize image

    Parameters
    ----------
    image: np.ndarray
        normalized image by mean and std, dtype float32, shape (H, W, 3)
    mean: tuple
    std: tuple

    Returns
    -------
    np.ndarray
        de-normalized image, dtype uint8
    """
    image = image * np.array(std) + np.array(mean)
    image = image * 255
    return image.astype(np.uint8)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("engine_dir", type=str, help="Directory containing TensorRT engine")
    parser.add_argument("model_name", type=str, help="Name prefix for engines to be loaded")
    parser.add_argument("--temporal_stride", type=int, default=2, help="Model temporal stride")
    parser.add_argument("--rtp_client_ip", type=str, default=None, help="RTP Client IP address")
    parser.add_argument("--rtp_port", type=int, default=5000, help="RTP Stream port")
    parser.add_argument("--no_show", action="store_true", help="If set, result will not be shown")
    args = parser.parse_args()

    # === Load model
    backbone_path = os.path.join(args.engine_dir, args.model_name + "_backbone_fp16.trt") 
    encoder_path = os.path.join(args.engine_dir, args.model_name + "_encoder_fp16.trt")
    model = TRTGestureTransformer(backbone_path, encoder_path, normalize_image=False)

    # === RTP Stream
    fps = 30 // args.temporal_stride
    frame_width, frame_height = 240, 240
    if args.rtp_client_ip is not None:
        rtp_writer = cv2.VideoWriter(
            gstreamer_sink(args.rtp_client_ip, args.rtp_port), 
            cv2.CAP_GSTREAMER, 0, float(fps), (frame_width, frame_height))
        if not rtp_writer.isOpened() :
            print("RTP Writer failed")
            exit()
        print('RTP Writer opened')
    else:
        rtp_writer = None

    # === IPN validation dataset
    ann_path = os.path.join(IPN_HAND_ROOT, "annotations", "ipnall.json")
    val_dataset = IPN(IPN_HAND_ROOT, ann_path, "validation",
            temporal_stride=args.temporal_stride,
            spatial_transform=get_val_spatial_transform_fn(), 
            target_transform=read_label_from_target_dict)

    # === Inference loop
    try:
        for sequences, labels in val_dataset:
            for i in range(sequences.shape[0]):
                np_img = sequences[i].permute(1, 2, 0).numpy()
                np_img = np.ascontiguousarray(np_img)
                frame = np_img.copy()
                frame = denormalize_image(frame)
                gesture_id = model(np_img)
                print("Predict/True", gesture_id, labels)
                draw_result_on_frame(frame, gesture_id)
                # # print(model.encoder.outputs[0].host)
                if rtp_writer is not None:
                    rtp_writer.write(frame)
                if not args.no_show:
                    cv2.imshow("frame", frame)
                    cv2.waitKey(1)
    except KeyboardInterrupt:
        print("Keyboard Interrupt detected")
    finally:
        cv2.destroyAllWindows()
        if rtp_writer is not None:
            rtp_writer.release()

