import argparse
import os
import time
import cv2
import numpy as np

from thanos.dataset import IPN, IPN_HAND_ROOT, one_hot_label_transform, INPUT_MEAN, INPUT_STD
from thanos.tensorrt_inference import TRTGestureTransformer
from thanos.tensorrt_inference.utils import draw_result_on_frame, draw_fps_on_frame

def gstreamer_pipeline(
    capture_width=640,
    capture_height=360,
    display_width=640,
    display_height=360,
    framerate=30,
    flip_method=6,
):
    return (
        "nvarguscamerasrc ! "
        "video/x-raw(memory:NVMM), "
        "width=(int)%d, height=(int)%d, "
        "format=(string)NV12, framerate=(fraction)%d/1 ! "
        "nvvidconv flip-method=%d ! "
        "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! appsink"
        % (
            capture_width,
            capture_height,
            framerate,
            flip_method,
            display_width,
            display_height,
        )
    )

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
    model = TRTGestureTransformer(backbone_path, encoder_path)

    # === Camera Capture
    cap = cv2.VideoCapture(gstreamer_pipeline(), cv2.CAP_GSTREAMER)
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(fps)
    frame_width, frame_height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if not cap.isOpened():
        print("Cannot open camera. Exiting")
        exit()
    print("Camera opened")

    # === RTP Stream
    if args.rtp_client_ip is not None:
        rtp_writer = cv2.VideoWriter(gstreamer_sink(args.rtp_client_ip, args.rtp_port), cv2.CAP_GSTREAMER, 0, float(fps), (frame_width, frame_height))
        if not rtp_writer.isOpened() :
            print("RTP Writer failed")
            exit()
        print('RTP Writer opened')
    else:
        rtp_writer = None

    # === Inference loop
    try:
        while True:
            tic = time.time()
            _, frame = cap.read()
            m_input = frame.copy()
            m_input = cv2.cvtColor(m_input, cv2.COLOR_BGR2RGB)
            gesture_id = model(m_input)
            toc = time.time()
            draw_result_on_frame(frame, gesture_id)
            draw_fps_on_frame(frame, int(1/(toc - tic)))
            if rtp_writer is not None:
                rtp_writer.write(frame)
            if not args.no_show:
                cv2.imshow("frame", frame)
                cv2.waitKey(1)
            print((toc - tic)*1000, "ms", end="\r")
    except KeyboardInterrupt:
        print("Keyboard Interrupt detected")
    finally:
        cv2.destroyAllWindows()
        cap.release()
        if rtp_writer is not None:
            rtp_writer.release()

