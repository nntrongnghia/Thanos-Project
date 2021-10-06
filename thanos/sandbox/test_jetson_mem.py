import torch
import cv2
import time
import numpy as np

def gstreamer_pipeline(
    capture_width=1280,
    capture_height=720,
    display_width=1280,
    display_height=720,
    framerate=60,
    flip_method=0,
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

# @profile
def roll_new_frame(time_frames, new_frame, permute=None):
    if not isinstance(new_frame, torch.Tensor):
        new_frame = new_frame.astype(np.float32)
        new_frame = torch.tensor(new_frame, device=time_frames.device)
    if permute is not None:
        new_frame = new_frame.permute(permute)

    time_frames.roll(1, 0)
    time_frames[0] = new_frame 

    # time_frames = torch.cat([new_frame[None], time_frames[1:]])

    return time_frames


if __name__ == "__main__":
    with torch.no_grad():
        time_frames = torch.rand((24, 3, 480, 640), device=torch.device("cuda")) # jetson limit
        # time_frames = torch.rand((40, 3, 480, 640))
        cap = cv2.VideoCapture(
            gstreamer_pipeline(
                capture_width=640, 
                capture_height=480, 
                display_width=640, 
                display_height=480,
                framerate=30, 
                flip_method=0), 
            cv2.CAP_GSTREAMER)
        if not cap.isOpened():
            print("Cannot open camera. Exiting")
            exit()
        print('Camera opened')

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                tic = time.time()
                time_frames = roll_new_frame(time_frames, frame, permute=(2, 0, 1))
                toc = time.time()
                print("torch.tensor + .cuda() + roll + assigment", (toc - tic)*1000, "ms")
        except KeyboardInterrupt:
            print("Keyboard Interrupt")
        finally:
            cap.release()
    
