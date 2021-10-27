import os
import argparse
import numpy as np
import cv2
from thanos.tensorrt_inference import TRTExecutor
from thanos.dataset import INPUT_MEAN, INPUT_STD

class TRTGestureTransformer():
    INPUT_MEAN = np.array(INPUT_MEAN, dtype=np.float32)
    INPUT_STD = np.array(INPUT_STD, dtype=np.float32)
    def __init__(self, backbone_path, encoder_path, normalize_image=True):
        """
        Parameters
        ----------
        backbone_path: str
            path to TRT engine for Resnet backbone
        encoder_path: str
            path to TRT engine for encoder
        temporal_length: int
            number sequences for encoder input
        input_size: tuple of int
            image size for Resnet input
        """
        self.backbone = TRTExecutor(backbone_path)
        self.encoder = TRTExecutor(encoder_path)
        self.backbone.print_bindings_info()
        self.encoder.print_bindings_info()
        self.input_size = self.backbone.inputs[0].shape[-2:]
        assert self.input_size[0] == self.input_size[1]
        self.seq_len = self.encoder.inputs[0].shape[1]
        self.encoder_dim = self.encoder.inputs[0].shape[-1]
        self.normalize_image = normalize_image

    def preprocess_image(self, img):
        """
        1. Resize and center crop image
        2. Normalize image
        3. Transpose image to (C, H, W)

        Parameters
        ----------
        img: np.ndarray
            dtype uint8, shape (H, W, 3)
        
        Returns
        -------
        np.ndarray
            dtype float32, shape (3, H, W)
        """
        # 1. Resize
        scale = self.input_size[0]/img.shape[0]
        size = (int(img.shape[1]*scale), int(img.shape[0]*scale))
        img = cv2.resize(img, size)
        # 1. Center crop
        center_x = img.shape[1] // 2
        x = center_x - self.input_size[1]//2
        img = img[:, x:x + self.input_size[1]]
        # 2. Normalize image
        if self.normalize_image:
            img = img.astype(np.float32)/255.0
            img = (img - self.INPUT_MEAN)/self.INPUT_STD
        # 3. Transpose image
        img = np.transpose(img, (2, 0, 1))
        img = np.ascontiguousarray(img)
        return img

    def process_frame(self, img):
        """Inference 1 image by backbone, 
        Output will be store in memory buffer of self.backbone.outputs
        """
        assert isinstance(img, np.ndarray)
        assert img.shape[-2] == self.input_size[0], f"{img.shape[-2]} {self.input_size[0]}"
        assert img.shape[-1] == self.input_size[1], f"{img.shape[-1]} {self.input_size[1]}"
        self.backbone(img)

    def update_sequence_ft_vectors(self):
        """Append output of resnet to self.encoder.inputs sequence in memory
        """
        new_ft_vector = self.backbone.outputs[0].host.reshape(self.encoder_dim) # (1, encoder_dim)
        new_sequence = np.roll(
            self.encoder.inputs[0].host.reshape(self.seq_len, self.encoder_dim), 
            shift=1, 
            axis=0) # (22, 512)
        new_sequence[0] = new_ft_vector
        self.encoder.inputs[0].host = new_sequence

    def process_sequence(self):
        """Inference sequences of feature vector (output of backbone)
        Input is already stored in memory by self.
        """
        m_outputs = self.encoder.execute()
        return m_outputs["logits"]

    

    def __call__(self, img):
        """
        Parameters
        -----------
        img: np.ndarray
            input image, dtype uint8, shape (H, W, 3)

        Returns
        -------
        int
            Predicted gesture id
        """
        img = self.preprocess_image(img)
        self.process_frame(img)
        self.backbone.stream.synchronize()
        self.update_sequence_ft_vectors()
        logits = self.process_sequence()
        self.encoder.stream.synchronize()
        return np.argmax(logits[0])


if __name__ == "__main__":
    import time

    backbone_path = "weights/baseline_backbone_fp16.trt"
    encoder_path = "weights/baseline_encoder_fp16.trt"

    model = TRTGestureTransformer(backbone_path, encoder_path)
    img = np.random.rand(360, 640, 3)

    while True:
        t = time.time()
        model(img)
        latency = time.time() - t
        print(latency * 1000, "ms")