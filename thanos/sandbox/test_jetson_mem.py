import cv2
import pycuda.autoinit
import pycuda.driver as cuda
import tensorrt as trt
import time
import numpy as np


if __name__ == "__main__":
    host_ft_maps = cuda.pagelocked_zeros((22, 512),np.float32)
    device_ft_maps = cuda.mem_alloc(host_ft_maps.nbytes)
    try:
        while True:
            host_ft_map = np.random.rand(512).astype(np.float32)

            tic = time.time()
            cuda.memcpy_dtoh(host_ft_maps, device_ft_maps)
            host_ft_maps = np.roll(host_ft_maps, 1, axis=0)
            host_ft_maps[0] = host_ft_map
            cuda.memcpy_htod(device_ft_maps, host_ft_maps)
            toc = time.time()
            

            print((toc - tic)*1000, "ms")
    except KeyboardInterrupt:
        print("Keyboard Interrupt")
    
