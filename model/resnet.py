import torch
import torchvision
import torch.nn as nn

from utils import count_parameters

if __name__ == "__main__":
    import cv2
    import time

    model = torchvision.models.resnet18()
    print(count_parameters(model))

    cap = cv2.VideoCapture(0)
    while True:
        _, np_img = cap.read()
        # np_img = cv2.resize(np_img, [np_img.shape[1]//2, np_img.shape[0]//2])
        img = torch.tensor(np_img).to(torch.float).permute([2, 0, 1])[None]

        t = time.time()
        ft_map = model(img)
        print(ft_map.shape)
        print((time.time() - t) * 1000, "ms")


        cv2.imshow("img", np_img)
        if cv2.waitKey(1) == 27: # ESC
            break

    cap.release()
    cv2.destroyAllWindows()
