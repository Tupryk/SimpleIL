import os
import cv2
import numpy as np


def images_to_np_array(paths=["./scene_images"], image_dim=256, lim=np.nan):
    data = []
    count = 0
    for path in paths:
        for im_file in os.listdir(path):
            count += 1
            image = cv2.imread(f"{path}/{im_file}")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, (image_dim, image_dim))
            image = image/255.0
            image = np.transpose(image, (2, 0, 1))  # HWC to CHW
            data.append(image)
            if count >= lim:
                data = np.array(data)
                return data

    data = np.array(data)
    return data


def images_to_np_array_from_im_paths(im_paths, image_dim=256):
    data = []
    for im_path in im_paths:
        image = cv2.imread(im_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (image_dim, image_dim))
        image = image/255.0
        image = np.transpose(image, (2, 0, 1))  # HWC to CHW
        data.append(image)

    data = np.array(data)
    return data


def images_to_np_array_numbered(path, image_dim=256, format=".jpg"):
    data = []
    count = len(os.listdir(path))
    for i in range(count):
        image = cv2.imread(f"{path}/cam1_{i+1}{format}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (image_dim, image_dim))
        image = image/255.0
        image = np.transpose(image, (2, 0, 1))  # HWC to CHW
        data.append(image)

    data = np.array(data)
    return data
