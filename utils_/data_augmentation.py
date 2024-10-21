import cv2
import numpy as np


def augment_image(or_image: np.ndarray, samples: int=50, output_size: int=256, save_as: str=""):    
    augmented_images = []
    for i in range(samples):
        image = np.transpose(or_image, (1, 2, 0))

        # Contrast adjustment
        contract_factor = np.random.random() * 3 +.2
        image = np.clip((image - 0.5) * contract_factor + 0.5, 0, 1)

        # # Color channel swap
        channel_var = np.random.random()
        r = np.copy(image[:, :, 0])
        g = np.copy(image[:, :, 1])
        b = np.copy(image[:, :, 2])
        if channel_var > .6:
            image[:, :, 0] = b
            image[:, :, 1] = r
            image[:, :, 2] = g
        elif channel_var > .3:
            image[:, :, 0] = g
            image[:, :, 1] = b
            image[:, :, 2] = r

        # Cropping
        x = .2 * np.random.random()
        y = .2 * np.random.random()
        w = .1 * np.random.random() + .9 -x
        h = .1 * np.random.random() + .9 -y
        x = int(x*output_size)
        y = int(y*output_size)
        w = int(w*output_size)
        h = int(h*output_size)
        image = image[y:y+h, x:x+w]

        # Rotate image
        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        angle = 360 * np.random.random()
        scale = 1.0
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, scale)
        image = cv2.warpAffine(image, rotation_matrix, (w, h))

        # Gaussian noise
        noise = np.random.randn(*image.shape) * .05 * np.random.random()
        image = image + noise

        image = np.clip(image * 255.0, 0, 255).astype(np.uint8)
        image = cv2.resize(image, (output_size, output_size))

        if len(save_as):
            cv2.imwrite(f"./augmented/{save_as}_{i:04}.jpg", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        
        image = np.transpose(image, (2, 0, 1))
        augmented_images.append(np.copy(image))

    augmented_images = np.array(augmented_images)
    return augmented_images
