import numpy as np
import cv2

import torch

from torchvision import transforms

def preprocess_image(image):
    
    """
    Args:
        image: numpy array of the shape (H, W, C)
        
    Returns:
        4d torch tensor of the shape (1, C, H, W)
    """
    
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    
    preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
    
    image_resized = cv2.resize(image, (224, 224))
    
    return preprocess(image_resized).unsqueeze(0)



def apply_cam(raw_image, cam):
    
    """
    Args:
        image: numpy array of the shape (H, W, C) in RGB order (compatible to torch)
        cam: numpy array of the shape (H_c, W_c)
        
    Returns:
        result: numpy array of the shape (H, W, C) in BGR order (compatible to cv2)
    """
    
    h, w, _ = raw_image.shape
    cam_resized = cv2.resize(cam, (w, h))
    cam_resized = cv2.applyColorMap(np.uint8(cam_resized * 255.0), cv2.COLORMAP_JET)
    result = cam_resized.astype(np.float) + raw_image[..., ::-1].astype(np.float)
    result = result / result.max() * 255.0
    
    return np.uint8(result)