import numpy as np
import cv2

import torch
import torch.nn
import torch.nn.functional as F

from sklearn.linear_model import Ridge

def generate_cam(feature, weight):
    """
    Args:
        feature: 3d numpy array of size (C, H, W)
        weight: 1d numpy array of size (C, )
    
    Returns:
        cam: 2d numpy array of size (H, W)
    """
    
    cam = np.zeros_like(feature[0])
    for c in range(feature.shape[0]):
        cam += feature[c] * weight[c]
    cam = np.maximum(cam, 0)
    cam /= np.max(cam)+1e-5
    
    return cam


def gradcam(model, image):
    """
    Args:
        model: torch nn.Module consists of feature extractor(model.features) and classifier(model.classifier)
        image: preprocessed 4d torch tensor of the shape (1, C, H, W)
        
    Returns:
        cam: 2d numpy array of size (H, W)
    """
    
    feature = model.features(image)
    output = model.classifier(feature.view(1, -1))
    
    max_idx = output.argmax().item()
    
    grad = torch.autograd.grad(output[0, max_idx], feature)[0].squeeze()
    grad_weight = grad.view(grad.size(0), -1).sum(dim=1)
    
    feature_np = feature.squeeze().detach().cpu().numpy()
    grad_np = grad_weight.cpu().numpy()
    
    return generate_cam(feature_np, grad_np)


def limecam(model, image):
    """
    Args:
        model: torch nn.Module consists of feature extractor(model.features) and classifier(model.classifier)
        image: preprocessed 4d torch tensor of the shape (1, C, H, W)
        
    Returns:
        cam: 2d numpy array of size (H, W)
    """
    num_samples = 1024 # num_samples should be larger than num_channels
    num_channels = 512
    
    device = image.device.type
    
    feature = model.features(image)
    feature_np = feature.squeeze().detach().cpu().numpy()
    
    data = np.random.binomial(1, 0.5, (num_samples, num_channels))
    data[0, :] = 1
    
    features = []
    for row in data:
        temp = np.array(feature_np)
        zeros = np.where(row == 0)[0]
        mask = np.zeros(feature_np.shape[1:])
        for z in zeros:
            temp[z] = mask
        features.append(temp)
    features = np.array(features)
    
    output = model.classifier(torch.Tensor(features).to(device).view(num_samples, -1))
    max_idx = output[0].argmax().item()
    
    target = output.detach().cpu().numpy()[:, max_idx]
    
    expl_model = Ridge()
    expl_model.fit(data, target)
    
    return generate_cam(feature_np, expl_model.coef_)