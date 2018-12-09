import argparse

import numpy as np
import cv2

import torch
import torch.nn
import torch.nn.functional as F

from torchvision import models, transforms

from cam import *
from utils import *
    

parser = argparse.ArgumentParser(description='PyTorch code')
parser.add_argument('--image_path', required=True, help='path to image')
parser.add_argument('--result_path', default=None, help='path to explanation result')
parser.add_argument('--model', default='vgg16', help='currently only support vgg16')
parser.add_argument('--method', required=True, help='limecam | gradcam')
parser.add_argument('--no_cuda', action='store_true', default=False, help='disables GPU usage')



def main():
    
    args = parser.parse_args()
    
    device = torch.device('cuda' if not args.no_cuda and torch.cuda.is_available() else 'cpu')
    
    # load imagenet classes
    classes = list()
    with open('synset_words.txt') as lines:
        for line in lines:
            line = line.strip().split(' ', 1)[1]
            line = line.split(', ', 1)[0].replace(' ', '_')
            classes.append(line)
    
    # prepare model
    if args.model == 'vgg16':
        model = models.vgg16(pretrained=True)
    else:
        'Select proper model'
    model.to(device)
    model.eval()
    
    # prepare image
    image_path = args.image_path
    # image_path = 'images/sample.jpeg'
    raw_image = cv2.imread(image_path)[..., ::-1]
    input = preprocess_image(raw_image).to(device)
    probs = F.softmax(model(input), dim=1).detach().cpu()
    pred = probs.argmax().item()
    
    # get explanation
    if args.method == 'limecam':
        cam = limecam(model, input)
    elif args.method == 'gradcam':
        cam = gradcam(model, input)
    else:
        'Select proper explanation method'
        
    # apply cam to image
    result = apply_cam(raw_image, cam)
    
    # save result image
    if args.result_path is None:
        result_path = '../results/result_{}_{}.png'.format(args.method, classes[pred])
    else:
        result_path = args.result_path
    cv2.imwrite(result_path, result)
    print('[{:.5f}] {}'.format(probs[0, pred], classes[pred]))
        


if __name__ == '__main__':
    main()