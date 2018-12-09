import torch
import torch.nn
import torch.nn.functional as F

import torchvision.models as models



class vgg16(nn.Module):
    
    def __init__(self, num_classes=1000, pretrained=True, batch_norm=False):
        super(vgg16, self).__init__()
        
        if batch_norm:
            pretrained_model = models.vgg16_bn(pretrained)
        else:
            pretrained_model = models.vgg16(pretrained)
            
        self.features = nn.Sequential(*list(pretrained_model.features.children()))
        for param in self.parameters():
            param.requires_grad = False
        self.classifier = nn.Sequential(*list(pretrained_model.classifier.children()))
        
        
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
        self.preprocess = transforms.Compose([
           transforms.Resize((224,224)),
           transforms.ToTensor(),
           normalize
        ])

        
                
        
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    
    def extract_feature_from_image(self, image, target_layer=-2, cuda=True):
        """
        Extract feature at given layer for given image
        
        Args:
            image: PIL Image object
            
        Returns:
            A 3d numpy array (channel, height, width)
        """
        
        image_tensor = self.preprocess(image).unsqueeze(0)
        
        if cuda:
            image_tensor = image_tensor.cuda()
            
        if cuda:
            modules = list(self.features.module.children())[:target_layer+1]
        else:
            modules = list(self.features.children())[:target_layer+1]
        
        x = image_tensor
        for module in modules:
            x = module(x)
            
        if cuda:
            x = x.cpu()        
        
        return x.data.numpy()[0]


    def forward_from_features(self, features):
        """
        Forward from features

        Args:
            features: A 4d numpy array (batch_size, channel, height, width)

        Returns:
            output: A 2d numpy array (batch_size, num_classes)
        """

        feature_tensor = torch.Tensor(features).cuda()

        pooling = nn.MaxPool2d(kernel_size=2, stride=2)

        x = feature_tensor
        x = pooling(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)

        return x.data.cpu().numpy()


    def gradient_from_features(self, features, index):
        """
        Gradient from features

        Args:
            features: A 4d numpy array (batch_size, channel, height, width)

        Returns:
            gradients: A 4d numpy array
        """

        # pooling = nn.MaxPool2d(kernel_size=2, stride=2)

        grad_list = []
        def hook_grad_input(module, grad_input, grad_output):
            grad_list.append(grad_input[0].data.cpu().numpy())

        handle_backward = self.features.module[-1].register_backward_hook(hook_grad_input)

        x = torch.Tensor(features).cuda()
        x.requires_grad_()
        x = self.features.module[-1](x)
        x = x.view(x.size(0), -1)
        output = self.classifier(x)

        # one_hot = torch.zeros_like(output)
        # one_hot[:, index] = 1

        # output.backward(gradient=one_hot)
        output[:, index].sum().backward()
        
        handle_backward.remove()
        # del x, x1, x2, output, one_hot
        # del x, output, one_hot
        del x, output
        # torch.cuda.empty_cache()

        grad_x = grad_list[-1]

        return grad_x