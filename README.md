# lime-cam-pytorch
Pytorch implementation of LIME-CAM (LIME-based variant of CAM) and Grad-CAM


## Usage

```sh
python main.py --help
```

* ```--image_path```: a path to an image (required)
* ```--result_path```: a path to the explanation result (default: results/result_{method}_{class}.png)
* ```--model```: a model name from ```torchvision.models```, e.g., 'vgg16' (default: vgg16)
* ```--no-cuda```: disables GPU usage
