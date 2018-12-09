# lime-cam-pytorch
Pytorch implementation of LIME-CAM (LIME-based variant of CAM) and Grad-CAM


## Dependencies
* Python 2.7
* Numpy
* pytorch 0.4.0
* torchvision 0.2.1
* opencv
* sklearn


## Usage

```sh
python main.py --help
```

* ```--image_path```: a path to an image (required)
* ```--result_path```: a path to the explanation result (default: results/result_{method}_{class}.png)
* ```--model```: a model name from ```torchvision.models```, e.g., 'vgg16' (default: vgg16)
* ```--method```: a method to generate the explanation e.g., 'limecam, 'gradcam' (required)
* ```--no-cuda```: disables GPU usage


## References

\[1\] R. R. Selvaraju, A. Das, R. Vedantam, M. Cogswell, D. Parikh, and D. Batra. "Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization". arXiv, 2016<br>
\[2\] M. T. Riberio, S. Singh, and C. Guestrin. ""Why should I Trust You?": Explaining the Predictions of Any Classifier". arXiv, 2016
