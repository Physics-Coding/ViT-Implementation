# Vision Transformer

## Introduction

![ViT.png](https://s2.loli.net/2022/01/19/w3CyXNrhEeI7xOF.png)

Network for Vision Transformer. The pytorch version. 

## Quick start

1.Clone this repository

```shell
git clone https://github.com/Physics-Coding/Vit-Implementation.git
```
2.Install Vit-Implementation from source.

```shell
cd Vit-Implementation
pip install -r requirements.txt
```
3.Modifying the [config.py](https://github.com/Runist/torch_Vision_Transformer/blob/master/config.py).

4.Start train your model.

```shell
python train.py
```
5.Open tensorboard to watch loss, learning rate etc. You can also see training process and training process and validation prediction.

```shell
tensorboard --logdir ./summary/log
```
![tensorboard.png](https://s2.loli.net/2022/10/12/p7KtB1uXMkqvreN.png)

6.test the model.

```shell
python predict.py --your_model_path
```

## Reference

Appreciate the work from the following repositories:

- [WZMIAOMIAO](https://github.com/WZMIAOMIAO)/[vision_transformer](https://github.com/WZMIAOMIAO/deep-learning-for-image-processing/tree/master/pytorch_classification/vision_transformer)


## License

Code and datasets are released for non-commercial and research purposes **only**. For commercial purposes, please contact the authors.
