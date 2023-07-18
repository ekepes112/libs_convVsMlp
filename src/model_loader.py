from model_def_alexnet import AlexNet1D
from model_def_densenet import DenseNet1D
from model_def_vgg import VGG1D
from model_def_resnet_bn import ResNetBatchNorm1D
from model_def_resnet_bottleneck import ResNetBottleneck1D
from model_def_resnet_simple import ResNetSimple1D
from model_def_mlp_equivalent import MLP

models = {
    'alexnet': AlexNet1D,
    'vgg': VGG1D,
    'resnet_bottleneck': ResNetBottleneck1D,
    'resnet_bn': ResNetBatchNorm1D,
    'resnet_simple': ResNetSimple1D,
    'densenet': DenseNet1D,
    'mlp': MLP
}
