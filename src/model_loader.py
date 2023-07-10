from model_def_alexnet import AlexNet1D

models = {
    'alexnet': AlexNet1D(),
    'vgg': VGG1D(),
    'resnet_bottleneck': ResNetBottleneck1D(),
    'resnet_bn': ResNetBatchNorm1D(),
    'resnet_simple': ResNetSimple1D(),
    'densenet': DenseNet1D(),
}