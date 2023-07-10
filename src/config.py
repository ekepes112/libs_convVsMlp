PROJECT_NAME = "convVsMlp"
DATASET_TYPE = 'means'

MODEL_NAMES = ['alexnet', 'vgg', 'resnet_bottleneck', 'resnet_simple', 'densenet', 'resnet_bn']
MODEL_PARAMS = {
    'alexnet': {},
    'vgg': {},
    'resnet_bottleneck': {},
    'resnet_bn': {},
    'resnet_simple': {},
    'densenet': {},
}

LR_SCAN_END = 1e-1
LR_SCAN_START = 1e-8
LR_SCAN_STEP_SIZE = .2
LR_SCAN_WARMUP = 10
BATCH_SIZE = 64

TRAIN_FOLDS = [1,2,4,5]
TEST_FOLDS = 3
COMPOUND_LIST = ['SiO2','TiO2','Al2O3','FeOT','MgO','CaO','Na2O','K2O']
SHIFT_MAGNITUDES = [-3,-2,-1,1,2,3]