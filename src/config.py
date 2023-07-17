PROJECT_NAME = "convVsMlp"
DRIVE_PATH = '/content/onedrive'
RESULTS_PATH = '/content/gdrive/My Drive/projects/convVsMlp/temp'
DATA_PATH = '.archive/datasets/'
DATASET_TYPE = 'means'
CHECKPOINT_DIR = '/content/checkpoints'

MODEL_NAMES = ['alexnet', 'vgg', 'resnet_bottleneck',
               'resnet_simple', 'densenet', 'resnet_bn']
MODEL_PARAMS = {
    'alexnet': {},
    'vgg': {},
    'resnet_bottleneck': {},
    'resnet_bn': {},
    'resnet_simple': {},
    'densenet': {},
}
SHARED_MODEL_PARAMS = {
    'prediction_head_params':{
        'lambda_l1':0.,
        'lambda_l2':0.,
        'dropout_rate':.4,
    },
    'loss_func':'mse',
    'eval_metrics':[],
}
INITIAL_LEARNING_RATES = {
    'alexnet': 5e-5,
    'vgg': 3e-4,
    'resnet_bottleneck': 2.5e-6,
    'resnet_bn': 2e-5,
    'resnet_simple': 1e-5,
    'densenet': 3.5e-6,
}
OPTIMIZERS = {
    'alexnet': 'Adam_8',
    'vgg': 'Adam_9',
    'resnet_bottleneck': 'Adam_6',
    'resnet_bn': 'Adam_1',
    'resnet_simple': 'Adam_1',
    'densenet': 'Adam_5',
}

TRAIN_EPOCHS = 500

TRAIN_FOLDS = [1, 2, 4, 5]
TEST_FOLD = 3
COMPOUND_LIST = ['SiO2', 'TiO2', 'Al2O3', 'FeOT', 'MgO', 'CaO', 'Na2O', 'K2O']
SHIFT_MAGNITUDES = [-3, -2, -1, 1, 2, 3]
NORMALIZE_TO_UNIT_MAXIMUM = False
