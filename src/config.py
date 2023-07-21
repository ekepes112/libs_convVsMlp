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
    'mlp': {},
}
SHARED_MODEL_PARAMS = {
    'prediction_head_params':{
        'lambda_l1':0.,
        'lambda_l2':0.,
        'dropout_rate':0.,
    },
    'loss_func':'mse',
    'eval_metrics':[],
}
INITIAL_LEARNING_RATES = {
    'alexnet_training': 5e-5,
    'vgg_training': 3e-4,
    'resnet_bottleneck_training': 2.5e-6,
    'resnet_bn_training': 2e-5,
    'resnet_simple_training': 1e-5,
    'densenet_training': 3.5e-6,
    'mlp_13100_training': 4.5e-5,
    'mlp_6500_training': 5.5e-5,
    'mlp_3200_training': 6.6e-5,
    'mlp_1600_training': 8.1e-5,
}
OPTIMIZERS = {
    'alexnet_training': 'Adam_8',
    'vgg_training': 'Adam_9',
    'resnet_bottleneck_training': 'Adam_6',
    'resnet_bn_training': 'Adam_1',
    'resnet_simple_training': 'Adam_1',
    'densenet_training': 'Adam_5',
    'mlp_13100_training': 'Adam_4',
    'mlp_6500_training': 'Adam_9',
    'mlp_3200_training': 'Adam_5',
    'mlp_1600_training': 'Adam_5',
}

TRAIN_EPOCHS = 500
BATCH_SIZE = 64

TRAIN_FOLDS = [1, 2, 4, 5]
TEST_FOLD = 3
COMPOUND_LIST = ['SiO2', 'TiO2', 'Al2O3', 'FeOT', 'MgO', 'CaO', 'Na2O', 'K2O']
SHIFT_MAGNITUDES = [-3, -2, -1, 1, 2, 3]
NORMALIZE_TO_UNIT_MAXIMUM = False
